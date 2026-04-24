"""Pure helpers for low-Neff pseudoMSA benchmark sweeps."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


RESULT_FIELDS = [
    "protein_id",
    "strategy",
    "variant_param",
    "cache_implementation",
    "target_n",
    "candidate_n_requested",
    "raw_candidates",
    "valid_candidates",
    "selected_sequences",
    "neff",
    "gen_time_s",
    "peak_vram_gb",
    "selection_time_s",
    "ptm",
    "mean_plddt",
    "rmsd",
    "tm_score",
    "status",
    "error",
]

SUMMARY_FIELDS = ["objective", *RESULT_FIELDS]


@dataclass(frozen=True)
class SamplingVariant:
    strategy: str
    variant_param: str
    cache_implementation: str | None
    decode_conf: dict[str, Any]
    extra: dict[str, Any]


@dataclass(frozen=True)
class FilterResult:
    sequences: list[str]
    raw_count: int
    valid_count: int


@dataclass(frozen=True)
class Candidate:
    sequences: list[str]
    selected_count: int
    neff: float | None = None


def normalize_cache_implementations(
    cache_implementations: str | Iterable[str | None] | None,
) -> list[str | None]:
    if cache_implementations is None:
        return [None]
    if isinstance(cache_implementations, str):
        return [cache_implementations]
    normalized = list(cache_implementations)
    return normalized or [None]


def generate_temperature_variants(
    cache_implementations: str | Iterable[str | None] | None,
) -> Iterable[SamplingVariant]:
    base_conf = {"repetition_penalty": 1.15}
    for cache_implementation in normalize_cache_implementations(cache_implementations):
        for temperature in (0.3, 0.5, 0.7, 0.9):
            for top_k in (1, 3, 5, 10):
                for top_p in (0.70, 0.80, 0.90):
                    decode_conf = {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        **base_conf,
                    }
                    yield SamplingVariant(
                        strategy="temperature_low_neff",
                        variant_param=(
                            f"temperature={temperature},top_k={top_k},top_p={top_p:.2f}"
                        ),
                        cache_implementation=cache_implementation,
                        decode_conf=decode_conf,
                        extra={},
                    )

        for knob, value in (
            ("min_p", 0.05),
            ("typical_p", 0.80),
            ("eta_cutoff", 6e-4),
            ("epsilon_cutoff", 6e-4),
        ):
            decode_conf = {
                "temperature": 0.5,
                "top_k": 5,
                "top_p": 0.80,
                **base_conf,
                knob: value,
            }
            yield SamplingVariant(
                strategy="temperature_low_neff",
                variant_param=(
                    f"temperature=0.5,top_k=5,top_p=0.80,{knob}={value}"
                ),
                cache_implementation=cache_implementation,
                decode_conf=decode_conf,
                extra={},
            )


def generate_embedding_variants(
    cache_implementations: str | Iterable[str | None] | None,
) -> Iterable[SamplingVariant]:
    for cache_implementation in normalize_cache_implementations(cache_implementations):
        for sigma in (0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10):
            yield SamplingVariant(
                strategy="embedding_walk_low_neff",
                variant_param=f"sigma={sigma}",
                cache_implementation=cache_implementation,
                decode_conf={
                    "temperature": 0.5,
                    "top_k": 5,
                    "top_p": 0.85,
                    "repetition_penalty": 1.15,
                },
                extra={"sigma": sigma, "depth_decay": 0.8},
            )


def _dedupe_ordered(sequences: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for sequence in sequences:
        if sequence in seen:
            continue
        seen.add(sequence)
        deduped.append(sequence)
    return deduped


def filter_exact_length(
    sequences: Iterable[str],
    length: int,
    *,
    dedupe: bool = True,
) -> FilterResult:
    all_sequences = list(sequences)
    valid_sequences = [sequence for sequence in all_sequences if len(sequence) == length]
    return FilterResult(
        sequences=_dedupe_ordered(valid_sequences) if dedupe else valid_sequences,
        raw_count=len(all_sequences),
        valid_count=len(valid_sequences),
    )


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires sequences of equal length")
    return sum(left_char != right_char for left_char, right_char in zip(left, right))


def select_first_valid(sequences: Sequence[str], target_n: int) -> Candidate:
    selected = list(sequences[:target_n])
    return Candidate(sequences=selected, selected_count=len(selected))


def select_lowest_neff(
    query: str,
    sequences: Iterable[str],
    target_n: int,
    neff_fn: Callable[[Sequence[str]], float],
    *,
    candidate_window: int | None = None,
    dedupe: bool = True,
) -> Candidate:
    candidates = _dedupe_ordered(sequences) if dedupe else list(sequences)
    candidates = sorted(candidates, key=lambda sequence: (hamming_distance(query, sequence), sequence))
    selected: list[str] = []
    neff: float | None = None

    while candidates and len(selected) < target_n:
        window_size = candidate_window or len(candidates)
        window = candidates[:window_size]
        best_sequence = min(
            window,
            key=lambda sequence: (
                neff_fn([query, *selected, sequence]),
                hamming_distance(query, sequence),
                sequence,
            ),
        )
        selected.append(best_sequence)
        candidates.remove(best_sequence)
        neff = neff_fn([query, *selected])

    return Candidate(sequences=selected, selected_count=len(selected), neff=neff)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _lowest_neff_row(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [dict(row) for row in rows if _as_float(row.get("neff")) is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (_as_float(row.get("neff")), str(row.get("strategy", ""))))


def summarize_best_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_vram_gb: float | None = None,
    min_mean_plddt: float | None = None,
    min_tm_score: float | None = None,
) -> dict[str, dict[str, Any] | None]:
    all_rows = [dict(row) for row in rows]
    fixed_rows = [
        row
        for row in all_rows
        if row.get("selected_sequences") == row.get("target_n")
    ]
    fold_rows = [
        row
        for row in fixed_rows
        if row.get("mean_plddt") is not None or row.get("tm_score") is not None
    ]
    if min_mean_plddt is not None:
        fold_rows = [
            row
            for row in fold_rows
            if _as_float(row.get("mean_plddt")) is not None
            and _as_float(row.get("mean_plddt")) >= min_mean_plddt
        ]
    if min_tm_score is not None:
        fold_rows = [
            row
            for row in fold_rows
            if _as_float(row.get("tm_score")) is not None
            and _as_float(row.get("tm_score")) >= min_tm_score
        ]

    vram_rows = fixed_rows
    if max_vram_gb is not None:
        vram_rows = [
            row
            for row in fixed_rows
            if _as_float(row.get("peak_vram_gb")) is not None
            and _as_float(row.get("peak_vram_gb")) <= max_vram_gb
        ]

    return {
        "fixed_count_best": _lowest_neff_row(fixed_rows),
        "fold_aware_best": _lowest_neff_row(fold_rows),
        "vram_aware_best": _lowest_neff_row(vram_rows),
    }


def best_lowest_neff(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    return _lowest_neff_row(rows)


def summary_rows(
    summaries: Mapping[str, Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for objective, row in summaries.items():
        if row is None:
            continue
        rows.append({"objective": objective, **dict(row)})
    return rows


def build_generation_config_kwargs(
    decode_conf: dict[str, Any],
    cache_implementation: str | None,
) -> dict[str, Any]:
    kwargs = dict(decode_conf)
    if cache_implementation is not None and cache_implementation != "default":
        kwargs["cache_implementation"] = cache_implementation
    return kwargs


def make_result_row(
    protein_id: str,
    strategy: str,
    variant_param: str,
    cache_implementation: str,
    target_n: int,
    candidate_n_requested: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in RESULT_FIELDS}
    row.update(
        {
            "protein_id": protein_id,
            "strategy": strategy,
            "variant_param": variant_param,
            "cache_implementation": cache_implementation,
            "target_n": target_n,
            "candidate_n_requested": candidate_n_requested,
            "raw_candidates": 0,
            "valid_candidates": 0,
            "selected_sequences": 0,
            "neff": 0.0,
            "gen_time_s": 0.0,
            "peak_vram_gb": 0.0,
            "selection_time_s": 0.0,
            "status": "pending",
            "error": "",
        }
    )
    return row


def write_a3m(path: str | Path, query_id: str, query_seq: str, sequences: Sequence[str]) -> None:
    output_path = Path(path)
    with output_path.open("w") as handle:
        handle.write(f">{query_id}\n{query_seq}\n")
        for index, sequence in enumerate(sequences):
            handle.write(f">generated_{index}\n{sequence}\n")


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str] = RESULT_FIELDS,
) -> None:
    output_path = Path(path)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
