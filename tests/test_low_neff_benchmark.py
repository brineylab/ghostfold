from ghostfold.benchmark.low_neff import (
    RESULT_FIELDS,
    SUMMARY_FIELDS,
    filter_exact_length,
    generate_temperature_variants,
    select_first_valid,
    select_lowest_neff,
    summarize_best_rows,
)


def test_generate_temperature_variants_includes_modern_sampling_knobs():
    variants = list(generate_temperature_variants(["dynamic", "static"]))

    assert len(variants) == 2 * ((4 * 4 * 3) + 4)
    assert {variant.strategy for variant in variants} == {"temperature_low_neff"}
    assert {variant.cache_implementation for variant in variants} == {"dynamic", "static"}
    assert "protein_id" in RESULT_FIELDS
    assert SUMMARY_FIELDS == ["objective", *RESULT_FIELDS]

    variant_params = {variant.variant_param for variant in variants}
    assert "temperature=0.3,top_k=1,top_p=0.70" in variant_params
    assert "temperature=0.9,top_k=10,top_p=0.90" in variant_params
    assert "temperature=0.5,top_k=5,top_p=0.80,min_p=0.05" in variant_params
    assert "temperature=0.5,top_k=5,top_p=0.80,typical_p=0.8" in variant_params
    assert "temperature=0.5,top_k=5,top_p=0.80,eta_cutoff=0.0006" in variant_params
    assert "temperature=0.5,top_k=5,top_p=0.80,epsilon_cutoff=0.0006" in variant_params

    modern = {
        key: variant
        for variant in variants
        for key in ("min_p", "typical_p", "eta_cutoff", "epsilon_cutoff")
        if key in variant.decode_conf
    }
    assert modern["min_p"].decode_conf == {
        "temperature": 0.5,
        "top_k": 5,
        "top_p": 0.8,
        "repetition_penalty": 1.15,
        "min_p": 0.05,
    }


def test_filter_exact_length_removes_wrong_lengths_and_tracks_counts():
    result = filter_exact_length(["AAAA", "AA", "BBBB", "AAAA", "CCCC"], length=4)

    assert result.raw_count == 5
    assert result.valid_count == 4
    assert result.sequences == ["AAAA", "BBBB", "CCCC"]

    undeduped = filter_exact_length(["AAAA", "AA", "BBBB", "AAAA"], length=4, dedupe=False)
    assert undeduped.raw_count == 4
    assert undeduped.valid_count == 3
    assert undeduped.sequences == ["AAAA", "BBBB", "AAAA"]


def test_select_first_valid_returns_prefix():
    result = select_first_valid(["AAAA", "BBBB", "CCCC"], target_n=2)

    assert result.sequences == ["AAAA", "BBBB"]
    assert result.selected_count == 2
    assert result.neff is None


def test_select_lowest_neff_prefers_redundant_candidates():
    query = "AAAA"
    candidates = ["TTTT", "AAAT", "AAAC"]

    def neff_fn(sequences):
        return len(set(sequences))

    result = select_lowest_neff(
        query,
        candidates,
        target_n=2,
        neff_fn=neff_fn,
        candidate_window=3,
    )

    assert result.sequences == ["AAAC", "AAAT"]
    assert result.selected_count == 2
    assert result.neff == 3


def test_summarize_best_rows_picks_fixed_count_and_vram_best():
    rows = [
        {
            "protein_id": "p1",
            "strategy": "a",
            "variant_param": "slow",
            "cache_implementation": "dynamic",
            "target_n": 2,
            "candidate_n_requested": 10,
            "raw_candidates": 10,
            "valid_candidates": 10,
            "selected_sequences": 2,
            "neff": 3.0,
            "gen_time_s": 1.0,
            "peak_vram_gb": 7.0,
            "selection_time_s": 0.1,
            "ptm": 0.6,
            "mean_plddt": 70.0,
            "rmsd": None,
            "tm_score": 0.6,
            "status": "ok",
            "error": "",
        },
        {
            "protein_id": "p1",
            "strategy": "b",
            "variant_param": "best",
            "cache_implementation": "static",
            "target_n": 2,
            "candidate_n_requested": 10,
            "raw_candidates": 10,
            "valid_candidates": 10,
            "selected_sequences": 2,
            "neff": 1.5,
            "gen_time_s": 1.0,
            "peak_vram_gb": 9.0,
            "selection_time_s": 0.1,
            "ptm": 0.5,
            "mean_plddt": 80.0,
            "rmsd": None,
            "tm_score": 0.7,
            "status": "ok",
            "error": "",
        },
        {
            "protein_id": "p1",
            "strategy": "c",
            "variant_param": "partial",
            "cache_implementation": "dynamic",
            "target_n": 2,
            "candidate_n_requested": 10,
            "raw_candidates": 10,
            "valid_candidates": 10,
            "selected_sequences": 1,
            "neff": 1.0,
            "gen_time_s": 1.0,
            "peak_vram_gb": 2.0,
            "selection_time_s": 0.1,
            "ptm": 0.7,
            "mean_plddt": 90.0,
            "rmsd": None,
            "tm_score": 0.9,
            "status": "ok",
            "error": "",
        },
    ]

    summary = summarize_best_rows(rows, max_vram_gb=8.0, min_mean_plddt=75.0, min_tm_score=0.65)

    assert summary["fixed_count_best"]["variant_param"] == "best"
    assert summary["vram_aware_best"]["variant_param"] == "slow"
    assert summary["fold_aware_best"]["variant_param"] == "best"
