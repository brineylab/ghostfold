import tempfile

from ghostfold.msa.generation import generate_sequences_for_coverages_batched

from .base import BaseStrategy


class BaselineStrategy(BaseStrategy):
    """Default ghostfold coverage-based MSA generation.

    Mirrors the exact path taken by `ghostfold run` — coverage-based batched
    ProstT5 round-trips with the default decoding config.  Used as the
    control condition in benchmarks.
    """

    name = "baseline"

    def generate_msa(
        self,
        query_seq: str,
        model,
        tokenizer,
        device,
        config: dict,
    ) -> list[str]:
        num_return_sequences: int = config.get("num_return_sequences", 100)
        inference_batch_size: int = config.get("inference_batch_size", 8)
        coverage_values: list[float] = config.get("coverage_values", [1.0])
        decode_conf: dict = config.get(
            "decode_conf", {"temperature": 0.7, "top_k": 20, "top_p": 0.95}
        )
        decoding_configs = [decode_conf]

        with tempfile.TemporaryDirectory() as tmp_dir:
            sequences = generate_sequences_for_coverages_batched(
                query_seq=query_seq,
                full_len=len(query_seq),
                decoding_configs=decoding_configs,
                num_return_sequences=num_return_sequences,
                multiplier=1,
                coverage_list=coverage_values,
                model=model,
                tokenizer=tokenizer,
                device=device,
                project_dir=tmp_dir,
                inference_batch_size=inference_batch_size,
            )
        return [s.replace("-", "") for s in sequences if s.replace("-", "")]
