"""Tests for MSA generation strategies (all GPU-free via mock model)."""
from unittest.mock import MagicMock, patch

import torch

from ghostfold.msa.strategies import STRATEGIES, BaseStrategy
from ghostfold.msa.strategies.threedipperturb import (
    ThreeDiPerturbStrategy,
    _3DI_SUBST_MATRIX,
    _3DI_TOKENS,
    _mutate_3di,
    _substitution_probs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUERY = "MQCDGLDGADGTSNGQAGASGLAGG"


def _mock_tokenizer():
    tok = MagicMock()
    ids_mock = MagicMock()
    ids_mock.input_ids = torch.zeros(1, 10, dtype=torch.long)
    ids_mock.attention_mask = torch.ones(1, 10, dtype=torch.long)
    ids_mock.to.return_value = ids_mock
    tok.return_value = ids_mock
    tok.batch_decode.return_value = ["a c d e f g h i k l m n p q r s t v w y a c d e f"]
    return tok


def _mock_model(query_seq: str):
    model = MagicMock()
    L = len(query_seq)
    # encoder returns hidden states of shape (1, L, 64)
    enc = MagicMock()
    enc.last_hidden_state = torch.randn(1, L, 64)
    model.encoder.return_value = enc
    # generate returns token ids
    model.generate.return_value = torch.zeros(1, L, dtype=torch.long)
    return model


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_strategies_dict_complete(self):
        assert set(STRATEGIES.keys()) == {
            "encoder_perturb", "diverse_beam", "round_trip", "3di_perturb"
        }

    def test_each_strategy_is_base_subclass(self):
        for cls in STRATEGIES.values():
            assert issubclass(cls, BaseStrategy)

    def test_each_strategy_has_name(self):
        for name, cls in STRATEGIES.items():
            instance = cls()
            assert isinstance(instance.name, str)
            assert len(instance.name) > 0


# ---------------------------------------------------------------------------
# EncoderPerturbStrategy
# ---------------------------------------------------------------------------

class TestEncoderPerturbStrategy:
    @patch("ghostfold.msa.strategies.encoder_perturb.generate_aa")
    def test_returns_list(self, mock_gen_aa):
        mock_gen_aa.return_value = [QUERY, QUERY[::-1]]
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["encoder_perturb"]()
        config = {
            "noise_scales": [0.1],
            "num_return_sequences": 2,
            "decode_conf": {},
        }
        result = strat.generate_msa(QUERY, model, tok, device, config)
        assert isinstance(result, list)

    @patch("ghostfold.msa.strategies.encoder_perturb.generate_aa")
    def test_one_batch_per_noise_scale(self, mock_gen_aa):
        mock_gen_aa.return_value = ["AAAA"]
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["encoder_perturb"]()
        config = {"noise_scales": [0.05, 0.2, 0.4], "num_return_sequences": 1, "decode_conf": {}}
        strat.generate_msa(QUERY, model, tok, device, config)
        # model.generate called once per noise scale (3 scales)
        assert model.generate.call_count == 3

    @patch("ghostfold.msa.strategies.encoder_perturb.generate_aa")
    def test_encoder_called_once(self, mock_gen_aa):
        mock_gen_aa.return_value = ["AAAA"]
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["encoder_perturb"]()
        config = {"noise_scales": [0.1, 0.2, 0.3], "num_return_sequences": 1, "decode_conf": {}}
        strat.generate_msa(QUERY, model, tok, device, config)
        # Encoder should only be called once regardless of number of noise scales
        assert model.encoder.call_count == 1


# ---------------------------------------------------------------------------
# DiverseBeamStrategy
# ---------------------------------------------------------------------------

class TestDiverseBeamStrategy:
    @patch("ghostfold.msa.strategies.diverse_beam.generate_aa")
    def test_returns_list(self, mock_gen_aa):
        mock_gen_aa.return_value = [QUERY]
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["diverse_beam"]()
        result = strat.generate_msa(QUERY, model, tok, device, {"num_beams": 4})
        assert isinstance(result, list)

    @patch("ghostfold.msa.strategies.diverse_beam.generate_aa")
    def test_diverse_beam_params_forwarded(self, mock_gen_aa):
        mock_gen_aa.return_value = ["AAAA"]
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["diverse_beam"]()
        strat.generate_msa(QUERY, model, tok, device, {"num_beams": 8, "diversity_penalty": 1.5})
        call_kwargs = model.generate.call_args.kwargs
        assert call_kwargs["num_beams"] == 8
        assert call_kwargs["num_beam_groups"] == 8
        assert call_kwargs["diversity_penalty"] == 1.5
        assert call_kwargs["do_sample"] is False


# ---------------------------------------------------------------------------
# RoundTripStrategy
# ---------------------------------------------------------------------------

class TestRoundTripStrategy:
    @patch("ghostfold.msa.strategies.round_trip.generate_aa")
    @patch("ghostfold.msa.strategies.round_trip.generate_3di")
    def test_returns_list(self, mock_3di, mock_aa):
        mock_3di.return_value = ["acdef"] * 4
        mock_aa.return_value = [QUERY] * 4
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["round_trip"]()
        config = {"n_seeds": 4, "n_rounds": 3, "decode_conf": {}}
        result = strat.generate_msa(QUERY, model, tok, device, config)
        assert isinstance(result, list)

    @patch("ghostfold.msa.strategies.round_trip.generate_aa")
    @patch("ghostfold.msa.strategies.round_trip.generate_3di")
    def test_output_length(self, mock_3di, mock_aa):
        n_seeds, n_rounds = 4, 3
        mock_3di.return_value = ["acdef"] * n_seeds
        mock_aa.return_value = [QUERY] * n_seeds
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = STRATEGIES["round_trip"]()
        config = {"n_seeds": n_seeds, "n_rounds": n_rounds, "decode_conf": {}}
        result = strat.generate_msa(QUERY, model, tok, device, config)
        # Total = n_seeds * n_rounds
        assert len(result) == n_seeds * n_rounds


# ---------------------------------------------------------------------------
# ThreeDiPerturbStrategy + matrix helpers
# ---------------------------------------------------------------------------

class TestThreeDiPerturbStrategy:
    def test_matrix_is_symmetric_and_complete(self):
        for t1 in _3DI_TOKENS:
            assert t1 in _3DI_SUBST_MATRIX
            for t2 in _3DI_TOKENS:
                assert t2 in _3DI_SUBST_MATRIX[t1]
                # Symmetric
                assert _3DI_SUBST_MATRIX[t1][t2] == _3DI_SUBST_MATRIX[t2][t1]

    def test_diagonal_highest_per_row(self):
        for t in _3DI_TOKENS:
            diag = _3DI_SUBST_MATRIX[t][t]
            assert all(diag >= v for v in _3DI_SUBST_MATRIX[t].values())

    def test_substitution_probs_sum_to_one(self):
        for t in _3DI_TOKENS:
            probs = _substitution_probs(t)
            assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_mutate_3di_preserves_length(self):
        seq = "acdefghiklmnpqrstvwy"
        mutated = _mutate_3di(seq, 0.2)
        assert len(mutated) == len(seq)

    def test_mutate_3di_zero_rate_unchanged(self):
        seq = "acdefghiklmnpqrstvwy"
        mutated = _mutate_3di(seq, 0.0)
        assert mutated == seq

    @patch("ghostfold.msa.strategies.threedipperturb.generate_aa")
    @patch("ghostfold.msa.strategies.threedipperturb.generate_3di")
    def test_returns_list(self, mock_3di, mock_aa):
        mock_3di.return_value = ["acdefghiklmnpqrstvwy"]
        mock_aa.return_value = [QUERY] * 3
        model = _mock_model(QUERY)
        tok = _mock_tokenizer()
        device = torch.device("cpu")

        strat = ThreeDiPerturbStrategy()
        config = {
            "mutation_rates": [0.1, 0.2],
            "n_3di_seeds": 1,
            "num_return_sequences": 3,
            "decode_conf": {},
        }
        result = strat.generate_msa(QUERY, model, tok, device, config)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# BaselineStrategy
# ---------------------------------------------------------------------------

class TestBaselineStrategy:
    @patch("ghostfold.msa.strategies.baseline.generate_sequences_for_coverages_batched")
    def test_returns_list(self, mock_gen):
        mock_gen.return_value = [QUERY, QUERY[::-1]]
        from ghostfold.msa.strategies.baseline import BaselineStrategy
        import torch
        strat = BaselineStrategy()
        result = strat.generate_msa(QUERY, _mock_model(QUERY), _mock_tokenizer(), torch.device("cpu"), {})
        assert isinstance(result, list)

    @patch("ghostfold.msa.strategies.baseline.generate_sequences_for_coverages_batched")
    def test_calls_batched_generator(self, mock_gen):
        mock_gen.return_value = []
        from ghostfold.msa.strategies.baseline import BaselineStrategy
        import torch
        strat = BaselineStrategy()
        strat.generate_msa(QUERY, _mock_model(QUERY), _mock_tokenizer(), torch.device("cpu"), {
            "num_return_sequences": 50,
        })
        assert mock_gen.called


# ---------------------------------------------------------------------------
# EncoderOnly3DiSubStrategy
# ---------------------------------------------------------------------------

class TestEncoderOnly3DiSubStrategy:
    @patch("ghostfold.msa.strategies.encoder_only_3di_sub.generate_aa")
    def test_returns_list(self, mock_gen_aa):
        mock_gen_aa.return_value = [QUERY]
        import torch
        from ghostfold.msa.strategies.encoder_only_3di_sub import EncoderOnly3DiSubStrategy

        mock_cnn = MagicMock()
        L = len(QUERY)
        logits = torch.zeros(1, 20, L)
        logits[0, 0, :] = 10.0
        mock_cnn.return_value = logits

        mock_enc_model = MagicMock()
        enc_out = MagicMock()
        enc_out.last_hidden_state = torch.randn(1, L, 64)
        mock_enc_model.return_value = enc_out

        strat = EncoderOnly3DiSubStrategy()
        result = strat.generate_msa(
            QUERY,
            _mock_model(QUERY),
            _mock_tokenizer(),
            torch.device("cpu"),
            {
                "encoder_model": mock_enc_model,
                "cnn_3di": mock_cnn,
                "mutation_rates": [0.1],
                "variants_per_rate": 2,
                "num_return_sequences": 1,
                "decode_conf": {},
            },
        )
        assert isinstance(result, list)

    @patch("ghostfold.msa.strategies.encoder_only_3di_sub.generate_aa")
    def test_mutation_produces_variants(self, mock_gen_aa):
        mock_gen_aa.return_value = [QUERY]
        import torch
        from ghostfold.msa.strategies.encoder_only_3di_sub import EncoderOnly3DiSubStrategy, _predict_3di_encoder

        mock_cnn = MagicMock()
        L = len(QUERY)
        logits = torch.zeros(1, 20, L)
        mock_cnn.return_value = logits

        mock_enc = MagicMock()
        enc_out = MagicMock()
        enc_out.last_hidden_state = torch.randn(1, L, 64)
        mock_enc.return_value = enc_out
        mock_enc_tokenizer = _mock_tokenizer()

        threedi = _predict_3di_encoder(QUERY, mock_enc, mock_enc_tokenizer, torch.device("cpu"), mock_cnn)
        assert len(threedi) == L
        assert all(c in "acdefghiklmnpqrstvwy" for c in threedi)


# ---------------------------------------------------------------------------
# TemperatureSweepStrategy
# ---------------------------------------------------------------------------

class TestTemperatureSweepStrategy:
    @patch("ghostfold.msa.strategies.temperature_sweep.generate_aa")
    @patch("ghostfold.msa.strategies.temperature_sweep.generate_3di")
    def test_returns_list(self, mock_3di, mock_aa):
        mock_3di.return_value = ["acdefghiklmnpqrstvwy"]
        mock_aa.return_value = [QUERY] * 5
        import torch
        from ghostfold.msa.strategies.temperature_sweep import TemperatureSweepStrategy
        strat = TemperatureSweepStrategy()
        result = strat.generate_msa(
            QUERY,
            _mock_model(QUERY),
            _mock_tokenizer(),
            torch.device("cpu"),
            {"temperatures": [0.5, 1.0], "num_return_sequences": 5, "base_decode_conf": {}},
        )
        assert isinstance(result, list)

    @patch("ghostfold.msa.strategies.temperature_sweep.generate_aa")
    @patch("ghostfold.msa.strategies.temperature_sweep.generate_3di")
    def test_called_once_per_temperature(self, mock_3di, mock_aa):
        mock_3di.return_value = ["acdefghiklmnpqrstvwy"]
        mock_aa.return_value = [QUERY]
        import torch
        from ghostfold.msa.strategies.temperature_sweep import TemperatureSweepStrategy
        strat = TemperatureSweepStrategy()
        temps = [0.5, 1.0, 1.5]
        strat.generate_msa(
            QUERY,
            _mock_model(QUERY),
            _mock_tokenizer(),
            torch.device("cpu"),
            {"temperatures": temps, "num_return_sequences": 1, "base_decode_conf": {}},
        )
        assert mock_aa.call_count == len(temps)

    @patch("ghostfold.msa.strategies.temperature_sweep.generate_aa")
    @patch("ghostfold.msa.strategies.temperature_sweep.generate_3di")
    def test_repetition_penalty_included(self, mock_3di, mock_aa):
        mock_3di.return_value = ["acdefghiklmnpqrstvwy"]
        mock_aa.return_value = [QUERY]
        import torch
        from ghostfold.msa.strategies.temperature_sweep import TemperatureSweepStrategy, _DEFAULT_REPETITION_PENALTY
        strat = TemperatureSweepStrategy()
        strat.generate_msa(
            QUERY,
            _mock_model(QUERY),
            _mock_tokenizer(),
            torch.device("cpu"),
            {"temperatures": [1.0], "num_return_sequences": 1, "base_decode_conf": {}},
        )
        _, call_kwargs = mock_aa.call_args
        passed_conf = call_kwargs.get("decode_conf", mock_aa.call_args[0][5] if len(mock_aa.call_args[0]) > 5 else {})
        assert passed_conf.get("repetition_penalty") == _DEFAULT_REPETITION_PENALTY
