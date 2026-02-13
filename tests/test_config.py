import pytest
from pathlib import Path

from ghostfold.core.config import load_config, load_default_config, _deep_merge


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_override(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99}, "b": 3}

    def test_empty_override(self):
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}


class TestLoadDefaultConfig:
    def test_loads_expected_keys(self):
        cfg = load_default_config()
        assert "decoding_params" in cfg
        assert "num_return_sequences" in cfg
        assert "inference_batch_size" in cfg
        assert "multiplier" in cfg

    def test_decoding_params_structure(self):
        cfg = load_default_config()
        dp = cfg["decoding_params"]
        assert "base" in dp
        assert "matrix" in dp
        assert "temperature" in dp["matrix"]
        assert "repetition_penalty" in dp["matrix"]


class TestLoadConfig:
    def test_defaults_only(self):
        cfg = load_config()
        assert "decoding_params" in cfg

    def test_user_override(self, tmp_dir):
        user_config = tmp_dir / "custom.yaml"
        user_config.write_text("num_return_sequences: 10\n")
        cfg = load_config(user_config)
        assert cfg["num_return_sequences"] == 10
        # Defaults still present
        assert "decoding_params" in cfg

    def test_user_deep_override(self, tmp_dir):
        user_config = tmp_dir / "custom.yaml"
        user_config.write_text(
            "decoding_params:\n  base:\n    top_k: 50\n"
        )
        cfg = load_config(user_config)
        assert cfg["decoding_params"]["base"]["top_k"] == 50
        # Other base values preserved
        assert cfg["decoding_params"]["base"]["top_p"] == 0.95

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))
