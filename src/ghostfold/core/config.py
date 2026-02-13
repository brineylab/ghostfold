from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict. Override values win."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_default_config() -> dict:
    """Load the bundled default configuration from package data."""
    ref = importlib.resources.files("ghostfold.data").joinpath("default_config.yaml")
    with importlib.resources.as_file(ref) as config_path:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


def load_config(user_config_path: Optional[Path | str] = None) -> dict:
    """Load configuration, optionally merging user overrides with defaults.

    Args:
        user_config_path: Optional path to a user config YAML file.
            If provided, its values are deep-merged on top of defaults.
            If None, only the bundled defaults are returned.

    Returns:
        The merged configuration dictionary.

    Raises:
        FileNotFoundError: If user_config_path is provided but does not exist.
    """
    defaults = load_default_config()

    if user_config_path is None:
        return defaults

    user_path = Path(user_config_path)
    if not user_path.is_file():
        raise FileNotFoundError(
            f"User config file not found: {user_path}"
        )

    with open(user_path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    return _deep_merge(defaults, user_config)
