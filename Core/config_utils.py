"""Utilities for loading and saving Spectra project configuration."""

import json
from pathlib import Path

DEFAULT_CONFIG = {
    "distance": 200,
    "tx_power": 30,
    "Nx": 8,
    "Ny": 8,
    "steering": 0,
}


def load_config(path="spectra_config.json"):
    """Load configuration from JSON and fall back to defaults."""
    config_path = Path(path)

    if not config_path.exists():
        return dict(DEFAULT_CONFIG)

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    merged = dict(DEFAULT_CONFIG)
    merged.update({key: value for key, value in loaded.items() if value is not None})
    return merged


def save_config(path, config):
    """Persist a configuration dict to JSON."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({**DEFAULT_CONFIG, **config}, handle, indent=2)
