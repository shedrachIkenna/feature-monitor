"""Read drift detection settings from monitor/config.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


@dataclass(frozen=True)
class DriftConfig:
    """Window size and threshold calibration settings."""

    window_size: int
    threshold_percentile: float
    calibration_windows: int
    seed: int


def load_config(path: Path = CONFIG_PATH) -> DriftConfig:
    """Load and check the drift config; unknown or missing keys raise TypeError."""
    config = DriftConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
    if config.window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 < config.threshold_percentile < 100:
        raise ValueError("threshold_percentile must be between 0 and 100")
    if config.calibration_windows <= 0:
        raise ValueError("calibration_windows must be positive")
    return config
