from pathlib import Path

import pytest

from monitor.config import load_config

VALID = "window_size: 500\nthreshold_percentile: 99\ncalibration_windows: 1000\nseed: 0\n"


def test_project_config_loads() -> None:
    config = load_config()

    assert config.window_size > 0
    assert 0 < config.threshold_percentile < 100


def test_load_config_reads_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(VALID)

    config = load_config(path)

    assert (config.window_size, config.threshold_percentile) == (500, 99)


@pytest.mark.parametrize(
    "text, error",
    [
        (VALID.replace("500", "0"), ValueError),
        (VALID.replace("99", "100"), ValueError),
        (VALID + "extra: 1\n", TypeError),
    ],
)
def test_load_config_rejects_bad_values(tmp_path: Path, text: str, error: type) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text)

    with pytest.raises(error):
        load_config(path)
