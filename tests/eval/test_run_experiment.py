import csv

import numpy as np
import pytest

from eval import run_experiment
from eval.run_experiment import (
    EMBEDDING_SCHEMA,
    FEATURE_SCHEMA,
    alert_indices,
    calibrate_profile,
    extend_cache,
    load_embeddings,
    load_hits,
    normal_score_ratio,
    sweep_rows,
    sweep_thresholds,
    tabulate,
    write_rows,
)
from monitor.baseline import sample_prompts
from monitor.calibrate import unused_prompts
from monitor.config import DriftConfig
from monitor.extract import FeatureHit


def fake_features(calls: list[list[str]]):
    """An encoder that records each batch and gives prompt i the features [len(p), len(p) + 1]."""

    def encode(prompts: list[str]) -> dict[str, list]:
        calls.append(prompts)
        return {
            "feature_ids": [[len(p), len(p) + 1] for p in prompts],
            "activations": [[2.0, 1.0] for _ in prompts],
        }

    return encode


def test_extend_cache_encodes_only_missing_prompts_in_chunks(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    calls: list[list[str]] = []

    assert extend_cache(path, ["c", "a", "bb"], fake_features(calls), FEATURE_SCHEMA, chunk=2) == 3
    assert calls == [["a", "bb"], ["c"]]  # sorted, two chunks, each saved

    calls.clear()
    assert extend_cache(path, ["a", "dddd", "c"], fake_features(calls), FEATURE_SCHEMA) == 1
    assert calls == [["dddd"]]

    hits = load_hits(path)
    assert sorted(hits) == ["a", "bb", "c", "dddd"]
    assert hits["dddd"] == [FeatureHit(4, 2.0), FeatureHit(5, 1.0)]


def test_embeddings_round_trip_as_float64_rows(tmp_path) -> None:
    path = tmp_path / "embeddings.parquet"
    vectors = {"x": [0.5, -0.25], "y": [1.0, 0.0]}
    extend_cache(
        path,
        ["y", "x"],
        lambda ps: {"embedding": [vectors[p] for p in ps]},
        EMBEDDING_SCHEMA,
    )
    loaded = load_embeddings(path)
    assert loaded["x"].dtype == np.float64
    np.testing.assert_array_equal(loaded["x"], [0.5, -0.25])
    np.testing.assert_array_equal(loaded["y"], [1.0, 0.0])


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.1, 0.9, 0.9, 0.9, 0.1], [1]),  # staying above is one alert
        ([0.9, 0.1, 0.9, 0.1, 0.9], [0, 2, 4]),  # the first window counts; each rise counts
        ([0.5, 0.5, 0.1], []),  # equal to the threshold is not above it
        ([0.1, 0.2], []),
    ],
)
def test_alert_indices_count_rises_above_the_threshold(scores, expected) -> None:
    assert alert_indices(np.array(scores), 0.5).tolist() == expected


def test_tabulate_reports_delay_share_misses_and_alert_rates() -> None:
    window, length = 100, 1000
    rows = [{"prompt": "p", "is_shift": i >= 500} for i in range(length)]
    runs = {("normal_only", 0): rows, ("x_ramp", 0): rows, ("y_ramp", 0): rows}
    n_windows = length - window + 1
    quiet = np.zeros(n_windows)
    shifted = np.zeros(n_windows)
    shifted[[520, 521, 700]] = 1.0  # windows ending at requests 620, 621 and 800
    scores = {
        detector: {("normal_only", 0): quiet, ("x_ramp", 0): shifted, ("y_ramp", 0): quiet}
        for detector in run_experiment.DETECTORS
    }
    thresholds = dict.fromkeys(run_experiment.DETECTORS, 0.5)

    run_rows, alert_rows = tabulate("p1", runs, scores, thresholds, window)

    x = next(r for r in run_rows if r["scenario"] == "x_ramp" and r["detector"] == "sae")
    assert x["first_alert"] == 620
    assert x["delay"] == 120
    assert x["shift_share"] == 1.0
    assert x["alerts"] == 2
    assert x["alerts_per_1000"] == 2.0
    assert x["missed"] is False and x["normal_alert_first"] is False
    assert x["normal_score_ratio"] == 0.0  # the normal run's matching window scores 0
    y = next(r for r in run_rows if r["scenario"] == "y_ramp" and r["detector"] == "sae")
    assert y["missed"] is True and y["first_alert"] is None
    normal = next(r for r in run_rows if r["scenario"] == "normal_only")
    assert normal["alerts"] == 0 and "missed" not in normal
    assert [a["request_number"] for a in alert_rows if a["detector"] == "sae"] == [620, 800]


def test_write_rows_replaces_only_this_profiles_rows(tmp_path) -> None:
    path = tmp_path / "runs.csv"
    fields = ["profile", "value", "maybe"]
    write_rows(path, fields, "a", [{"profile": "a", "value": 1, "maybe": None}])
    write_rows(path, fields, "b", [{"profile": "b", "value": 2}])
    write_rows(path, fields, "a", [{"profile": "a", "value": 3}])

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"profile": "b", "value": "2", "maybe": ""},
        {"profile": "a", "value": "3", "maybe": ""},
    ]


@pytest.fixture
def small_pools(tmp_path, monkeypatch):
    """A 40-prompt train pool (10 in the baseline) and a 20-prompt check pool, files in tmp_path."""
    train = [f"t{i}" for i in range(40)]
    check = [f"c{i}" for i in range(20)]
    monkeypatch.setattr(run_experiment, "read_pool", lambda path: train)
    monkeypatch.setattr(run_experiment, "N_PROMPTS", 10)
    monkeypatch.setattr(
        run_experiment,
        "pool_prompts",
        lambda metadata: {
            "normal_train_unused": unused_prompts(train, metadata),
            "normal_test": check,
        },
    )
    monkeypatch.setattr(run_experiment, "baseline_path", lambda: tmp_path / "m_1.json")
    rng = np.random.default_rng(0)
    return {p: [FeatureHit(int(f), 1.0) for f in rng.choice(30, 3, False)] for p in train + check}


def test_calibrate_writes_then_reproduces_then_refuses_a_mismatch(small_pools, tmp_path) -> None:
    config = DriftConfig(
        window_size=5,
        threshold_percentile=90,
        calibration_windows=20,
        seed=0,
        check_interval_seconds=60,
    )
    first = calibrate_profile(small_pools, config)
    assert (tmp_path / "m_1.json").exists() and (tmp_path / "m_1_threshold.json").exists()

    assert calibrate_profile(small_pools, config) == first  # existing files are reproduced

    # Calibration prompts only (train prompts outside the baseline), so the baseline still matches.
    unused = set(small_pools) - set(sample_prompts([f"t{i}" for i in range(40)], 10))
    changed = {
        p: [FeatureHit(29, 1.0)] if p.startswith("t") and p in unused else h
        for p, h in small_pools.items()
    }
    with pytest.raises(RuntimeError, match="threshold"):
        calibrate_profile(changed, config)


def test_normal_score_ratio_lines_up_the_same_normal_prompts() -> None:
    window = 4
    # Requests 1-10; shifted prompts at requests 3 and 9 (0-based rows 2 and 8).
    rows = [{"is_shift": i in (2, 8)} for i in range(10)]
    # normal_only window i covers normal prompts i .. i+3; give each a distinct score.
    normal_scores = np.arange(10.0)
    # Alert at request 10: its window (rows 6-9) holds normal prompts 5, 6, 7 (one shifted row 8,
    # and row 2 came before). Same normal prompts plus the next one: normal_only window 5.
    assert normal_score_ratio(rows, normal_scores, 10, 2.0, window) == 2.5
    # Alert at request 4: nothing shifted before the window, so normal_only window 0.
    assert normal_score_ratio(rows, normal_scores, 4, 2.0, window) == 0.0


def test_sweep_uses_the_saved_threshold_at_the_configured_percentile() -> None:
    calib = {"sae": np.arange(101.0), "embedding_mean": np.arange(101.0) / 10}
    swept = sweep_thresholds(calib, {"sae": 99.3, "embedding_mean": 9.93}, 99.0)
    assert swept["sae"][99.0] == 99.3 and swept["embedding_mean"][99.0] == 9.93
    assert swept["sae"][90.0] == 90.0 and swept["embedding_mean"][100.0] == 10.0


def test_sweep_rows_report_false_alarms_and_delays_per_threshold() -> None:
    window, length = 100, 1000
    rows = [{"prompt": "p", "is_shift": i >= 500} for i in range(length)]
    runs = {("normal_only", 0): rows, ("x_ramp", 0): rows}
    normal = np.zeros(length - window + 1)
    normal[[10, 20]] = 0.7  # two separate rises above 0.5, none above 0.8
    shifted = np.zeros(length - window + 1)
    shifted[600:] = 0.9  # crosses both thresholds at request 700
    scores = {
        d: {("normal_only", 0): normal, ("x_ramp", 0): shifted} for d in run_experiment.DETECTORS
    }
    thresholds = {d: {50.0: 0.5, 99.0: 0.8} for d in run_experiment.DETECTORS}

    out = sweep_rows("p1", runs, scores, thresholds, window)

    assert len(out) == 4  # 2 detectors x 2 thresholds x 1 shift scenario
    loose, strict = [r for r in out if r["detector"] == "sae"]
    assert (loose["threshold"], loose["fa_alerts_per_1000"], loose["fa_run_rate"]) == (
        0.5,
        2.0,
        1.0,
    )
    assert (strict["fa_alerts_per_1000"], strict["fa_run_rate"]) == (0.0, 0.0)
    assert loose["median_delay"] == strict["median_delay"] == 200
    assert loose["scenario"] == "x_ramp" and loose["detected"] == 1
