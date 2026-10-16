import random

import numpy as np
import pytest

from eval import multiseed
from eval.multiseed import (
    delay_summary,
    detection,
    false_alarms,
    first_alert,
    load_scenarios,
    mark_normal_first,
    representations,
    rolling_scores,
    shift_share,
    starts_with_what,
    sweep,
)
from monitor.drift import score_window


def random_sets(n: int, ids: int, k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [rng.sample(range(ids), k) for _ in range(n)]


def test_rolling_scores_match_score_window_on_every_window() -> None:
    sets = random_sets(60, ids=40, k=5, seed=0) + [[]]  # a prompt with no features
    baseline = {f: c for f, c in enumerate(range(1, 31))}  # IDs 30-39 never seen at baseline
    window = 20

    scores = rolling_scores(sets, baseline, window)

    assert len(scores) == len(sets) - window + 1
    for i, score in enumerate(scores):
        assert score == pytest.approx(score_window(baseline, sets[i : i + window]), abs=1e-12)


def test_rolling_scores_is_empty_before_the_first_full_window() -> None:
    assert rolling_scores([[1], [2]], {1: 1}, 3).size == 0


def test_first_alert_is_the_request_number_ending_the_first_window_above() -> None:
    scores = np.array([0.1, 0.2, 0.5, 0.6])
    assert first_alert(scores, 0.4, window_size=10) == 12  # third window ends at request 12
    assert first_alert(scores, 0.6, window_size=10) is None  # strictly above, as in the job


def test_detection_reports_delay_after_shift_start_and_shift_share(monkeypatch) -> None:
    monkeypatch.setattr(multiseed, "SHIFT_START", 4)
    rows = [{"is_shift": i >= 4} for i in range(8)]
    scores = np.array([0.0, 0.0, 0.0, 0.9, 0.9])  # windows of 4 ending at requests 4..8

    assert detection(rows, scores, 0.5, window_size=4) == {
        "first_alert": 7,
        "delay": 3,
        "shift_share": 0.75,
    }
    assert detection(rows, scores, 1.0, window_size=4)["delay"] is None
    assert shift_share(rows, 4, 4) == 0.0


def test_false_alarms_counts_windows_and_runs() -> None:
    runs = [np.array([0.1, 0.3]), np.array([0.1, 0.1]), np.array([0.1, 0.1])]
    assert false_alarms(runs, 0.2) == {"window_rate": 1 / 6, "run_rate": 1 / 3}


def test_delay_summary_median_needs_more_than_half_detected() -> None:
    assert delay_summary([10, 30, None]) == {
        "detected": 2,
        "runs": 3,
        "median": 30,
        "min": 10,
        "max": 30,
    }
    half = delay_summary([10, None])
    assert (half["median"], half["detected"]) == (None, 1)
    assert delay_summary([None]) == {
        "detected": 0,
        "runs": 1,
        "median": None,
        "min": None,
        "max": None,
    }


def test_sweep_gives_false_alarms_and_delays_per_threshold(monkeypatch) -> None:
    monkeypatch.setattr(multiseed, "SHIFT_START", 2)
    scores = {
        ("normal_only", 0): np.array([0.1, 0.3]),
        ("a_ramp", 0): np.array([0.1, 0.5]),
        ("a_ramp", 1): np.array([0.6, 0.6]),
    }
    rows = {key: [{"is_shift": i >= 2} for i in range(3)] for key in scores}

    low, high = sweep(scores, rows, {90.0: 0.2, 99.0: 0.55}, window_size=2)

    assert (low["window_rate"], low["run_rate"]) == (0.5, 1.0)
    assert low["a_ramp"]["min"] == 0 and low["a_ramp"]["max"] == 1
    assert (high["window_rate"], high["run_rate"]) == (0.0, 0.0)
    assert high["a_ramp"]["detected"] == 1


def test_representations_encode_only_uncached_prompts(tmp_path) -> None:
    cache = tmp_path / "reps.json"
    calls = []

    def encode(prompts):
        calls.append(prompts)
        return [[len(p)] for p in prompts], [0 for _ in prompts]

    first = representations({"a", "bb"}, cache, encode)
    second = representations({"a", "bb", "ccc"}, cache, encode)

    assert calls == [["a", "bb"], ["ccc"]]
    assert first["bb"] == {"sae": [2], "cluster": 0}
    assert second["ccc"] == {"sae": [3], "cluster": 0}


@pytest.mark.parametrize(
    ("text", "expected"),
    [("What is X?", True), ("what's X", True), ("  What are", True), ("How to treat X?", False)],
)
def test_starts_with_what(text, expected) -> None:
    assert starts_with_what(text) is expected


def test_nowhat_scenario_is_medical_ramp_with_another_shift_pool() -> None:
    scenarios = load_scenarios()
    medical, nowhat = scenarios["medical_ramp"], scenarios["medical_nowhat_ramp"]
    assert nowhat.shift_pool == "shift_medical_nowhat"
    assert (nowhat.schedule, nowhat.seed, nowhat.total_requests, nowhat.normal_pool) == (
        medical.schedule,
        medical.seed,
        medical.total_requests,
        medical.normal_pool,
    )


@pytest.mark.parametrize(
    ("first", "normal_first", "expected"),
    [(600, None, False), (600, 700, False), (600, 600, True), (600, 502, True), (None, 502, False)],
)
def test_mark_normal_first(first, normal_first, expected) -> None:
    found = mark_normal_first({"first_alert": first}, normal_first)
    assert found["normal_alert_first"] is expected
