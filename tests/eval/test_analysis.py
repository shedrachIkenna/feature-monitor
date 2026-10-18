import os

import pytest

from eval import analysis
from eval.analysis import (
    failure_cases,
    figure_delays,
    head_to_head,
    results_table,
    same_embedding_rows,
    shift_enrichment,
    spread,
    using_profile,
    window_rates,
)
from monitor.extract import FeatureHit
from monitor.model import PROFILE_ENV


def run(profile, detector, scenario, seed, first, alerts=0, ratio=None, normal_first="False"):
    return {
        "profile": profile,
        "detector": detector,
        "scenario": scenario,
        "seed": seed,
        "first_alert": first,
        "delay": None if first is None or scenario == "normal_only" else first - 500,
        "alerts": alerts,
        "alerts_per_1000": alerts / 2,
        "normal_score_ratio": ratio,
        "normal_alert_first": normal_first,
    }


@pytest.fixture
def rows():
    """Two seeds of normal_only and medical_ramp for every series; seed 1 is a failure case."""
    out = []
    for profile in analysis.PROFILES:
        sae_first = {0: 600.0, 1: 800.0} if profile == "gpt2_small" else {0: 650.0, 1: None}
        for seed in (0, 1):
            out += [
                run(profile, "sae", "normal_only", seed, None if seed else 900.0, alerts=1 - seed),
                run(profile, "embedding_mean", "normal_only", seed, None),
                run(profile, "sae", "medical_ramp", seed, sae_first[seed], ratio=0.97 - seed / 10),
                run(
                    profile,
                    "embedding_mean",
                    "medical_ramp",
                    seed,
                    700.0,
                    ratio=0.5,
                    normal_first="True" if seed else "False",
                ),
            ]
    return out


def test_spread_handles_one_and_no_values() -> None:
    assert spread([4.0]) == {"mean": 4.0, "sd": 0.0, "min": 4.0, "max": 4.0}
    assert spread([])["mean"] is None


def test_results_table_summarises_delays_near_line_and_false_alarms(rows) -> None:
    table = results_table(rows)
    gpt2 = next(
        r
        for r in table
        if r["series"] == "SAE features, GPT-2 Small" and r["scenario"] == "medical_ramp"
    )
    assert gpt2["detected"] == "2/2"
    assert (gpt2["delay_mean"], gpt2["delay_min"], gpt2["delay_max"]) == (200.0, 100.0, 300.0)
    assert gpt2["near_line"] == 1  # 0.97 counts, 0.87 does not
    assert (gpt2["fa_mean"], gpt2["fa_runs"]) == (0.25, "1/2")
    gemma = next(
        r
        for r in table
        if r["series"] == "SAE features, Gemma 2 2B" and r["scenario"] == "medical_ramp"
    )
    assert gemma["detected"] == "1/2" and gemma["delay_mean"] == 150.0


def test_head_to_head_and_failure_cases_find_later_and_missed_runs(rows) -> None:
    assert head_to_head(rows, "gpt2_small") == {"sae_first": 1, "embedding_first_or_sae_missed": 1}
    cases = failure_cases(rows)
    assert [(c["profile"], c["seed"], c["sae_alert"]) for c in cases] == [
        ("gemma2_2b", 1, None),
        ("gpt2_small", 1, 800.0),
    ]
    assert all(c["embedding_normal_alerted_first"] for c in cases)


def test_same_embedding_rows_notices_a_difference(rows) -> None:
    assert same_embedding_rows(rows)
    rows[3]["first_alert"] = 701.0
    assert not same_embedding_rows(rows)


def test_shift_enrichment_ranks_features_the_shift_adds() -> None:
    hits = {p: [FeatureHit(f, 1.0) for f in fs] for p, fs in {"a": [1, 2], "b": [1, 3]}.items()}
    baseline = {"prompt_count": 10, "counts": {1: 10, 3: 1}}  # 1 is everywhere at baseline
    top = shift_enrichment(hits, ["a", "b"], baseline, top=2)
    assert [r["feature"] for r in top] == [2, 3]
    assert top[0] == {"feature": 2, "shift_rate": 500.0, "baseline_rate": 0.0}
    assert window_rates(hits, [{"prompt": "a"}, {"prompt": "b"}], [1, 2]) == {1: 1000.0, 2: 500.0}


def test_using_profile_restores_the_previous_setting(monkeypatch) -> None:
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    with using_profile("gemma2_2b"):
        assert os.environ[PROFILE_ENV] == "gemma2_2b"
    assert PROFILE_ENV not in os.environ


def test_figure_delays_writes_a_png(rows, tmp_path) -> None:
    for r in list(rows):
        for scenario in ("medical_nowhat_ramp", "new_topic_ramp", "injection_ramp"):
            if r["scenario"] == "medical_ramp" and r["delay"] is not None:
                rows.append(r | {"scenario": scenario})
    out = tmp_path / "fig2.png"
    figure_delays(rows, out)
    assert out.read_bytes().startswith(b"\x89PNG")
