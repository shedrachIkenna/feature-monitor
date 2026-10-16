import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from traffic.generate import (
    RECORD_TOKEN_ENV,
    RECORD_TOKEN_HEADER,
    Scenario,
    build_sequence,
    load_scenario,
    mix_per_block,
    record_headers,
    send,
    shift_fraction,
)

SCENARIO_DIR = Path(__file__).parents[2] / "traffic" / "scenarios"
SCENARIO_FILES = sorted(SCENARIO_DIR.glob("*.yaml"))


def fake_pools(scenario: Scenario) -> dict[str, list[str]]:
    """Small stand-in pools, since the real data files are not in Git."""
    pools = {scenario.normal_pool: [f"normal {i}" for i in range(300)]}
    if scenario.shift_pool:
        pools[scenario.shift_pool] = [f"shift {i}" for i in range(125)]
    return pools


def test_all_four_scenarios_exist_and_load():
    names = {load_scenario(p).name for p in SCENARIO_FILES}
    assert names == {"normal_only", "medical_ramp", "injection_ramp", "new_topic_ramp"}


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_same_seed_gives_identical_sequence(path):
    scenario = load_scenario(path)
    assert build_sequence(scenario, fake_pools(scenario)) == build_sequence(
        scenario, fake_pools(scenario)
    )


def test_different_seed_gives_different_sequence():
    scenario = load_scenario(SCENARIO_DIR / "medical_ramp.yaml")
    other = Scenario(**{**scenario.__dict__, "seed": scenario.seed + 1})
    assert build_sequence(scenario, fake_pools(scenario)) != build_sequence(
        other, fake_pools(other)
    )


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_mix_matches_schedule_within_two_points_per_block(path):
    scenario = load_scenario(path)
    rows = build_sequence(scenario, fake_pools(scenario))
    assert len(rows) == scenario.total_requests
    assert [r["index"] for r in rows] == list(range(scenario.total_requests))
    for start, scheduled, actual in mix_per_block(scenario, rows):
        assert abs(actual - scheduled) <= 2, f"block {start}: {actual} vs {scheduled}"


def test_ground_truth_matches_pool():
    scenario = load_scenario(SCENARIO_DIR / "medical_ramp.yaml")
    for row in build_sequence(scenario, fake_pools(scenario)):
        assert row["is_shift"] == (row["pool"] == "shift_medical")
        assert row["prompt"].startswith("shift" if row["is_shift"] else "normal")


def test_small_pool_repeats_evenly():
    scenario = load_scenario(SCENARIO_DIR / "injection_ramp.yaml")
    rows = build_sequence(scenario, fake_pools(scenario))
    counts = Counter(r["prompt"] for r in rows if r["is_shift"])
    assert max(counts.values()) - min(counts.values()) <= 1


def test_shift_fraction_interpolates_linearly():
    schedule = [(0, 0.0), (500, 0.0), (1500, 0.3), (2000, 0.3)]
    assert shift_fraction(schedule, 250) == 0.0
    assert shift_fraction(schedule, 1000) == pytest.approx(0.15)
    assert shift_fraction(schedule, 1500) == pytest.approx(0.3)
    assert shift_fraction(schedule, 1999) == pytest.approx(0.3)


def test_load_scenario_rejects_shift_without_pool(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nseed: 1\ntotal_requests: 100\nnormal_pool: n\nshift_pool: null\n"
        "schedule:\n  - {at: 0, shift: 0.0}\n  - {at: 100, shift: 0.5}\n"
    )
    with pytest.raises(ValueError, match="shift_pool"):
        load_scenario(path)


def test_send_posts_each_prompt_with_scenario_tag():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"reply": "", "request_id": len(seen), "top_features": []})

    rows = [{"index": 0, "pool": "p", "is_shift": False, "prompt": "hi"}]
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    results = list(send(rows, "medical_ramp", client))
    assert seen == [{"prompt": "hi", "scenario": "medical_ramp"}]
    assert results == [{**rows[0], "status": 200, "request_id": 1}]


def test_send_stops_at_the_first_refusal_to_record():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(403, json={"detail": "Recording is disabled on this server."})

    rows = [{"index": i, "pool": "p", "is_shift": False, "prompt": f"p{i}"} for i in range(3)]
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    with pytest.raises(SystemExit, match="Recording is disabled"):
        list(send(rows, "medical_ramp", client))
    assert len(seen) == 1


def test_record_headers_send_the_token_only_when_set(monkeypatch):
    monkeypatch.delenv(RECORD_TOKEN_ENV, raising=False)
    assert record_headers() == {}
    monkeypatch.setenv(RECORD_TOKEN_ENV, "s3cret")
    assert record_headers() == {"X-Record-Token": "s3cret"}


def test_record_token_names_match_the_api():
    from backend import app

    assert (RECORD_TOKEN_ENV, RECORD_TOKEN_HEADER) == (
        app.RECORD_TOKEN_ENV,
        app.RECORD_TOKEN_HEADER,
    )
