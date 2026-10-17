import json
import time

import pytest

from backend import demo, storage
from backend.app import EngineResult
from monitor.extract import FeatureHit


class RecordingEngine:
    """Echoes prompts and remembers them, with no model behind it."""

    model_name = "fake-gpt2"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, prompt: str) -> EngineResult:
        self.prompts.append(prompt)
        return EngineResult(f"echo: {prompt}", [FeatureHit(7, 1.0)], 1.0, 2.0)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A fresh database plus tiny pools and one 4-request scenario that is all shift."""
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    storage.init_db()
    data, scenarios = tmp_path / "data", tmp_path / "scenarios"
    data.mkdir()
    scenarios.mkdir()
    for pool, texts in (("normal_test", ["n1", "n2", "n3"]), ("shift_x", ["s1", "s2"])):
        (data / f"{pool}.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    (scenarios / "all_shift.yaml").write_text(
        "name: all_shift\nseed: 1\ntotal_requests: 4\nnormal_pool: normal_test\n"
        "shift_pool: shift_x\nschedule:\n  - {at: 0, shift: 1.0}\n  - {at: 4, shift: 1.0}\n"
    )
    return data, scenarios


def make(dirs, engine, busy=lambda: False, warmup_target=3, interval_s=60.0) -> demo.DemoTraffic:
    data, scenarios = dirs
    return demo.DemoTraffic(engine, busy, warmup_target, interval_s, None, data, scenarios)


def wait_until(condition, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_warms_up_to_the_target_then_waits_for_the_interval(dirs) -> None:
    engine = RecordingEngine()
    traffic = make(dirs, engine)
    assert traffic.status() == {"warmup": {"stored": 0, "target": 3}, "scenario": None}

    traffic.start()
    wait_until(lambda: storage.total_requests() >= 3)
    time.sleep(0.5)
    traffic.stop()

    # The first trickle prompt is due at once; warm-up fills the rest; then nothing for 60 s.
    assert storage.total_requests() == 3
    assert {r["scenario"] for r in storage.recent_requests()} == {demo.TRICKLE_LABEL}
    assert set(engine.prompts) <= {"n1", "n2", "n3"}
    assert traffic.status()["warmup"] is None


def test_sends_nothing_while_a_visitor_is_being_served(dirs) -> None:
    engine = RecordingEngine()
    traffic = make(dirs, engine, busy=lambda: True)

    traffic.start()
    time.sleep(0.5)
    traffic.stop()

    assert engine.prompts == []


def test_runs_a_scenario_in_order_then_clears_it(dirs) -> None:
    engine = RecordingEngine()
    traffic = make(dirs, engine, warmup_target=0)
    traffic.start_scenario("all_shift")
    assert traffic.status()["scenario"] == {"name": "all_shift", "sent": 0, "total": 4}

    traffic.start()
    wait_until(lambda: traffic.status()["scenario"] is None)
    traffic.stop()

    scenario_rows = [r for r in storage.recent_requests() if r["scenario"] == "all_shift"]
    assert len(scenario_rows) == 4
    assert all(p.startswith("s") for p in engine.prompts[:4])


def test_rejects_unknown_and_overlapping_scenarios(dirs) -> None:
    traffic = make(dirs, RecordingEngine())

    with pytest.raises(KeyError):
        traffic.start_scenario("nope")
    traffic.start_scenario("all_shift")
    with pytest.raises(RuntimeError):
        traffic.start_scenario("all_shift")


def test_a_failing_prompt_does_not_stop_the_thread(dirs) -> None:
    class FlakyEngine(RecordingEngine):
        def run(self, prompt: str) -> EngineResult:
            if not self.prompts:
                self.prompts.append("failed")
                raise RuntimeError("boom")
            return super().run(prompt)

    engine = FlakyEngine()
    traffic = make(dirs, engine, interval_s=0.1)

    traffic.start()
    wait_until(lambda: storage.total_requests() >= 1)
    traffic.stop()

    assert engine.prompts[0] == "failed"


def test_demo_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv(demo.DEMO_TRAFFIC_ENV, raising=False)
    monkeypatch.delenv(demo.INTERVAL_ENV, raising=False)
    assert not demo.demo_enabled()
    assert demo.demo_interval() == demo.DEFAULT_INTERVAL_S

    monkeypatch.setenv(demo.DEMO_TRAFFIC_ENV, "1")
    monkeypatch.setenv(demo.INTERVAL_ENV, "0")
    assert demo.demo_enabled()
    with pytest.raises(ValueError):
        demo.demo_interval()


def test_the_shipped_scenarios_are_offered() -> None:
    assert demo.scenario_names() == [
        "injection_ramp",
        "medical_ramp",
        "new_topic_ramp",
        "normal_only",
    ]
