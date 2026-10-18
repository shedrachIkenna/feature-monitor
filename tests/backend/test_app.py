import json
import logging
import threading
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend import demo, storage
from backend.app import (
    ADMIN_TOKEN_ENV,
    ADMIN_TOKEN_HEADER,
    CHAT_RATE_LIMIT_ENV,
    EXTRACT_ENV,
    MAX_CHAT_QUEUE_ENV,
    RECORD_TOKEN_ENV,
    RECORD_TOKEN_HEADER,
    REQUIRE_RECORD_TOKEN_ENV,
    RETENTION_CHECK_HOURS,
    SPACE_ID_ENV,
    EngineResult,
    create_app,
    delete_old_rows,
    extraction_enabled,
    max_chat_queue,
    request_log,
)
from monitor.config import load_config
from monitor.extract import FeatureHit
from monitor.job import DriftJob
from monitor.model import PROFILE_ENV, active_profile


class FakeEngine:
    """Echoes the prompt and returns fixed features, with no model behind it."""

    model_name = "fake-gpt2"
    device = "cpu"
    extract = True
    loads = 0

    def __init__(self) -> None:
        FakeEngine.loads += 1

    def run(self, prompt: str) -> EngineResult:
        return EngineResult(f"echo: {prompt}", [FeatureHit(7, 3.5), FeatureHit(2, 1.25)], 1.0, 2.0)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client for an app with a fake engine and a fresh temporary database."""
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.delenv(RECORD_TOKEN_ENV, raising=False)
    monkeypatch.delenv(REQUIRE_RECORD_TOKEN_ENV, raising=False)
    for name in (SPACE_ID_ENV, ADMIN_TOKEN_ENV, CHAT_RATE_LIMIT_ENV, demo.DEMO_TRAFFIC_ENV):
        monkeypatch.delenv(name, raising=False)
    FakeEngine.loads = 0
    with TestClient(create_app(FakeEngine)) as test_client:
        yield test_client


def test_health_reports_model_and_device(client) -> None:
    profile = active_profile()
    assert client.get("/health").json() == {
        "status": "ok",
        "model": "fake-gpt2",
        "model_label": profile.label,
        "neuronpedia_id": profile.neuronpedia_id,
        "device": "cpu",
        "extract": True,
        "storage_resets": False,
    }


def test_health_follows_the_profile_variable(client, monkeypatch) -> None:
    monkeypatch.setenv(PROFILE_ENV, "gemma2_2b")
    health = client.get("/health").json()
    assert health["model_label"] == "Gemma 2 2B"
    assert health["neuronpedia_id"] == "gemma-2-2b/16-gemmascope-res-16k"


def test_health_says_storage_resets_on_a_space(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.setenv(SPACE_ID_ENV, "someone/feature-monitor")
    with TestClient(create_app(FakeEngine)) as client:
        assert client.get("/health").json()["storage_resets"] is True


def test_chat_returns_reply_and_stores_request(client) -> None:
    response = client.post("/chat", json={"prompt": "hi", "scenario": "medical"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "echo: hi"
    assert body["top_features"] == [
        {"feature_id": 7, "activation": 3.5},
        {"feature_id": 2, "activation": 1.25},
    ]
    [row] = storage.recent_requests()
    assert row["id"] == body["request_id"]
    assert (row["scenario"], row["model_name"], row["gen_ms"], row["extract_ms"]) == (
        "medical",
        "fake-gpt2",
        1.0,
        2.0,
    )


def test_chat_with_record_false_answers_but_stores_nothing(client) -> None:
    response = client.post("/chat", json={"prompt": "playground", "record": False})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "echo: playground"
    assert body["request_id"] is None
    assert [f["feature_id"] for f in body["top_features"]] == [7, 2]
    assert storage.recent_requests() == []
    assert client.get("/features/stats").json()["request_count"] == 0


@pytest.mark.parametrize("prompt", ["", "x" * 2001])
def test_chat_rejects_prompt_outside_length_limits(client, prompt) -> None:
    assert client.post("/chat", json={"prompt": prompt}).status_code == 422
    assert storage.recent_requests() == []


def test_chat_accepts_prompt_at_length_limit(client) -> None:
    assert client.post("/chat", json={"prompt": "x" * 2000}).status_code == 200


def test_engine_loads_once_across_requests(client) -> None:
    for _ in range(3):
        client.post("/chat", json={"prompt": "hi"})
    assert FakeEngine.loads == 1


def test_recent_lists_newest_first_with_features_and_limit(client) -> None:
    for prompt in ["a", "b", "c"]:
        client.post("/chat", json={"prompt": prompt})

    recent = client.get("/features/recent", params={"limit": 2}).json()
    assert [r["prompt"] for r in recent] == ["c", "b"]
    assert [f["feature_id"] for f in recent[0]["features"]] == [7, 2]
    assert client.get("/features/recent", params={"limit": 0}).status_code == 422


def test_stats_counts_only_requests_inside_window(client) -> None:
    old = datetime.now(UTC) - timedelta(hours=2)
    storage.save_request("old", "r", "fake-gpt2", [FeatureHit(99, 1.0)], 1.0, 1.0, created_at=old)
    client.post("/chat", json={"prompt": "new"})
    client.post("/chat", json={"prompt": "newer"})

    stats = client.get("/features/stats", params={"minutes": 60}).json()
    assert stats["request_count"] == 2
    assert stats["counts"] == {"7": 2, "2": 2}

    wide = client.get("/features/stats", params={"minutes": 180}).json()
    assert wide["request_count"] == 3
    assert wide["counts"]["99"] == 1


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("1", True), ("0", False)])
def test_extraction_flag_is_on_unless_set_to_zero(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv(EXTRACT_ENV, raising=False)
    else:
        monkeypatch.setenv(EXTRACT_ENV, value)
    assert extraction_enabled() is expected


def test_alerts_endpoint_lists_stored_alerts(client) -> None:
    storage.save_alert("embedding", 0.05, {"top": [{"id": 3}]})

    [alert] = client.get("/alerts").json()

    assert (alert["kind"], alert["score"], alert["details"]) == (
        "embedding",
        0.05,
        {"top": [{"id": 3}]},
    )


def test_app_without_job_schedules_only_the_daily_cleanup(client) -> None:
    [scheduled] = client.app.state.scheduler.get_jobs()

    assert scheduled.func is delete_old_rows
    assert scheduled.trigger.interval.total_seconds() == RETENTION_CHECK_HOURS * 3600


def test_drift_job_is_scheduled_at_the_configured_interval(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    job = DriftJob([], window_size=500)

    with TestClient(create_app(FakeEngine, lambda: job)) as test_client:
        [scheduled] = [j for j in test_client.app.state.scheduler.get_jobs() if j.func == job.run]

    assert scheduled.trigger.interval.total_seconds() == load_config().check_interval_seconds
    # Late checks must run, not be skipped (the default skips any check over 1 s late).
    assert scheduled.misfire_grace_time is None
    assert scheduled.coalesce is True


class BlockingEngine(FakeEngine):
    """A fake engine whose runs wait until the test releases them."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Semaphore(0)

    def run(self, prompt: str) -> EngineResult:
        self.started.release()
        assert self.release.wait(timeout=10)
        return super().run(prompt)


class FailingEngine(FakeEngine):
    def run(self, prompt: str) -> EngineResult:
        raise RuntimeError("model crashed")


def test_chat_rejects_requests_over_the_queue_cap_and_stores_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.setenv(MAX_CHAT_QUEUE_ENV, "2")
    with TestClient(create_app(BlockingEngine)) as client:
        engine: BlockingEngine = client.app.state.engine
        responses = []
        threads = [
            threading.Thread(
                target=lambda i=i: responses.append(client.post("/chat", json={"prompt": f"p{i}"}))
            )
            for i in range(2)
        ]
        for thread in threads:
            thread.start()
        for _ in threads:
            assert engine.started.acquire(timeout=10)

        rejected = client.post("/chat", json={"prompt": "one too many"})
        engine.release.set()
        for thread in threads:
            thread.join(timeout=10)

        assert rejected.status_code == 429
        assert rejected.headers["retry-after"] == "1"
        assert [r.status_code for r in responses] == [200, 200]
        assert sorted(r["prompt"] for r in storage.recent_requests()) == ["p0", "p1"]
        # Finished requests free their places in the queue.
        assert client.post("/chat", json={"prompt": "later"}).status_code == 200


def test_failed_chat_frees_its_place_in_the_queue(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.setenv(MAX_CHAT_QUEUE_ENV, "1")
    app = create_app(FailingEngine)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/chat", json={"prompt": "a"}).status_code == 500
        assert client.post("/chat", json={"prompt": "b"}).status_code == 500


@pytest.mark.parametrize(("value", "expected"), [(None, 4), ("1", 1), ("10", 10)])
def test_chat_queue_cap_reads_its_environment_variable(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv(MAX_CHAT_QUEUE_ENV, raising=False)
    else:
        monkeypatch.setenv(MAX_CHAT_QUEUE_ENV, value)
    assert max_chat_queue() == expected


def test_chat_queue_cap_below_one_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(MAX_CHAT_QUEUE_ENV, "0")
    with pytest.raises(ValueError):
        max_chat_queue()


def record_token_client(tmp_path, monkeypatch, token: str | None, required: bool) -> TestClient:
    """A client for an app whose record-token settings are `token` and `required`."""
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    for name, value in (
        (RECORD_TOKEN_ENV, token),
        (REQUIRE_RECORD_TOKEN_ENV, "1" if required else None),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return TestClient(create_app(FakeEngine))


@pytest.mark.parametrize(
    ("token", "required", "header", "expected"),
    [
        ("s3cret", False, "s3cret", 200),
        ("s3cret", True, "s3cret", 200),
        ("s3cret", False, None, 403),
        ("s3cret", False, "wrong", 403),
        (None, True, None, 403),
        (None, True, "anything", 403),
        (None, False, None, 200),
    ],
)
def test_recorded_chat_needs_the_record_token_when_one_is_set_or_required(
    tmp_path, monkeypatch, token, required, header, expected
) -> None:
    headers = {RECORD_TOKEN_HEADER: header} if header else {}
    with record_token_client(tmp_path, monkeypatch, token, required) as client:
        response = client.post("/chat", json={"prompt": "hi"}, headers=headers)

        assert response.status_code == expected
        assert len(storage.recent_requests()) == (1 if expected == 200 else 0)


def test_unrecorded_chat_never_needs_the_record_token(tmp_path, monkeypatch) -> None:
    with record_token_client(tmp_path, monkeypatch, "s3cret", True) as client:
        response = client.post("/chat", json={"prompt": "hi", "record": False})

        assert response.status_code == 200
        assert storage.recent_requests() == []


def limited_client(tmp_path, monkeypatch, space: bool) -> TestClient:
    """A client allowing 3 unrecorded prompts per minute per address, on a Space or not."""
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.setenv(CHAT_RATE_LIMIT_ENV, "3/minute")
    for name in (RECORD_TOKEN_ENV, REQUIRE_RECORD_TOKEN_ENV, demo.DEMO_TRAFFIC_ENV):
        monkeypatch.delenv(name, raising=False)
    if space:
        monkeypatch.setenv(SPACE_ID_ENV, "someone/feature-monitor")
    else:
        monkeypatch.delenv(SPACE_ID_ENV, raising=False)
    return TestClient(create_app(FakeEngine))


def unrecorded(client: TestClient, forwarded: str | None = None) -> int:
    headers = {"X-Forwarded-For": forwarded} if forwarded else {}
    response = client.post("/chat", json={"prompt": "hi", "record": False}, headers=headers)
    return response.status_code


def test_unrecorded_prompts_are_rate_limited_per_address(tmp_path, monkeypatch) -> None:
    with limited_client(tmp_path, monkeypatch, space=False) as client:
        assert [unrecorded(client) for _ in range(4)] == [200, 200, 200, 429]
        response = client.post("/chat", json={"prompt": "hi", "record": False})

        assert "Too many prompts" in response.json()["detail"]
        assert response.headers["Retry-After"] == "60"


def test_recorded_prompts_are_not_rate_limited(tmp_path, monkeypatch) -> None:
    with limited_client(tmp_path, monkeypatch, space=False) as client:
        statuses = [client.post("/chat", json={"prompt": "hi"}).status_code for _ in range(5)]

        assert statuses == [200] * 5


def test_on_a_space_the_last_forwarded_address_is_the_key(tmp_path, monkeypatch) -> None:
    with limited_client(tmp_path, monkeypatch, space=True) as client:
        # A caller changing the entries it controls (all but the last) gains nothing.
        statuses = [unrecorded(client, f"10.0.0.{i}, 198.51.100.1") for i in range(4)]
        other = unrecorded(client, "198.51.100.2")

        assert statuses == [200, 200, 200, 429]
        assert other == 200


def test_off_a_space_the_forwarded_header_is_ignored(tmp_path, monkeypatch) -> None:
    with limited_client(tmp_path, monkeypatch, space=False) as client:
        statuses = [unrecorded(client, f"198.51.100.{i}") for i in range(4)]

        assert statuses == [200, 200, 200, 429]


def admin_client(tmp_path, monkeypatch, token: str | None) -> TestClient:
    """A client with demo traffic on (over tiny pools) and the given admin token."""
    data, scenarios = tmp_path / "data", tmp_path / "scenarios"
    data.mkdir()
    scenarios.mkdir()
    (data / "normal_test.jsonl").write_text(json.dumps({"text": "n"}) + "\n")
    (scenarios / "calm.yaml").write_text(
        "name: calm\nseed: 1\ntotal_requests: 2\nnormal_pool: normal_test\nshift_pool: null\n"
        "schedule:\n  - {at: 0, shift: 0.0}\n  - {at: 2, shift: 0.0}\n"
    )
    monkeypatch.setattr(demo, "DATA_DIR", data)
    monkeypatch.setattr(demo, "SCENARIO_DIR", scenarios)
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.setenv(demo.DEMO_TRAFFIC_ENV, "1")
    if token:
        monkeypatch.setenv(ADMIN_TOKEN_ENV, token)
    else:
        monkeypatch.delenv(ADMIN_TOKEN_ENV, raising=False)
    return TestClient(create_app(FakeEngine))


@pytest.mark.parametrize(
    ("token", "header", "name", "expected"),
    [
        (None, "anything", "calm", 403),
        ("s3cret", None, "calm", 403),
        ("s3cret", "wrong", "calm", 403),
        ("s3cret", "s3cret", "nope", 404),
        ("s3cret", "s3cret", "calm", 202),
    ],
)
def test_starting_a_scenario_needs_the_admin_token(
    tmp_path, monkeypatch, token, header, name, expected
) -> None:
    headers = {ADMIN_TOKEN_HEADER: header} if header else {}
    with admin_client(tmp_path, monkeypatch, token) as client:
        response = client.post("/admin/scenario", json={"name": name}, headers=headers)

        assert response.status_code == expected


def test_demo_status_reports_a_started_scenario(tmp_path, monkeypatch) -> None:
    with admin_client(tmp_path, monkeypatch, "s3cret") as client:
        assert client.get("/demo/status").json()["enabled"] is True
        client.post(
            "/admin/scenario", json={"name": "calm"}, headers={ADMIN_TOKEN_HEADER: "s3cret"}
        )
        # The thread may already be sending it, or have finished its 2 prompts.
        status = client.get("/demo/status").json()

        assert status["scenario"] in (
            None,
            {"name": "calm", "sent": 0, "total": 2},
            {"name": "calm", "sent": 1, "total": 2},
        )


def test_demo_is_off_by_default(client) -> None:
    assert client.get("/demo/status").json() == {"enabled": False, "warmup": None, "scenario": None}


def test_admin_needs_demo_traffic_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "s3cret")
    with record_token_client(tmp_path, monkeypatch, None, False) as client:
        response = client.post(
            "/admin/scenario", json={"name": "calm"}, headers={ADMIN_TOKEN_HEADER: "s3cret"}
        )

        assert response.status_code == 409


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(record.getMessage()))


def test_each_request_logs_one_json_line_without_the_prompt(client) -> None:
    handler = ListHandler()
    request_log.addHandler(handler)
    try:
        client.post("/chat", json={"prompt": "secret words", "record": False})
    finally:
        request_log.removeHandler(handler)

    # The startup cleanup may log from its own thread at the same time.
    [line] = [entry for entry in handler.lines if entry["event"] == "http"]
    assert (line["method"], line["path"], line["status"]) == ("POST", "/chat", 200)
    assert (line["gen_ms"], line["extract_ms"], line["recorded"]) == (1.0, 2.0, False)
    assert line["total_ms"] >= 0
    assert "secret words" not in json.dumps(line)


def test_cleanup_deletes_rows_older_than_the_retention(client) -> None:
    storage.save_request(
        "p", "r", "m", [], 1.0, 2.0, created_at=datetime.now(UTC) - timedelta(days=8)
    )
    storage.save_request("p", "r", "m", [], 1.0, 2.0)

    delete_old_rows()

    assert storage.total_requests() == 1
