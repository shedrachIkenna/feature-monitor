import threading
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.app import (
    EXTRACT_ENV,
    MAX_CHAT_QUEUE_ENV,
    RECORD_TOKEN_ENV,
    RECORD_TOKEN_HEADER,
    REQUIRE_RECORD_TOKEN_ENV,
    EngineResult,
    create_app,
    extraction_enabled,
    max_chat_queue,
)
from monitor.config import load_config
from monitor.extract import FeatureHit
from monitor.job import DriftJob


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
    FakeEngine.loads = 0
    with TestClient(create_app(FakeEngine)) as test_client:
        yield test_client


def test_health_reports_model_and_device(client) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "model": "fake-gpt2",
        "device": "cpu",
        "extract": True,
    }


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


def test_app_without_job_has_no_scheduler(client) -> None:
    assert client.app.state.scheduler is None


def test_drift_job_is_scheduled_at_the_configured_interval(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    job = DriftJob([], window_size=500)

    with TestClient(create_app(FakeEngine, lambda: job)) as test_client:
        [scheduled] = test_client.app.state.scheduler.get_jobs()

    assert scheduled.func == job.run
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
