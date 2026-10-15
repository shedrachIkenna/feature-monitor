from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.app import EngineResult, create_app
from monitor.extract import FeatureHit


class FakeEngine:
    """Echoes the prompt and returns fixed features, with no model behind it."""

    model_name = "fake-gpt2"
    device = "cpu"
    loads = 0

    def __init__(self) -> None:
        FakeEngine.loads += 1

    def run(self, prompt: str) -> EngineResult:
        return EngineResult(f"echo: {prompt}", [FeatureHit(7, 3.5), FeatureHit(2, 1.25)], 1.0, 2.0)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client for an app with a fake engine and a fresh temporary database."""
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    FakeEngine.loads = 0
    with TestClient(create_app(FakeEngine)) as test_client:
        yield test_client


def test_health_reports_model_and_device(client) -> None:
    assert client.get("/health").json() == {"status": "ok", "model": "fake-gpt2", "device": "cpu"}


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
