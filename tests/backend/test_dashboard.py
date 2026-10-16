from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend import dashboard, storage
from backend.app import create_app
from monitor.extract import FeatureHit
from monitor.job import Detector, DriftJob, sae_sets
from tests.backend.test_app import FakeEngine

WINDOW = 10
NORMAL = [1, 2, 3, 4, 5, 6]
SHIFTED = [11, 12, 13, 14, 15, 16]
# Bucket-aligned; "now" is 12 minutes later, so the last bucket (12:10) is empty.
T0 = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
NOW = T0 + timedelta(minutes=12)


def detector() -> Detector:
    """An SAE-style detector whose baseline is every prompt having the NORMAL features."""
    return Detector("sae", {f: 100 for f in NORMAL}, 100, 0.1, sae_sets, lambda f, *_: f"label {f}")


def add(features: list[int], prompt: str, created_at: datetime) -> None:
    hits = [FeatureHit(f, 10.0 - rank) for rank, f in enumerate(features)]
    storage.save_request(prompt, "r", "fake", hits, 1.0, 1.0, created_at=created_at)


def check(kind: str, score: float, threshold: float, created_at: datetime) -> None:
    storage.save_drift_check(kind, score, threshold, WINDOW, 1, WINDOW, created_at)


def seed() -> None:
    """Old traffic, 10 normal requests in the 12:00 bucket, 10 shifted ones at 12:05, and checks."""
    add(SHIFTED, "too old", T0 - timedelta(hours=2))
    check("sae", 0.9, 0.1, T0 - timedelta(hours=2))
    for i in range(10):
        add(NORMAL, f"normal {i}", T0 + timedelta(minutes=1, seconds=i))
        add(SHIFTED, f"shift {i}", T0 + timedelta(minutes=6, seconds=i))
    check("sae", 0.01, 0.1, T0 + timedelta(minutes=2))
    check("sae", 0.03, 0.1, T0 + timedelta(minutes=3))
    check("sae", 0.5, 0.1, T0 + timedelta(minutes=7))
    check("sae", 0.2, 0.1, T0 + timedelta(minutes=8))
    check("embedding", 0.2, 0.3, T0 + timedelta(minutes=7))


def make_client(tmp_path, monkeypatch, job: DriftJob | None) -> TestClient:
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    monkeypatch.setattr(dashboard, "utc_now", lambda: NOW)
    storage.init_db()
    seed()
    return TestClient(create_app(FakeEngine, (lambda: job) if job else None))


@pytest.fixture
def client(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, DriftJob([detector()], WINDOW)) as test_client:
        yield test_client


@pytest.fixture
def client_without_job(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch, None) as test_client:
        yield test_client


def test_bucket_range_covers_the_hours_up_to_now() -> None:
    buckets = dashboard.bucket_range(1, NOW, 300)
    assert buckets[0] == int((T0 - timedelta(minutes=50)).timestamp())  # 11:12 rounds to 11:10
    assert buckets[-1] == int((T0 + timedelta(minutes=10)).timestamp())
    assert len(buckets) == 13


def test_timeline_fills_every_bucket_with_the_highest_score(client) -> None:
    body = client.get("/dashboard/timeline", params={"hours": 1}).json()

    assert (body["bucket_minutes"], body["detectors"]) == (5, ["embedding", "sae"])
    assert (body["since"], body["until"]) == (
        "2026-10-15T11:10:00.000000Z",
        "2026-10-15T12:15:00.000000Z",
    )
    buckets = {b["start"][11:16]: b for b in body["buckets"]}
    assert len(buckets) == 13
    assert buckets["11:10"]["request_count"] == 0  # the 2-hour-old request is out of range
    assert buckets["12:00"]["request_count"] == 10
    assert buckets["12:00"]["sae"] == {"score": 0.03, "threshold": 0.1, "alert": False}
    assert buckets["12:00"]["embedding"] == {"score": None, "threshold": None, "alert": False}
    assert buckets["12:05"]["request_count"] == 10
    assert buckets["12:05"]["sae"] == {"score": 0.5, "threshold": 0.1, "alert": True}
    assert buckets["12:05"]["embedding"] == {"score": 0.2, "threshold": 0.3, "alert": False}
    assert buckets["12:10"]["request_count"] == 0
    assert buckets["12:10"]["sae"]["score"] is None


def test_timeline_with_one_minute_buckets_keeps_checks_apart(client) -> None:
    body = client.get("/dashboard/timeline", params={"hours": 1, "bucket_minutes": 1}).json()

    assert body["bucket_minutes"] == 1
    assert body["since"] == "2026-10-15T11:12:00.000000Z"
    buckets = {b["start"][11:16]: b for b in body["buckets"]}
    assert len(buckets) == 61
    # The 12:07 and 12:08 checks share a 5-minute bucket but get one minute each here.
    assert buckets["12:07"]["sae"]["score"] == 0.5
    assert buckets["12:08"]["sae"]["score"] == 0.2
    assert buckets["12:06"]["request_count"] == 10
    assert buckets["12:01"]["request_count"] == 10


@pytest.mark.parametrize("path", ["/dashboard/timeline", "/dashboard/feature/11"])
def test_unsupported_bucket_size_is_rejected(client, path) -> None:
    assert client.get(path, params={"bucket_minutes": 2}).status_code == 422


def test_timeline_works_without_a_job(client_without_job) -> None:
    response = client_without_job.get("/dashboard/timeline", params={"hours": 1})
    assert response.status_code == 200
    assert client_without_job.get("/dashboard/timeline", params={"hours": 0}).status_code == 422


def test_drifting_splits_latest_window_into_rising_and_falling(client) -> None:
    body = client.get("/dashboard/drifting", params={"limit": 3}).json()

    assert (body["window_size"], body["request_count"]) == (WINDOW, WINDOW)
    # Saved interleaved (normal i, then shift i), the shifted ones get ids 3, 5, ..., 21.
    assert body["last_request_id"] == 21
    assert [f["id"] for f in body["rising"]] == [11, 12, 13]
    assert [f["id"] for f in body["falling"]] == [1, 2, 3]
    top = body["rising"][0]
    assert top["label"] == "label 11"
    assert top["window_rate"] == pytest.approx(11 / 12 * 1000)
    assert top["baseline_rate"] == pytest.approx(1 / 102 * 1000)
    assert top["ratio"] == pytest.approx(top["window_rate"] / top["baseline_rate"])
    assert all(f["ratio"] < 1 for f in body["falling"])
    assert set(top) == {"id", "label", "baseline_rate", "window_rate", "ratio", "adjusted_p"}


def test_drifting_is_empty_until_the_window_is_full(tmp_path, monkeypatch) -> None:
    job = DriftJob([detector()], window_size=1000)
    with make_client(tmp_path, monkeypatch, job) as test_client:
        body = test_client.get("/dashboard/drifting").json()

    assert body["request_count"] == 21
    assert (body["rising"], body["falling"], body["last_request_id"]) == ([], [], None)


def test_feature_gives_rate_per_bucket_and_latest_strong_prompts(client) -> None:
    body = client.get("/dashboard/feature/11", params={"hours": 1}).json()

    assert (body["feature_id"], body["label"], body["strong_max_rank"]) == (11, "label 11", 5)
    buckets = {b["start"][11:16]: b for b in body["buckets"]}
    assert buckets["12:00"] == {
        "start": "2026-10-15T12:00:00.000000Z",
        "request_count": 10,
        "hits": 0,
        "rate": 0.0,
    }
    assert buckets["12:05"]["rate"] == 1000.0
    assert buckets["12:10"]["rate"] is None  # no requests, so no rate
    prompts = body["strong_prompts"]
    assert [p["prompt"] for p in prompts] == ["shift 9", "shift 8", "shift 7", "shift 6", "shift 5"]
    assert (prompts[0]["rank"], prompts[0]["activation"]) == (1, 10.0)


def test_feature_with_one_minute_buckets(client) -> None:
    params = {"hours": 1, "bucket_minutes": 1}
    body = client.get("/dashboard/feature/11", params=params).json()

    buckets = {b["start"][11:16]: b for b in body["buckets"]}
    assert (body["bucket_minutes"], len(buckets)) == (1, 61)
    assert (buckets["12:06"]["hits"], buckets["12:06"]["rate"]) == (10, 1000.0)
    assert buckets["12:05"]["rate"] is None


def test_feature_ranked_below_top_five_has_no_strong_prompts(client) -> None:
    body = client.get("/dashboard/feature/16", params={"hours": 1}).json()  # always rank 6

    assert body["strong_prompts"] == []
    assert {b["start"][11:16]: b["hits"] for b in body["buckets"]}["12:05"] == 10


def test_feature_rejects_negative_id(client) -> None:
    assert client.get("/dashboard/feature/-1").status_code == 422


def test_labels_returns_each_requested_feature_once(client) -> None:
    labels = client.get("/dashboard/labels", params={"ids": "11,3,11"}).json()
    assert labels == {"11": "label 11", "3": "label 3"}


@pytest.mark.parametrize("ids", ["", "1,,2", "-1", "a", ",".join(str(i) for i in range(21))])
def test_labels_rejects_bad_or_too_many_ids(client, ids) -> None:
    assert client.get("/dashboard/labels", params={"ids": ids}).status_code == 422


@pytest.mark.parametrize(
    "path", ["/dashboard/drifting", "/dashboard/feature/11", "/dashboard/labels?ids=11"]
)
def test_sae_endpoints_need_the_drift_job(client_without_job, path) -> None:
    assert client_without_job.get(path).status_code == 503
