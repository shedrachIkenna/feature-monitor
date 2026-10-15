import pytest

from backend import storage
from monitor.extract import FeatureHit
from monitor.job import Detector, DriftJob, example_prompts, sae_sets

WINDOW = 10
NORMAL = [1, 2, 3, 4, 5, 6]
SHIFTED = [11, 12, 13, 14, 15, 16]


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    storage.init_db()


def detector() -> Detector:
    """A detector over stored features whose baseline is every prompt having NORMAL."""
    return Detector(
        name="sae",
        baseline_counts={f: 100 for f in NORMAL},
        baseline_n=100,
        threshold=0.1,
        to_sets=sae_sets,
        describe=lambda f, *_: f"label {f}",
    )


def add(features: list[int], n: int, prompt: str = "p") -> None:
    for _ in range(n):
        hits = [FeatureHit(f, 1.0) for f in features]
        storage.save_request(prompt, "r", "fake", hits, gen_ms=1.0, extract_ms=1.0)


def test_no_check_until_the_window_is_full(db) -> None:
    add(SHIFTED, WINDOW - 1)

    assert DriftJob([detector()], WINDOW).run() == []
    assert storage.list_alerts() == []


def test_normal_window_raises_no_alert_but_stores_the_check(db) -> None:
    add(NORMAL, WINDOW)
    job = DriftJob([detector()], WINDOW)

    assert job.run() == []
    [check] = storage.list_drift_checks()
    assert (check["kind"], check["threshold"], check["window_size"]) == ("sae", 0.1, WINDOW)
    assert (check["first_request_id"], check["last_request_id"]) == (1, WINDOW)
    assert check["score"] == pytest.approx(0.0)


def test_shifted_window_raises_alert_with_top_five(db) -> None:
    add(NORMAL, WINDOW)
    add(SHIFTED, WINDOW)
    job = DriftJob([detector()], WINDOW)

    [alert_id] = job.run()

    [alert] = storage.list_alerts()
    [check] = storage.list_drift_checks()
    assert alert["id"] == alert_id
    assert alert["score"] == check["score"]
    assert alert["kind"] == "sae"
    assert alert["score"] > 0.1
    details = alert["details"]
    assert (details["first_request_id"], details["last_request_id"]) == (11, 20)
    assert details["threshold"] == 0.1
    assert details["window_size"] == WINDOW
    assert len(details["top"]) == 5
    assert all(t["label"] == f"label {t['id']}" for t in details["top"])
    assert all(t["ratio"] != 1 for t in details["top"])


def test_same_window_is_not_checked_twice(db) -> None:
    add(SHIFTED, WINDOW)
    job = DriftJob([detector()], WINDOW)

    assert len(job.run()) == 1
    assert job.run() == []
    add(SHIFTED, 1)
    assert len(job.run()) == 1
    assert [c["last_request_id"] for c in storage.list_drift_checks()] == [WINDOW, WINDOW + 1]


def test_example_prompts_describe_a_cluster() -> None:
    requests = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}, {"prompt": "d"}]
    sets = [[0], [1], [0], [0]]

    assert example_prompts(0, requests, sets) == "a | c"
