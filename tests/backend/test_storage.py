import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend import storage
from monitor.extract import FeatureHit

T0 = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point storage at a fresh temporary database file."""
    path = tmp_path / "test.db"
    monkeypatch.setenv(storage.DB_PATH_ENV, str(path))
    storage.init_db()
    return path


def save(hits: list[FeatureHit], created_at: datetime, prompt: str = "p") -> int:
    return storage.save_request(
        prompt, "r", "gpt2-small", hits, gen_ms=1.0, extract_ms=2.0, created_at=created_at
    )


def test_db_path_comes_from_environment(db) -> None:
    assert storage.db_path() == db
    assert db.exists()


def test_db_path_default_is_anchored_to_repo(monkeypatch) -> None:
    monkeypatch.delenv(storage.DB_PATH_ENV, raising=False)
    assert storage.db_path() == storage.DEFAULT_DB_PATH
    assert storage.DEFAULT_DB_PATH.is_absolute()


def test_init_db_is_idempotent(db) -> None:
    storage.init_db()
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert {"requests", "feature_hits", "alerts"} <= tables
    assert {"idx_requests_created_at", "idx_feature_hits_feature_id"} <= tables


def test_save_request_round_trips_with_ranks(db) -> None:
    hits = [FeatureHit(7, 3.5), FeatureHit(2, 1.25)]
    request_id = storage.save_request(
        "hi", "hello", "gpt2-small", hits, 10.0, 4.0, scenario="medical", created_at=T0
    )

    [row] = storage.recent_requests()
    assert row["id"] == request_id
    assert row["created_at"] == "2026-10-15T12:00:00.000000Z"
    assert row["scenario"] == "medical"
    with sqlite3.connect(db) as conn:
        stored = conn.execute(
            "SELECT feature_id, activation, rank FROM feature_hits ORDER BY rank"
        ).fetchall()
    assert stored == [(7, 3.5, 1), (2, 1.25, 2)]


def test_save_request_rolls_back_when_a_hit_fails(db) -> None:
    bad_hits = [FeatureHit(1, 1.0), FeatureHit(None, 1.0)]  # NULL feature_id violates schema
    with pytest.raises(sqlite3.IntegrityError):
        save(bad_hits, T0)

    assert storage.recent_requests() == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feature_hits").fetchone() == (0,)


def test_hits_require_an_existing_request(db) -> None:
    with storage._connect(None) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO feature_hits VALUES (999, 1, 1.0, 1)")


def test_recent_requests_newest_first_with_limit(db) -> None:
    for minutes, prompt in [(0, "a"), (2, "c"), (1, "b")]:
        save([], T0 + timedelta(minutes=minutes), prompt=prompt)

    assert [r["prompt"] for r in storage.recent_requests()] == ["c", "b", "a"]
    assert [r["prompt"] for r in storage.recent_requests(limit=2)] == ["c", "b"]


def test_feature_counts_and_request_count_use_half_open_window(db) -> None:
    save([FeatureHit(1, 1.0), FeatureHit(2, 1.0)], T0 - timedelta(seconds=1))  # before
    save([FeatureHit(1, 1.0), FeatureHit(2, 1.0)], T0)  # at since: included
    save([FeatureHit(1, 1.0)], T0 + timedelta(minutes=5))
    save([], T0 + timedelta(minutes=6))  # no hits, still counted as a request
    save([FeatureHit(1, 1.0)], T0 + timedelta(hours=1))  # at until: excluded

    until = T0 + timedelta(hours=1)
    assert storage.feature_counts(T0, until) == {1: 2, 2: 1}
    assert storage.request_count(T0, until) == 3


def test_window_accepts_other_timezones(db) -> None:
    save([FeatureHit(5, 1.0)], T0)
    # 14:00 at +02:00 is 12:00 UTC, i.e. exactly T0.
    t0_plus_two = datetime(2026, 10, 15, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    assert storage.feature_counts(t0_plus_two, t0_plus_two + timedelta(minutes=1)) == {5: 1}
    assert storage.feature_counts(t0_plus_two - timedelta(minutes=1), t0_plus_two) == {}


def test_naive_timestamps_are_rejected(db) -> None:
    naive = datetime(2026, 10, 15)  # noqa: DTZ001 - deliberately naive
    with pytest.raises(ValueError):
        storage.feature_counts(naive, naive + timedelta(days=1))
