"""SQLite storage for monitored requests, their SAE feature hits, and drift alerts.

Hits live in their own table, one row per (request, feature), instead of a JSON column on
`requests`. The drift job asks "in how many requests did feature X appear in this window?";
with an index on `feature_hits.feature_id` SQLite answers that by lookup, whereas a JSON
column would have to be parsed for every request on every query. Separate rows also let
the database enforce one hit per rank per request.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only import: monitor.extract pulls in torch, which storage does not need.
    from monitor.extract import FeatureHit

DB_PATH_ENV = "FEATURE_MONITOR_DB"
# Anchored to the repo, not the working directory, so every entry point shares one file.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "feature_monitor.db"
# Fixed-width UTC format: comparing these strings as text orders them as times.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt TEXT NOT NULL,
    reply TEXT NOT NULL,
    model_name TEXT NOT NULL,
    scenario TEXT,
    gen_ms REAL NOT NULL,
    extract_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feature_hits (
    request_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    feature_id INTEGER NOT NULL,
    activation REAL NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY (request_id, rank)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    feature_id INTEGER,
    score REAL NOT NULL,
    details TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_feature_hits_feature_id ON feature_hits(feature_id);
"""


def db_path() -> Path:
    """Return the database path from the environment, or the repo default."""
    return Path(os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH)


def format_timestamp(moment: datetime) -> str:
    """Convert a timezone-aware datetime to the stored UTC string format."""
    if moment.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT)


@contextmanager
def _connect(path: Path | None) -> Iterator[sqlite3.Connection]:
    """Open a connection with foreign keys on, closing it afterwards."""
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn


def init_db(path: Path | None = None) -> None:
    """Create the tables and indexes if they do not exist yet."""
    with _connect(path) as conn:
        conn.executescript(SCHEMA)


def save_request(
    prompt: str,
    reply: str,
    model_name: str,
    hits: Sequence[FeatureHit],
    gen_ms: float,
    extract_ms: float,
    scenario: str | None = None,
    created_at: datetime | None = None,
    path: Path | None = None,
) -> int:
    """Store a request and its ranked hits in one transaction; return the request id."""
    timestamp = format_timestamp(created_at or datetime.now(UTC))
    with _connect(path) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO requests (created_at, prompt, reply, model_name, scenario, gen_ms,"
            " extract_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, prompt, reply, model_name, scenario, gen_ms, extract_ms),
        )
        request_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO feature_hits (request_id, feature_id, activation, rank)"
            " VALUES (?, ?, ?, ?)",
            [
                (request_id, hit.feature_id, hit.activation, rank)
                for rank, hit in enumerate(hits, start=1)
            ],
        )
    return request_id


def recent_requests(limit: int = 50, path: Path | None = None) -> list[dict]:
    """Return up to `limit` requests, newest first, without their hits."""
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def feature_counts(since: datetime, until: datetime, path: Path | None = None) -> dict[int, int]:
    """Count, per feature, the requests in [since, until) where it was among the hits."""
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT h.feature_id, COUNT(DISTINCT h.request_id) FROM feature_hits h"
            " JOIN requests r ON r.id = h.request_id"
            " WHERE r.created_at >= ? AND r.created_at < ? GROUP BY h.feature_id",
            (format_timestamp(since), format_timestamp(until)),
        ).fetchall()
    return {feature_id: count for feature_id, count in rows}


def request_count(since: datetime, until: datetime, path: Path | None = None) -> int:
    """Count requests in [since, until), the denominator for feature frequencies."""
    with _connect(path) as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE created_at >= ? AND created_at < ?",
            (format_timestamp(since), format_timestamp(until)),
        ).fetchone()
    return count
