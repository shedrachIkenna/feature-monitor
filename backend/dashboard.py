"""Read-only views for the dashboard: the drift timeline, drifting features, and one feature.

Time is cut into fixed buckets of 5 minutes (the default) or 1 minute, aligned to the UTC clock
(12:00-12:05, 12:05-12:10, ...). Every bucket in the range is returned, empty or not, so a chart's
time axis has no gaps. The drift job checks about once a minute, so a 5-minute bucket can hold
several checks per detector; the timeline shows the highest score, so a threshold crossing is never
hidden inside a bucket. 1-minute buckets hold about one check each, for short scenario runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from backend import storage
from monitor.job import Detector, DriftJob, describe_drift, rank_drift

BUCKET_MINUTES = (1, 5)
DEFAULT_BUCKET_MINUTES = 5
# "Fired strongly": among the prompt's 5 strongest features. Rank is used instead of an
# activation cutoff because activation scales differ from feature to feature.
STRONG_MAX_RANK = 5
STRONG_PROMPTS = 5


def utc_now() -> datetime:
    """The current time; tests replace it to pin the bucket range."""
    return datetime.now(UTC)


def bucket_range(hours: int, now: datetime, size: int) -> list[int]:
    """Bucket starts in Unix seconds, from the bucket holding now - hours to the one holding now."""
    last = int(now.timestamp()) // size * size
    first = int((now - timedelta(hours=hours)).timestamp()) // size * size
    return list(range(first, last + 1, size))


def _moment(bucket: int) -> datetime:
    return datetime.fromtimestamp(bucket, UTC)


def _bucket_of(timestamp: str, size: int) -> int:
    seconds = int(
        datetime.strptime(timestamp, storage.TIMESTAMP_FORMAT).replace(tzinfo=UTC).timestamp()
    )
    return seconds // size * size


def timeline(
    hours: int, bucket_minutes: int = DEFAULT_BUCKET_MINUTES, path: Path | None = None
) -> dict:
    """Per bucket: request count and, per detector, the highest check score and its threshold."""
    size = bucket_minutes * 60
    buckets = bucket_range(hours, utc_now(), size)
    since, until = _moment(buckets[0]), _moment(buckets[-1] + size)
    counts = storage.request_counts_by_bucket(since, until, size, path)

    # Highest-scoring check per (detector, bucket); checks come oldest first, so ties keep the first.
    best: dict[tuple[str, int], dict] = {}
    for check in storage.list_drift_checks(since, until, path):
        key = (check["kind"], _bucket_of(check["created_at"], size))
        if key not in best or check["score"] > best[key]["score"]:
            best[key] = check
    detectors = sorted({kind for kind, _ in best})

    def summary(kind: str, bucket: int) -> dict:
        check = best.get((kind, bucket))
        if check is None:
            return {"score": None, "threshold": None, "alert": False}
        return {
            "score": check["score"],
            "threshold": check["threshold"],
            "alert": check["score"] > check["threshold"],
        }

    return {
        "bucket_minutes": bucket_minutes,
        "since": storage.format_timestamp(since),
        "until": storage.format_timestamp(until),
        "detectors": detectors,
        "buckets": [
            {
                "start": storage.format_timestamp(_moment(b)),
                "request_count": counts.get(b, 0),
                **{kind: summary(kind, b) for kind in detectors},
            }
            for b in buckets
        ],
    }


def drifting(job: DriftJob, detector: Detector, limit: int) -> dict:
    """The latest window's most significant rising and falling IDs against the detector's baseline."""
    requests = job.latest_window()
    result: dict = {"window_size": job.window_size, "request_count": len(requests)}
    if len(requests) < job.window_size:
        return {
            **result,
            "first_request_id": None,
            "last_request_id": None,
            "rising": [],
            "falling": [],
        }

    sets = detector.to_sets(requests)
    drifts = rank_drift(detector, sets)
    rising = [d for d in drifts if d.ratio > 1][:limit]
    falling = [d for d in drifts if d.ratio < 1][:limit]
    return {
        **result,
        "first_request_id": requests[0]["id"],
        "last_request_id": requests[-1]["id"],
        "rising": describe_drift(detector, rising, requests, sets),
        "falling": describe_drift(detector, falling, requests, sets),
    }


def feature(
    feature_id: int,
    label: str,
    hours: int,
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES,
    path: Path | None = None,
) -> dict:
    """A feature's rate per bucket (per 1,000 requests, unsmoothed) and its latest strong prompts."""
    size = bucket_minutes * 60
    buckets = bucket_range(hours, utc_now(), size)
    since, until = _moment(buckets[0]), _moment(buckets[-1] + size)
    requests = storage.request_counts_by_bucket(since, until, size, path)
    hits = storage.feature_counts_by_bucket(feature_id, since, until, size, path)
    return {
        "feature_id": feature_id,
        "label": label,
        "bucket_minutes": bucket_minutes,
        "since": storage.format_timestamp(since),
        "until": storage.format_timestamp(until),
        "buckets": [
            {
                "start": storage.format_timestamp(_moment(b)),
                "request_count": requests.get(b, 0),
                "hits": hits.get(b, 0),
                "rate": hits.get(b, 0) / requests[b] * 1000 if requests.get(b) else None,
            }
            for b in buckets
        ],
        "strong_max_rank": STRONG_MAX_RANK,
        "strong_prompts": storage.strong_hits(feature_id, STRONG_MAX_RANK, STRONG_PROMPTS, path),
    }
