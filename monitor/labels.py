"""Attach human-readable labels to Neuronpedia feature IDs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from monitor.extract import extract_features

MODEL_ID = "gpt2-small"
SOURCE_ID = "8-res-jb"
CACHE_PATH = Path(__file__).resolve().with_name("labels_cache.json")
REQUEST_TIMEOUT_SECONDS = 10
RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 10.0


def _feature_url(feature_id: int | str) -> str:
    """Return the single-feature Neuronpedia endpoint for a feature ID."""
    return f"https://www.neuronpedia.org/api/feature/{MODEL_ID}/{SOURCE_ID}/{feature_id}"


def _read_cache() -> dict[str, str]:
    """Load the feature-to-label cache from disk, if present."""
    if not CACHE_PATH.exists():
        return {}

    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    cache: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            cache[str(key)] = value
    return cache


def _write_cache(cache: dict[str, str]) -> None:
    """Persist the label cache to a local JSON file."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _explanation_from_payload(payload: object) -> str | None:
    """Return the first explanation description in a Neuronpedia feature response."""
    if not isinstance(payload, dict) or not isinstance(payload.get("explanations"), list):
        return None
    for explanation in payload["explanations"]:
        if isinstance(explanation, dict):
            description = explanation.get("description")
            if isinstance(description, str) and description.strip():
                return description.strip()
    return None


def _fetch_feature_payload(feature_id: int | str) -> object:
    """Fetch a single feature's Neuronpedia payload."""
    request = urllib_request.Request(_feature_url(feature_id), method="GET")
    with urllib_request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _retry_delay(exc: urllib_error.HTTPError) -> float:
    """Return the pause before a retry, honouring a numeric Retry-After header up to a cap."""
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(max(float(retry_after), 0.0), MAX_RETRY_DELAY_SECONDS)
    except (TypeError, ValueError):
        return RETRY_DELAY_SECONDS


def _fetch_with_one_retry(feature_key: str) -> object | None:
    """Fetch a payload, pausing and retrying once on transient errors; None on failure."""
    for attempt in range(2):
        try:
            return _fetch_feature_payload(feature_key)
        except urllib_error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                return None
            delay = _retry_delay(exc)
        except OSError:  # timeouts and connection errors
            delay = RETRY_DELAY_SECONDS
        except ValueError:  # response was not valid JSON
            return None
        if attempt == 0:
            time.sleep(delay)
    return None


def get_label(feature_id: int | str) -> str:
    """Return the readable label for a feature or 'unlabelled' on failure."""
    feature_key = str(feature_id)
    cache = _read_cache()
    if feature_key in cache:
        return cache[feature_key]

    payload = _fetch_with_one_retry(feature_key)
    if payload is None:
        return "unlabelled"

    text = _explanation_from_payload(payload) or "unlabelled"
    cache[feature_key] = text
    _write_cache(cache)
    return text


def main(prompt: str, k: int = 10) -> None:
    """Print the top features for a prompt together with their labels."""
    for hit in extract_features(prompt, k=k):
        print(f"{hit.feature_id}: {get_label(hit.feature_id)}")


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "How do I reset my password?"
    main(text, k=10)
