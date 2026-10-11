"""Attach human-readable labels to Neuronpedia feature IDs."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from monitor.extract import FeatureHit, extract_features

MODEL_ID = "gpt2-small"
SOURCE_ID = "8-res-jb"
CACHE_PATH = Path(__file__).resolve().with_name("labels_cache.json")
REQUEST_TIMEOUT_SECONDS = 10
GENERIC_RELEASE_LABEL = (
    "Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small"
)


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


def _extract_explanation_text(payload: object) -> str | None:
    """Pull the human-readable explanation text out of a Neuronpedia response."""
    if isinstance(payload, dict):
        explanations = payload.get("explanations")
        if isinstance(explanations, list):
            for explanation in explanations:
                if isinstance(explanation, dict):
                    description = explanation.get("description")
                    if isinstance(description, str) and description.strip():
                        return description.strip()

        for key in ("explanation", "label", "text", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        json_details = payload.get("jsonDetails")
        if json_details is not None:
            nested_text = _extract_explanation_text(json_details)
            if nested_text:
                return nested_text

        for value in payload.values():
            found = _extract_explanation_text(value)
            if found:
                return found
        return None

    if isinstance(payload, list):
        for item in payload:
            found = _extract_explanation_text(item)
            if found:
                return found
        return None

    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return _extract_explanation_text(parsed)

    return None


def _fetch_feature_payload(feature_id: int | str) -> object:
    """Fetch a single feature's Neuronpedia payload."""
    request = urllib_request.Request(_feature_url(feature_id), method="GET")
    with urllib_request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def get_label(feature_id: int | str) -> str:
    """Return the readable label for a feature or 'unlabelled' on failure."""
    feature_key = str(feature_id)
    cache = _read_cache()
    if feature_key in cache and cache[feature_key] != GENERIC_RELEASE_LABEL:
        return cache[feature_key]

    for _ in range(2):
        try:
            payload = _fetch_feature_payload(feature_key)
        except urllib_error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                return "unlabelled"
            continue
        except OSError:  # timeouts and connection errors
            continue
        except ValueError:  # response was not valid JSON
            return "unlabelled"

        label = _extract_explanation_text(payload)
        text = (label or "").strip()
        if not text or text == GENERIC_RELEASE_LABEL:
            text = "unlabelled"
        cache[feature_key] = text
        _write_cache(cache)
        return text

    return "unlabelled"


def _top_prompt_features(prompt: str, k: int = 10) -> list[FeatureHit]:
    """Return the top prompt features, capped to the requested count."""
    return extract_features(prompt, k=k)


def main(prompt: str, k: int = 10) -> None:
    """Print the top features for a prompt together with their labels."""
    for hit in _top_prompt_features(prompt, k=k):
        print(f"{hit.feature_id}: {get_label(hit.feature_id)}")


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "How do I reset my password?"
    main(text, k=10)
