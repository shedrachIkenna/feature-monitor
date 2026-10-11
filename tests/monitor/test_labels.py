import json
from typing import Self
from urllib import error as urllib_error

from monitor import labels


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_get_label_fetches_and_caches_label(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(labels, "CACHE_PATH", tmp_path / "labels.json")

    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return FakeResponse({"jsonDetails": '[{"text": "economic policy and inflation"}]'})

    monkeypatch.setattr(labels.urllib_request, "urlopen", fake_urlopen)

    assert labels.get_label(7) == "economic policy and inflation"
    assert labels.get_label(7) == "economic policy and inflation"
    assert calls == [labels._feature_url(7)]
    cached = json.loads((tmp_path / "labels.json").read_text())
    assert cached["7"] == "economic policy and inflation"


def test_get_label_refreshes_generic_release_cache(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "labels.json"
    cache_path.write_text(json.dumps({"7": labels.GENERIC_RELEASE_LABEL}))
    monkeypatch.setattr(labels, "CACHE_PATH", cache_path)
    monkeypatch.setattr(
        labels,
        "_fetch_feature_payload",
        lambda feature_id: {
            "description": labels.GENERIC_RELEASE_LABEL,
            "explanations": [{"description": "medical terms and inflammation"}],
        },
    )

    assert labels.get_label(7) == "medical terms and inflammation"
    cached = json.loads(cache_path.read_text())
    assert cached["7"] == "medical terms and inflammation"


def test_get_label_retries_once_then_returns_unlabelled(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "labels.json"
    monkeypatch.setattr(labels, "CACHE_PATH", cache_path)

    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) % 2 == 1:
            raise TimeoutError("request timed out")
        raise urllib_error.HTTPError(request.full_url, 429, "rate limited", hdrs=None, fp=None)

    monkeypatch.setattr(labels.urllib_request, "urlopen", fake_urlopen)

    assert labels.get_label(12) == "unlabelled"
    assert len(calls) == 2
    assert not cache_path.exists()

    assert labels.get_label(12) == "unlabelled"
    assert len(calls) == 4


def _count_calls_for_http_error(monkeypatch, tmp_path, code: int) -> int:
    monkeypatch.setattr(labels, "CACHE_PATH", tmp_path / "labels.json")
    calls: list[str] = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib_error.HTTPError(request.full_url, code, "error", hdrs=None, fp=None)

    monkeypatch.setattr(labels.urllib_request, "urlopen", fake_urlopen)
    assert labels.get_label(12) == "unlabelled"
    return len(calls)


def test_get_label_does_not_retry_permanent_http_error(monkeypatch, tmp_path) -> None:
    assert _count_calls_for_http_error(monkeypatch, tmp_path, 404) == 1


def test_get_label_retries_server_error_once(monkeypatch, tmp_path) -> None:
    assert _count_calls_for_http_error(monkeypatch, tmp_path, 503) == 2


def test_main_prints_top_prompt_features_with_labels(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        labels,
        "extract_features",
        lambda prompt, k=10: [
            labels.FeatureHit(feature_id=3, activation=0.9),
            labels.FeatureHit(feature_id=9, activation=0.7),
        ],
    )
    monkeypatch.setattr(
        labels,
        "get_label",
        lambda feature_id: "money and trade" if feature_id == 3 else "unlabelled",
    )

    labels.main("What is the economy?")

    captured = capsys.readouterr().out
    assert "3" in captured
    assert "money and trade" in captured
    assert "9" in captured
