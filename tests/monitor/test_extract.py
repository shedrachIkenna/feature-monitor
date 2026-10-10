from monitor.extract import extract_features
from monitor.model import load_model_and_sae


def test_extract_features_returns_top_valid_deterministic_hits() -> None:
    prompt = "How do I reset my password?"
    model, sae = load_model_and_sae()

    hits = extract_features(prompt)
    repeated_hits = extract_features(prompt)
    five_hits = extract_features(prompt, k=5)

    assert len(hits) == 20
    assert len(hits) <= 20
    assert len(five_hits) <= 5
    assert all(0 <= hit.feature_id < sae.cfg.d_sae for hit in hits)
    assert all(hit.activation > 0 for hit in hits)
    assert [hit.activation for hit in hits] == sorted(
        (hit.activation for hit in hits), reverse=True
    )
    assert hits == repeated_hits
    assert model.cfg.device == sae.cfg.device
