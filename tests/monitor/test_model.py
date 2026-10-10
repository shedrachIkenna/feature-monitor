from monitor.model import load_model_and_sae


def test_sae_input_size_matches_model_width() -> None:
    model, sae = load_model_and_sae()

    assert model.cfg.device == "cpu"
    assert sae.cfg.d_in == model.cfg.d_model