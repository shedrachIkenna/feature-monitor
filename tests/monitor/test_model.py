import pytest
import torch

from monitor.model import load_model_and_sae


@pytest.mark.requires_model
def test_sae_input_size_matches_model_width() -> None:
    model, sae = load_model_and_sae()

    assert model.cfg.device == "cpu"
    assert sae.cfg.d_in == model.cfg.d_model


@pytest.mark.requires_model
@pytest.mark.filterwarnings("ignore:HookedTransformer.*deprecated:DeprecationWarning")
def test_bridge_matches_deprecated_loader() -> None:
    """Delete this test once TransformerLens 4.0 removes HookedTransformer."""
    from transformer_lens import HookedTransformer

    model, sae = load_model_and_sae()
    reference = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    hook = sae.cfg.metadata.hook_name

    for prompt in ["How do I reset my password?", "What causes a migraine?"]:
        _, bridge_cache = model.run_with_cache(prompt, names_filter=[hook])
        _, reference_cache = reference.run_with_cache(prompt, names_filter=[hook])
        torch.testing.assert_close(bridge_cache[hook], reference_cache[hook])
