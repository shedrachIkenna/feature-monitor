"""Extract sparse feature activations from prompts."""

import re
from dataclasses import dataclass

import torch
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

from monitor.model import load_model_and_sae


@dataclass(frozen=True)
class FeatureHit:
    """One SAE feature and its maximum activation across prompt tokens."""

    feature_id: int
    activation: float


def extract_features(prompt: str, k: int = 20) -> list[FeatureHit]:
    """Return the active profile's top positive SAE features for a prompt."""
    model, sae = load_model_and_sae()
    return top_features(model, sae, prompt, k)


def top_features(model: TransformerBridge, sae: SAE, prompt: str, k: int = 20) -> list[FeatureHit]:
    """Return the top positive SAE features after excluding the BOS token."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    hook_point = sae.cfg.metadata.hook_name
    if hook_point is None:
        raise ValueError("The loaded SAE does not specify an activation hook")
    # Hooks of block N fire only if blocks 0..N run, so stop there; later blocks change nothing.
    match = re.match(r"blocks\.(\d+)\.", hook_point)
    stop_at_layer = int(match.group(1)) + 1 if match else None

    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=[hook_point], stop_at_layer=stop_at_layer
        )
        activations = cache[hook_point]
        feature_activations = sae.encode(activations)[0, 1:, :]
        if feature_activations.shape[0] == 0:
            return []

        max_activations = feature_activations.amax(dim=0)
        active_ids = torch.nonzero(max_activations > 0, as_tuple=True)[0]
        if active_ids.numel() == 0:
            return []

        count = min(k, active_ids.numel())
        values, indices = torch.topk(max_activations[active_ids], count, largest=True, sorted=True)
        feature_ids = active_ids[indices]

    return [
        FeatureHit(feature_id=int(feature_id.item()), activation=float(value.item()))
        for feature_id, value in zip(feature_ids, values, strict=True)
    ]
