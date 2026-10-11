"""Compare layer-8 activations and top-20 SAE features on MPS versus CPU.

Run with: uv run python -m eval.mps_check
Pass rule, fixed before the first run: on every sanity prompt the top-20 feature sets are
identical, and the largest activation difference is at most 0.1% of the largest activation.
"""

import json
from pathlib import Path

import torch
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

from monitor.model import MODEL_NAME, SAE_ID, SAE_RELEASE

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_RELATIVE_DIFF = 1e-3


def load_prompts() -> list[str]:
    """Return all prompts from both sanity-check prompt sets."""
    prompts = []
    for name in ("sanity_prompts.json", "sanity_prompts_matched.json"):
        groups = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
        prompts += [prompt for group in groups.values() for prompt in group]
    return prompts


def activations_and_top20(
    model: TransformerBridge, sae: SAE, prompt: str
) -> tuple[torch.Tensor, set[int]]:
    """Return a prompt's hook activations (on CPU) and its top-20 feature IDs."""
    hook = sae.cfg.metadata.hook_name
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(prompt), names_filter=[hook])
        activations = cache[hook]
        feature_max = sae.encode(activations)[0, 1:, :].amax(dim=0)
    return activations.cpu(), set(torch.topk(feature_max, 20).indices.cpu().tolist())


def main() -> None:
    """Run the comparison and print the verdict."""
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine.")
        return

    runs = {}
    for device in ("cpu", "mps"):
        model = TransformerBridge.boot_transformers(MODEL_NAME, device=device)
        model.enable_compatibility_mode()
        sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
        runs[device] = [activations_and_top20(model, sae, p) for p in load_prompts()]

    max_diff = max(float((c[0] - m[0]).abs().max()) for c, m in zip(runs["cpu"], runs["mps"]))
    max_activation = max(float(c[0].abs().max()) for c in runs["cpu"])
    same_top20 = sum(c[1] == m[1] for c, m in zip(runs["cpu"], runs["mps"]))
    relative = max_diff / max_activation
    passed = same_top20 == len(runs["cpu"]) and relative <= MAX_RELATIVE_DIFF

    print(f"torch {torch.__version__}, prompts: {len(runs['cpu'])}")
    print(f"identical top-20 sets: {same_top20}/{len(runs['cpu'])}")
    print(
        f"max abs activation difference: {max_diff:.3e} (largest activation {max_activation:.1f})"
    )
    print(f"relative difference: {relative:.3e} (limit {MAX_RELATIVE_DIFF:.0e})")
    print(f"verdict: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
