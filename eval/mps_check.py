"""Compare the active profile's hook activations and top-20 SAE features on MPS versus CPU.

Run with: uv run python -m eval.mps_check (FEATURE_MONITOR_PROFILE picks the model)
Pass rule, fixed before the first run: on every sanity prompt the top-20 feature sets are
identical, and the largest activation difference is at most 0.1% of the largest activation.
Also reports per-prompt extraction time on each device, memory, and the SAE's mean L0 (active
features per token), which should be near the L0 the SAE was trained at if the model is loaded
the way the SAE expects.
"""

import json
import resource
import statistics
import time
from pathlib import Path

import torch
from sae_lens import SAE
from transformer_lens.model_bridge import TransformerBridge

from monitor.model import active_profile, load_model_and_sae

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_RELATIVE_DIFF = 1e-3
GIB = 1024**3


def load_prompts() -> list[str]:
    """Return all prompts from both sanity-check prompt sets."""
    prompts = []
    for name in ("sanity_prompts.json", "sanity_prompts_matched.json"):
        groups = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
        prompts += [prompt for group in groups.values() for prompt in group]
    return prompts


def activations_and_top20(
    model: TransformerBridge, sae: SAE, prompt: str
) -> tuple[torch.Tensor, set[int], float]:
    """Return a prompt's hook activations (float32, on CPU), top-20 feature IDs and mean L0."""
    hook = sae.cfg.metadata.hook_name
    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(prompt), names_filter=[hook])
        activations = cache[hook]
        features = sae.encode(activations)[0, 1:, :]
        feature_max = features.amax(dim=0)
        l0 = float((features > 0).sum(dim=1).float().mean())
    top20 = set(torch.topk(feature_max, 20).indices.cpu().tolist())
    return activations.float().cpu(), top20, l0


def sync(device: str) -> None:
    """Wait for queued GPU work, so timings include it."""
    if device == "mps":
        torch.mps.synchronize()


def main() -> None:
    """Run the comparison and print the verdict, timings and memory."""
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine.")
        return

    profile = active_profile()
    prompts = load_prompts()
    runs, times = {}, {}
    for device in ("cpu", "mps"):
        model, sae = load_model_and_sae(device)
        activations_and_top20(model, sae, prompts[0])  # warm-up, not timed
        runs[device], times[device] = [], []
        for prompt in prompts:
            sync(device)
            start = time.perf_counter()
            runs[device].append(activations_and_top20(model, sae, prompt))
            sync(device)
            times[device].append((time.perf_counter() - start) * 1000)
        if device == "mps":
            mps_gib = torch.mps.driver_allocated_memory() / GIB

    max_diff = max(float((c[0] - m[0]).abs().max()) for c, m in zip(runs["cpu"], runs["mps"]))
    max_activation = max(float(c[0].abs().max()) for c in runs["cpu"])
    same_top20 = sum(c[1] == m[1] for c, m in zip(runs["cpu"], runs["mps"]))
    overlap = [len(c[1] & m[1]) for c, m in zip(runs["cpu"], runs["mps"])]
    relative = max_diff / max_activation
    passed = same_top20 == len(prompts) and relative <= MAX_RELATIVE_DIFF
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB  # bytes on macOS

    print(
        f"profile {profile.name} ({profile.model_name}, {profile.dtype}), torch {torch.__version__}"
    )
    print(f"prompts: {len(prompts)}; identical top-20 sets: {same_top20}/{len(prompts)}")
    print(f"smallest top-20 overlap: {min(overlap)}/20")
    print(
        f"max abs activation difference: {max_diff:.3e} (largest activation {max_activation:.1f})"
    )
    print(f"relative difference: {relative:.3e} (limit {MAX_RELATIVE_DIFF:.0e})")
    for device in ("cpu", "mps"):
        l0 = statistics.mean(run[2] for run in runs[device])
        print(
            f"{device}: median {statistics.median(times[device]):.1f} ms per prompt, "
            f"max {max(times[device]):.1f} ms; mean L0 {l0:.1f}"
        )
    print(
        f"memory: MPS driver {mps_gib:.2f} GiB after the MPS run; process peak RSS {peak_rss_gib:.2f} GiB"
    )
    print(f"verdict: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
