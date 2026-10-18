"""Two checks before extracting the active profile offline, on CPU; each prints when it finishes.

1. Truncation: the model cut after the SAE's layer must give bit-identical hook activations and
   top-20 features to the full model, on the 40 sanity prompts, at the profile's dtype.
2. Precision: does bfloat16 change the top-20 features against float32? Over a seeded sample of
   normal_test prompts (the traffic the monitor sees), with the truncated model. Rule, fixed
   before the first run: bfloat16 is acceptable for extraction if the mean top-20 overlap with
   float32 is at least 19 of 20; otherwise extract in float32.

Run with: FEATURE_MONITOR_PROFILE=gemma2_2b uv run python -m eval.precision_check
"""

import dataclasses
import gc
import random
import statistics
import time
from collections.abc import Callable

import torch

from eval.mps_check import activations_and_top20, load_prompts
from monitor.calibrate import CHECK_POOL
from monitor.extract import FeatureHit, top_features
from monitor.model import Profile, active_profile, load_profile
from traffic.generate import load_pool

N_PROMPTS = 200
SEED = 0
MIN_MEAN_OVERLAP = 19
DTYPES = ("bfloat16", "float32")


def run(
    profile: Profile, truncate: bool, prompts: list[str], measure: Callable
) -> tuple[list, float]:
    """Load one variant alone in memory, apply measure to each prompt; returns results, median ms."""
    load_profile.cache_clear()  # two copies of the model may not fit in memory
    gc.collect()
    model, sae = load_profile(profile, truncate=truncate)
    results, times = [], []
    for prompt in prompts:
        start = time.perf_counter()
        results.append(measure(model, sae, prompt))
        times.append((time.perf_counter() - start) * 1000)
    load_profile.cache_clear()
    return results, statistics.median(times)


def truncation_check(profile: Profile) -> bool:
    """Full against truncated model: identical activations and top-20 features on every prompt."""
    prompts = load_prompts()

    def measure(model, sae, prompt) -> tuple[torch.Tensor, list[FeatureHit]]:
        return activations_and_top20(model, sae, prompt)[0], top_features(model, sae, prompt)

    full, full_ms = run(profile, False, prompts, measure)
    short, short_ms = run(profile, True, prompts, measure)
    same_acts = sum(torch.equal(f[0], s[0]) for f, s in zip(full, short, strict=True))
    same_hits = sum(f[1] == s[1] for f, s in zip(full, short, strict=True))
    print(f"1. truncation ({profile.dtype}, {len(prompts)} sanity prompts):", flush=True)
    print(f"   identical hook activations {same_acts}/{len(prompts)}, top-20 features {same_hits}")
    print(f"   median ms per prompt: full {full_ms:.1f}, truncated {short_ms:.1f}", flush=True)
    return same_acts == same_hits == len(prompts)


def precision_check(profile: Profile) -> None:
    """bfloat16 against float32 top-20 sets with the truncated model."""
    prompts = random.Random(SEED).sample(load_pool(CHECK_POOL), N_PROMPTS)
    top20, ms = {}, {}
    for dtype in DTYPES:

        def measure(model, sae, prompt) -> set[int]:
            return {h.feature_id for h in top_features(model, sae, prompt)}

        top20[dtype], ms[dtype] = run(
            dataclasses.replace(profile, dtype=dtype), True, prompts, measure
        )
        print(f"   {dtype} done: median {ms[dtype]:.1f} ms per prompt", flush=True)
    overlap = [len(a & b) for a, b in zip(top20["bfloat16"], top20["float32"], strict=True)]
    mean_overlap = statistics.mean(overlap)
    print(f"2. precision ({N_PROMPTS} {CHECK_POOL} prompts, seed {SEED}):")
    print(f"   identical top-20 sets {overlap.count(20)}/{N_PROMPTS}")
    print(f"   top-20 overlap: mean {mean_overlap:.2f}, smallest {min(overlap)}/20")
    passed = mean_overlap >= MIN_MEAN_OVERLAP
    print(f"   verdict: {'bfloat16 is fine' if passed else 'extract in float32'}", flush=True)


def main() -> None:
    """Run the truncation check, then the precision check if truncation is exact."""
    profile = active_profile()
    print(f"profile {profile.name} ({profile.model_name})", flush=True)
    if not truncation_check(profile):
        print("Truncation changes the activations; precision check not run.")
        return
    precision_check(profile)


if __name__ == "__main__":
    main()
