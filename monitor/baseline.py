"""Build the normal-traffic feature baseline that drift detection compares against.

For each SAE feature, the baseline records how many of the sampled prompts had it in their
top k, and that count as a rate per 1,000 prompts.

Run with: uv run python -m monitor.baseline
"""

import json
import random
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from monitor.extract import FeatureHit, extract_features
from monitor.labels import get_label
from monitor.model import MODEL_NAME, SAE_ID, SAE_RELEASE

SEED = 0
N_PROMPTS = 3000
K = 20
POOL = "normal_train"
POOL_DIR = Path(__file__).resolve().parent.parent / "traffic" / "data"
BASELINES_DIR = Path(__file__).resolve().with_name("baselines")
LAYER = int(SAE_ID.split(".")[1])  # "blocks.8.hook_resid_pre" -> 8


def read_pool(path: Path) -> list[str]:
    """Read a JSONL pool of {"text": ...} rows written by traffic/prepare_data.py."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def sample_prompts(pool: list[str], n: int, seed: int = SEED) -> list[str]:
    """Return n distinct prompts drawn from the pool with a seeded generator."""
    if n > len(pool):
        raise ValueError(f"asked for {n} prompts but the pool has only {len(pool)}")
    return random.Random(seed).sample(pool, n)


def build_baseline(
    prompts: Iterable[str],
    extract: Callable[[str, int], list[FeatureHit]] = extract_features,
    k: int = K,
) -> dict:
    """Count each feature's top-k appearances across prompts and convert to rates per 1,000."""
    counts: Counter[int] = Counter()
    prompt_count = 0
    for prompt in prompts:
        counts.update({hit.feature_id for hit in extract(prompt, k)})
        prompt_count += 1
    if prompt_count == 0:
        raise ValueError("a baseline needs at least one prompt")
    return {
        "prompt_count": prompt_count,
        "counts": dict(sorted(counts.items())),
        "rate_per_1000": {
            feature_id: count / prompt_count * 1000 for feature_id, count in sorted(counts.items())
        },
    }


def save_baseline(baseline: dict, metadata: dict, path: Path) -> None:
    """Write metadata and baseline numbers to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, **baseline}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> tuple[dict, dict]:
    """Read a baseline file; returns (metadata, baseline) with integer feature IDs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline = {
        "prompt_count": payload["prompt_count"],
        "counts": {int(f): c for f, c in payload["counts"].items()},
        "rate_per_1000": {int(f): r for f, r in payload["rate_per_1000"].items()},
    }
    return payload["metadata"], baseline


def baseline_path(model: str = MODEL_NAME, layer: int = LAYER) -> Path:
    """Return monitor/baselines/<model>_<layer>.json."""
    return BASELINES_DIR / f"{model}_{layer}.json"


def main() -> None:
    """Build, save and reload the baseline, then print its 20 most common features."""
    prompts = sample_prompts(read_pool(POOL_DIR / f"{POOL}.jsonl"), N_PROMPTS)
    baseline = build_baseline(prompts)
    metadata = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": LAYER,
        "k": K,
        "pool": POOL,
        "prompt_count": baseline["prompt_count"],
        "seed": SEED,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = baseline_path()
    save_baseline(baseline, metadata, path)
    _, reloaded = load_baseline(path)
    if reloaded != baseline:
        raise RuntimeError("reloaded baseline differs from the one just built")
    print(f"Saved {path} ({len(baseline['counts'])} distinct features, reload matches)")

    top = sorted(baseline["counts"].items(), key=lambda item: (-item[1], item[0]))[:20]
    for feature_id, count in top:
        rate = baseline["rate_per_1000"][feature_id]
        print(f"{feature_id:>6}  {rate:6.1f}/1000  {get_label(feature_id)}")


if __name__ == "__main__":
    main()
