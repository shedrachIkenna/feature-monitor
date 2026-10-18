"""Download everything the app needs at startup, so a container can start without Hugging Face.

`models` loads the active profile's model, its SAE and the MiniLM embedder once, which saves them
in the Hugging Face cache (HF_HOME). `labels` fills the profile's monitor/labels_cache_<profile>.json
with the Neuronpedia label of every feature in the SAE baseline; features outside it are still
fetched on first use.

Run with: uv run python -m monitor.prefetch models labels
"""

import argparse

from monitor.baseline import baseline_path, load_baseline
from monitor.embed_drift import load_embedder
from monitor.labels import prefetch_labels
from monitor.model import load_model_and_sae


def baseline_feature_ids() -> list[int]:
    """Every feature that appears in the saved SAE baseline."""
    _, baseline = load_baseline(baseline_path())
    return sorted(baseline["counts"])


def main(steps: list[str]) -> None:
    """Run the requested download steps in order."""
    if "models" in steps:
        load_model_and_sae()
        load_embedder()
        print("Models cached.")
    if "labels" in steps:
        cached, fetched, failed = prefetch_labels(baseline_feature_ids())
        print(f"Labels: {cached} already cached, {fetched} fetched, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="+", choices=["models", "labels"])
    main(parser.parse_args().steps)
