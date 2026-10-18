"""Write the dashboard's recorded example: one saved replay of medical_ramp, with and without the shift.

Reads the per-request scores saved by eval.multiseed (SAE) and eval.embed_baselines (mean
embedding distance, the live embedding detector) for seed 0, and writes
frontend/recorded_example.json for the overview page. Both SAE lines use the same normal
prompts in the same order (same seed), so the gap between them is the effect of the shift.

Scores are divided by each detector's saved threshold (1 = alarm level) and kept every
EVERY-th window; the first crossings are found on every window before thinning.

Run with: uv run python -m eval.recorded_example
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "frontend" / "recorded_example.json"
SCENARIO = "medical_ramp"
SEED = 0
EVERY = 5


def first_above(ratios: list[float], first_request: int) -> int | None:
    """Request number of the first window above the alarm level, or None."""
    return next((first_request + i for i, r in enumerate(ratios) if r > 1), None)


def build(multiseed: dict, embed: dict) -> dict:
    """The recorded example's payload from the two results files."""
    window = multiseed["config"]["window_size"]
    sae_threshold = multiseed["detectors"]["sae"]["saved_threshold"]
    mean_threshold = embed["detectors"]["mean"]["saved_threshold"]
    sae = multiseed["scores"]["sae"]
    shift = [s / sae_threshold for s in sae[f"{SCENARIO}/{SEED}"]]
    no_shift = [s / sae_threshold for s in sae[f"normal_only/{SEED}"]]
    mean = [s / mean_threshold for s in embed["scores"]["mean"][f"{SCENARIO}/{SEED}"]]
    return {
        "source": "eval/results/multiseed.json and eval/results/embed_baselines.json",
        "scenario": SCENARIO,
        "seed": SEED,
        "window_size": window,
        "shift_start": multiseed["shift_start"],
        "request_numbers": list(range(window, window + len(shift), EVERY)),
        "sae_shift": [round(r, 4) for r in shift[::EVERY]],
        "sae_no_shift": [round(r, 4) for r in no_shift[::EVERY]],
        "first_above": {
            "sae_shift": first_above(shift, window),
            "sae_no_shift": first_above(no_shift, window),
            "embedding_mean": first_above(mean, window),
        },
    }


def main() -> None:
    multiseed = json.loads((RESULTS_DIR / "multiseed.json").read_text())
    embed = json.loads((RESULTS_DIR / "embed_baselines.json").read_text())
    payload = build(multiseed, embed)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(payload['request_numbers'])} points per line)")
    print(f"first above the alarm level: {payload['first_above']}")


if __name__ == "__main__":
    main()
