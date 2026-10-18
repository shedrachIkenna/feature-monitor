import csv

from eval.plot_experiment import load_sweep, plot_tradeoff
from eval.plot_multiseed import PANELS
from eval.run_experiment import SWEEP_FIELDS


def test_sweep_csv_loads_and_plots_including_missed_scenarios(tmp_path) -> None:
    path = tmp_path / "sweep.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, SWEEP_FIELDS)
        writer.writeheader()
        for profile, detector in (("gpt2_small", "sae"), ("gpt2_small", "embedding_mean")):
            for percentile, rate, median in ((99.0, 0.01, 80), (100.0, 0.0, "")):
                for scenario in PANELS:
                    writer.writerow(
                        {
                            "profile": profile,
                            "detector": detector,
                            "percentile": percentile,
                            "threshold": 0.1,
                            "fa_alerts_per_1000": 0,
                            "fa_run_rate": 0,
                            "fa_window_rate": rate,
                            "scenario": scenario,
                            "detected": 5,
                            "runs": 10,
                            "median_delay": median,  # "" when half or more runs missed
                            "min_delay": 70,
                            "max_delay": 90,
                        }
                    )
    rows = load_sweep(path)
    assert rows[0]["median_delay"] == 80.0 and rows[-1]["median_delay"] is None

    out = tmp_path / "tradeoff.png"
    plot_tradeoff(rows, out)  # Gemma rows absent: its series is skipped
    assert out.read_bytes().startswith(b"\x89PNG")
