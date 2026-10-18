"""First end-to-end run: send each scenario through the real app and record when each detector alerts.

For each scenario, the app (the model, SAE extraction and the scheduled drift job) starts in this
process on a fresh database, the scenario's requests are sent one at a time, and the run ends once
the job has checked the window holding the last request. A request number is the 1-based position
in the scenario. Checks run every check_interval_seconds of wall-clock time, so the first-alert
request number is only accurate to roughly one check interval's worth of requests.

Run with: uv run python -m eval.first_run
"""

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn

from backend import storage
from backend.app import ModelEngine, create_app
from monitor.config import load_config
from monitor.job import DriftJob, load_drift_job
from traffic.generate import build_sequence, load_pool, load_scenario, send

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "traffic" / "scenarios"
SCENARIOS = ("normal_only", "medical_ramp", "new_topic_ramp", "injection_ramp")
DB_DIR = ROOT / "data" / "first_run"
OUT_PATH = ROOT / "eval" / "results" / "first_run.json"
PORT = 8765
STARTUP_TIMEOUT_S = 600
FINAL_CHECK_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 60.0


def wait_until(condition, timeout_s: float, what: str) -> None:
    """Poll condition() every second; raise if it is still false after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(1)


def summarise(rows: list[dict], job: DriftJob, checks: list[dict], alerts: list[dict]) -> dict:
    """Per detector: alert count and the first alert's request number, shift share and top 5."""
    number_of = {r["request_id"]: r["index"] + 1 for r in rows}
    shift_of = {r["request_id"]: r["is_shift"] for r in rows}
    # Stored checks hold one row per detector; group them by the window they scored.
    scores: dict[int, dict[str, float]] = {}
    for c in checks:
        scores.setdefault(c["last_request_id"], {})[c["kind"]] = c["score"]
    detectors = {}
    for detector in job.detectors:
        mine = sorted(
            (a for a in alerts if a["kind"] == detector.name),
            key=lambda a: a["details"]["last_request_id"],
        )
        first = mine[0] if mine else None
        detectors[detector.name] = {
            "threshold": detector.threshold,
            "alerts": len(mine),
            "alert_request_numbers": [number_of[a["details"]["last_request_id"]] for a in mine],
            "first_alert_request_number": (
                number_of[first["details"]["last_request_id"]] if first else None
            ),
            "first_alert_score": first["score"] if first else None,
            "first_alert_window_shift_share": (
                sum(
                    shift_of[i]
                    for i in range(
                        first["details"]["first_request_id"],
                        first["details"]["last_request_id"] + 1,
                    )
                )
                / first["details"]["window_size"]
                if first
                else None
            ),
            "first_alert_top": first["details"]["top"] if first else None,
        }
    return {
        "checks": [{"request_number": number_of[i], "scores": s} for i, s in scores.items()],
        "detectors": detectors,
    }


def run_scenario(name: str) -> dict:
    """Serve a fresh app, send the scenario, wait for the final check, and summarise alerts."""
    db = DB_DIR / f"{name}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    os.environ[storage.DB_PATH_ENV] = str(db)

    job = load_drift_job()
    server = uvicorn.Server(
        uvicorn.Config(create_app(ModelEngine, lambda: job), port=PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    start = time.monotonic()
    try:
        wait_until(lambda: server.started, STARTUP_TIMEOUT_S, "the server to start")
        scenario = load_scenario(SCENARIO_DIR / f"{name}.yaml")
        names = {scenario.normal_pool} | ({scenario.shift_pool} if scenario.shift_pool else set())
        sequence = build_sequence(scenario, {n: load_pool(n) for n in names})
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=REQUEST_TIMEOUT_S) as client:
            rows = list(send(sequence, name, client))
        failed = sum(r["status"] != 200 for r in rows)
        if failed:
            raise RuntimeError(f"{name}: {failed} requests failed")
        last_id = rows[-1]["request_id"]

        def final_check_done() -> bool:
            checks = storage.list_drift_checks(path=db)
            return bool(checks) and checks[-1]["last_request_id"] == last_id

        wait_until(
            final_check_done,
            FINAL_CHECK_TIMEOUT_S,
            "the final check",
        )
    finally:
        server.should_exit = True
        thread.join()

    alerts = storage.list_alerts(limit=100_000, path=db)
    return {
        "scenario": name,
        "requests": len(rows),
        "minutes": round((time.monotonic() - start) / 60, 1),
        **summarise(rows, job, storage.list_drift_checks(path=db), alerts),
    }


def main() -> None:
    """Run every scenario, save the results, and print the first-alert table."""
    config = load_config()
    results = []
    for name in SCENARIOS:
        print(f"Running {name} ...", flush=True)
        results.append(run_scenario(name))
        OUT_PATH.write_text(
            json.dumps(
                {
                    "created": datetime.now(UTC).isoformat(timespec="seconds"),
                    "config": vars(config),
                    "scenarios": results,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Saved {OUT_PATH}")
    print("scenario         checks  detector   alerts  first alert at request  shift in window")
    for r in results:
        for detector, d in r["detectors"].items():
            share = d["first_alert_window_shift_share"]
            print(
                f"{r['scenario']:<16} {len(r['checks']):>6}  {detector:<9}  {d['alerts']:>6}"
                f"  {d['first_alert_request_number'] or '-':>22}"
                f"  {'-' if share is None else f'{share:.0%}':>15}"
            )


if __name__ == "__main__":
    main()
