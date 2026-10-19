"""Load test: time 100 sequential /chat requests with feature extraction on and off.

Each mode starts its own uvicorn server on a temporary database, so the real database is
untouched. Generation time and extraction time come from the stored rows; total time is
the client's wall-clock time for the whole HTTP request.

Results go to eval/results/latency_<profile>.json (FEATURE_MONITOR_PROFILE picks the model; the
server runs the full model, as the live app does).

Run with: uv run python -m eval.latency
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx

from backend.app import EXTRACT_ENV, MAX_NEW_TOKENS, SEED
from backend.storage import DB_PATH_ENV
from monitor.model import active_profile

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT / "eval" / "data" / "latency_prompts.json"
RESULTS_DIR = ROOT / "eval" / "results"
PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 120
EXPECTED_HITS = {True: 20, False: 0}
METRICS = ("gen_ms", "extract_ms", "total_ms")
PACKAGES = ("torch", "transformer-lens", "sae-lens", "fastapi", "uvicorn", "httpx")


def results_path() -> Path:
    """Return eval/results/latency_<profile>.json for the active profile."""
    return RESULTS_DIR / f"latency_{active_profile().name}.json"


def load_prompts(path: Path = PROMPTS_PATH) -> list[str]:
    """Return every prompt in the file, topic by topic, in file order."""
    topics: dict[str, list[str]] = json.loads(path.read_text())
    return [prompt for prompts in topics.values() for prompt in prompts]


def percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0-100) with linear interpolation, as numpy does by default."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return the median and 95th percentile of each timing metric."""
    return {
        metric: {
            "p50": percentile([row[metric] for row in rows], 50),
            "p95": percentile([row[metric] for row in rows], 95),
        }
        for metric in METRICS
    }


def overhead_pct(with_extract: float, without_extract: float) -> float:
    """Return how much longer `with_extract` is than `without_extract`, in percent."""
    return (with_extract - without_extract) / without_extract * 100


def _start_server(extract: bool, db_path: Path, log_path: Path) -> subprocess.Popen:
    """Start uvicorn with the extraction flag and database set, and wait until healthy."""
    env = {**os.environ, EXTRACT_ENV: "1" if extract else "0", DB_PATH_ENV: str(db_path)}
    log = log_path.open("w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--port", str(PORT)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            if httpx.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                return server
        except httpx.TransportError:
            pass
        time.sleep(1)
    server.terminate()
    raise RuntimeError(f"server not healthy after {STARTUP_TIMEOUT_S} s; see {log_path}")


def _stored_rows(db_path: Path, request_ids: list[int]) -> dict[int, dict]:
    """Return each stored request's timings and hit count, keyed by request id."""
    placeholders = ", ".join("?" * len(request_ids))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT r.id, r.gen_ms, r.extract_ms, COUNT(h.request_id) FROM requests r"
            " LEFT JOIN feature_hits h ON h.request_id = r.id"
            f" WHERE r.id IN ({placeholders}) GROUP BY r.id",
            request_ids,
        ).fetchall()
    return {
        request_id: {"gen_ms": gen_ms, "extract_ms": extract_ms, "hits": hits}
        for request_id, gen_ms, extract_ms, hits in rows
    }


def run_mode(extract: bool, prompts: list[str], workdir: Path) -> dict:
    """Send one untimed warm-up request, then every prompt in turn; check and summarise."""
    db_path = workdir / f"latency_extract_{int(extract)}.db"
    server = _start_server(extract, db_path, workdir / f"server_extract_{int(extract)}.log")
    try:
        with httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT_S) as client:
            health = client.get("/health").json()
            if health["extract"] is not extract:
                raise RuntimeError(f"server reports extract={health['extract']}, wanted {extract}")
            client.post("/chat", json={"prompt": "Hello, this is a warm-up request."})

            responses = []
            for prompt in prompts:
                start = time.perf_counter()
                response = client.post("/chat", json={"prompt": prompt, "scenario": "latency"})
                total_ms = (time.perf_counter() - start) * 1000
                request_id = response.json().get("request_id") if response.is_success else None
                responses.append(
                    {"status": response.status_code, "request_id": request_id, "total_ms": total_ms}
                )
    finally:
        server.terminate()
        server.wait(timeout=30)

    ids = [r["request_id"] for r in responses if r["request_id"] is not None]
    stored = _stored_rows(db_path, ids) if ids else {}
    rows = [{**r, **stored[r["request_id"]]} for r in responses if r["request_id"] in stored]
    checks = {
        "all_status_200": all(r["status"] == 200 for r in responses),
        "all_rows_stored": len(rows) == len(prompts),
        f"all_hit_counts_{EXPECTED_HITS[extract]}": all(
            r["hits"] == EXPECTED_HITS[extract] for r in rows
        ),
    }
    return {
        "extract": extract,
        "device": health["device"],
        "model": health["model"],
        "requests": len(responses),
        "checks": checks,
        "summary": summarise(rows) if rows else {},
        "per_request": rows,
    }


def _print_table(on: dict, off: dict, overhead: dict[str, float]) -> None:
    """Print p50 and p95 for each metric in both modes, then the overhead."""
    print(f"\nDevice: {on['device']}   requests per mode: {on['requests']}")
    print(f"{'metric':<12}{'on p50':>10}{'on p95':>10}{'off p50':>10}{'off p95':>10}   (ms)")
    for metric in METRICS:
        a, b = on["summary"][metric], off["summary"][metric]
        print(f"{metric:<12}{a['p50']:>10.1f}{a['p95']:>10.1f}{b['p50']:>10.1f}{b['p95']:>10.1f}")
    print(
        f"Extraction overhead on total: {overhead['p50']:+.1f}% at p50,"
        f" {overhead['p95']:+.1f}% at p95"
    )
    for mode in (on, off):
        print(f"Checks (extract={mode['extract']}): {mode['checks']}")


def main() -> None:
    """Run both modes, save the results file and print a summary table."""
    prompts = load_prompts()
    with tempfile.TemporaryDirectory() as tmp:
        on = run_mode(True, prompts, Path(tmp))
        off = run_mode(False, prompts, Path(tmp))

    passed = all(on["checks"].values()) and all(off["checks"].values())
    overhead = (
        {
            q: overhead_pct(on["summary"]["total_ms"][q], off["summary"]["total_ms"][q])
            for q in ("p50", "p95")
        }
        if passed
        else {}
    )
    results = {
        "created_at": datetime.now(UTC).isoformat(),
        "profile": active_profile().name,
        "prompts_file": str(PROMPTS_PATH.relative_to(ROOT)),
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "packages": {name: version(name) for name in PACKAGES},
        "checks_passed": passed,
        "total_overhead_pct": overhead,
        "extract_on": on,
        "extract_off": off,
    }
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {path.relative_to(ROOT)}")

    if not passed:
        print(f"Checks failed: on={on['checks']} off={off['checks']}")
        sys.exit(1)
    _print_table(on, off, overhead)


if __name__ == "__main__":
    main()
