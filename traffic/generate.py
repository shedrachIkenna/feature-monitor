"""Generate scenario traffic from the prompt pools, and send it to the API or write it to a file.

A scenario (traffic/scenarios/*.yaml) sets a seed, a request count and a schedule: points
(request index, shift fraction) joined by straight lines. Each block of 100 requests gets exactly
round(sum of the scheduled fractions) shift requests at seeded random positions, so every block
matches the schedule to within half a point. Each pool is drawn without replacement and reshuffled
when it runs out, so a small pool repeats evenly.

Every output row records `is_shift`, the ground truth for the evaluation.

    uv run python -m traffic.generate traffic/scenarios/medical_ramp.yaml --dry-run

A server that requires a record token (the Docker image does) refuses recorded prompts without
it; set FEATURE_MONITOR_RECORD_TOKEN to the server's token and it is sent with every request.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    # httpx is a dev dependency; the app imports this module for build_sequence without it.
    import httpx

DATA_DIR = Path(__file__).parent / "data"
RUNS_DIR = Path(__file__).parent / "runs"
BLOCK = 100
BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_S = 60.0
# Must match backend/app.py; repeated here so the generator does not import the model stack.
RECORD_TOKEN_ENV = "FEATURE_MONITOR_RECORD_TOKEN"
RECORD_TOKEN_HEADER = "X-Record-Token"


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    total_requests: int
    normal_pool: str
    shift_pool: str | None
    schedule: list[tuple[int, float]]


def load_scenario(path: Path) -> Scenario:
    """Read a scenario file and check its schedule covers 0..total_requests."""
    raw = yaml.safe_load(path.read_text())
    schedule = [(int(p["at"]), float(p["shift"])) for p in raw["schedule"]]
    scenario = Scenario(
        name=raw["name"],
        seed=int(raw["seed"]),
        total_requests=int(raw["total_requests"]),
        normal_pool=raw["normal_pool"],
        shift_pool=raw.get("shift_pool"),
        schedule=schedule,
    )
    points = [at for at, _ in schedule]
    if points[0] != 0 or points[-1] != scenario.total_requests:
        raise ValueError(f"{path}: schedule must run from 0 to {scenario.total_requests}")
    if any(b <= a for a, b in pairwise(points)):
        raise ValueError(f"{path}: schedule points must be strictly increasing")
    if any(not 0.0 <= s <= 1.0 for _, s in schedule):
        raise ValueError(f"{path}: shift fractions must be between 0 and 1")
    if scenario.shift_pool is None and any(s > 0 for _, s in schedule):
        raise ValueError(f"{path}: a non-zero shift needs a shift_pool")
    return scenario


def shift_fraction(schedule: list[tuple[int, float]], i: int) -> float:
    """Scheduled shift fraction at request `i`, interpolated linearly between points."""
    for (a, fa), (b, fb) in pairwise(schedule):
        if a <= i <= b:
            return fa + (fb - fa) * (i - a) / (b - a)
    raise ValueError(f"request {i} is outside the schedule")


def load_pool(name: str, data_dir: Path = DATA_DIR) -> list[str]:
    """Read one pool written by traffic/prepare_data.py."""
    with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def draws(pool: list[str], rng: random.Random) -> Iterator[str]:
    """Yield the pool in shuffled order, reshuffling each time it runs out."""
    while True:
        order = list(pool)
        rng.shuffle(order)
        yield from order


def build_sequence(scenario: Scenario, pools: dict[str, list[str]]) -> list[dict]:
    """The exact, seeded prompt sequence for a scenario, with ground truth per request."""
    positions_rng = random.Random(f"{scenario.seed}:positions")
    normal = draws(pools[scenario.normal_pool], random.Random(f"{scenario.seed}:normal"))
    shift = (
        draws(pools[scenario.shift_pool], random.Random(f"{scenario.seed}:shift"))
        if scenario.shift_pool
        else None
    )
    rows = []
    for start in range(0, scenario.total_requests, BLOCK):
        size = min(BLOCK, scenario.total_requests - start)
        expected = sum(shift_fraction(scenario.schedule, i) for i in range(start, start + size))
        n_shift = round(expected)
        flags = [True] * n_shift + [False] * (size - n_shift)
        positions_rng.shuffle(flags)
        for offset, is_shift in enumerate(flags):
            pool = scenario.shift_pool if is_shift else scenario.normal_pool
            rows.append(
                {
                    "index": start + offset,
                    "pool": pool,
                    "is_shift": is_shift,
                    "prompt": next(shift if is_shift else normal),
                }
            )
    return rows


def mix_per_block(scenario: Scenario, rows: list[dict]) -> list[tuple[int, float, float]]:
    """(block start, scheduled shift %, actual shift %) for each block of 100 requests."""
    table = []
    for start in range(0, len(rows), BLOCK):
        block = rows[start : start + BLOCK]
        scheduled = sum(shift_fraction(scenario.schedule, r["index"]) for r in block) / len(block)
        actual = sum(r["is_shift"] for r in block) / len(block)
        table.append((start, 100 * scheduled, 100 * actual))
    return table


def record_headers() -> dict[str, str]:
    """The record-token header, if FEATURE_MONITOR_RECORD_TOKEN is set."""
    token = os.environ.get(RECORD_TOKEN_ENV)
    return {RECORD_TOKEN_HEADER: token} if token else {}


def send(rows: list[dict], scenario_name: str, client: httpx.Client) -> Iterator[dict]:
    """POST each prompt to /chat with the scenario tag, one at a time; yield rows with results.

    Stops at the first 403: the server refuses to record, so every later request would fail too.
    """
    for row in rows:
        response = client.post("/chat", json={"prompt": row["prompt"], "scenario": scenario_name})
        if response.status_code == 403:
            raise SystemExit(
                f"/chat refused to record: {response.json().get('detail')} "
                f"Set {RECORD_TOKEN_ENV} to the server's token."
            )
        request_id = response.json()["request_id"] if response.status_code == 200 else None
        yield {**row, "status": response.status_code, "request_id": request_id}


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="write the sequence, send nothing")
    parser.add_argument("--out", type=Path, help="JSONL output (default traffic/runs/<name>.jsonl)")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    names = {scenario.normal_pool} | ({scenario.shift_pool} if scenario.shift_pool else set())
    pools = {name: load_pool(name, args.data_dir) for name in names}
    rows = build_sequence(scenario, pools)
    out = args.out or RUNS_DIR / f"{scenario.name}{'_dryrun' if args.dry_run else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with (
        out.open("w", encoding="utf-8") as f,
        httpx.Client(
            base_url=args.base_url, timeout=REQUEST_TIMEOUT_S, headers=record_headers()
        ) as client,
    ):
        results = iter(rows) if args.dry_run else send(rows, scenario.name, client)
        failed = 0
        for row in results:
            failed += row.get("status", 200) != 200
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sent = "" if args.dry_run else f", {failed} failed"
    print(f"{scenario.name}: {len(rows)} requests -> {out}{sent}")
    print(f"{'requests':>11}  {'scheduled %':>11}  {'actual %':>8}")
    for start, scheduled, actual in mix_per_block(scenario, rows):
        print(f"{start:>5}-{start + BLOCK - 1:<5}  {scheduled:>11.1f}  {actual:>8.1f}")


if __name__ == "__main__":
    main()
