"""Demo traffic for the live Space: a trickle of normal prompts, and drift scenarios on request.

One daemon thread sends prompts straight to the engine (no HTTP), so the /chat queue cap and
rate limit never see them, and stores them like any recorded request. It only starts a prompt
when no visitor's /chat is running or waiting, so a visitor waits at most for the one prompt
already being answered.

What it sends next, in order:
1. the next prompt of a running scenario, as fast as the model allows;
2. while fewer than `warmup_target` requests are stored (after a restart the database is
   empty), a normal prompt, as fast as the model allows, so drift checks can start;
3. otherwise a normal prompt every `interval_s` seconds.

Normal prompts come from `normal_test`, never `normal_train`, which built the baseline.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from backend import storage
from traffic.generate import DATA_DIR, build_sequence, draws, load_pool, load_scenario

if TYPE_CHECKING:
    from backend.app import Engine, EngineResult

logger = logging.getLogger(__name__)

DEMO_TRAFFIC_ENV = "FEATURE_MONITOR_DEMO_TRAFFIC"
INTERVAL_ENV = "FEATURE_MONITOR_DEMO_INTERVAL_SECONDS"
DEFAULT_INTERVAL_S = 20.0
SCENARIO_DIR = Path(__file__).resolve().parent.parent / "traffic" / "scenarios"
NORMAL_POOL = "normal_test"
# Stored in requests.scenario, so demo rows can be told apart from scenario and visitor rows.
TRICKLE_LABEL = "demo_normal"
SEED = 0
# How often the thread looks again while a visitor is being served or nothing is due.
POLL_S = 0.2


def demo_enabled() -> bool:
    """True only when the demo traffic flag is set to "1" (the Docker image sets it)."""
    return os.environ.get(DEMO_TRAFFIC_ENV) == "1"


def demo_interval() -> float:
    """Seconds between trickle prompts once warmed up; must be positive."""
    value = float(os.environ.get(INTERVAL_ENV, DEFAULT_INTERVAL_S))
    if value <= 0:
        raise ValueError(f"{INTERVAL_ENV} must be positive, got {value}")
    return value


def scenario_names(scenario_dir: Path | None = None) -> list[str]:
    """Names of the scenarios that can be started, from traffic/scenarios/*.yaml."""
    return sorted(p.stem for p in (scenario_dir or SCENARIO_DIR).glob("*.yaml"))


class DemoTraffic:
    """Sends the trickle and scenarios from a background thread; see the module docstring."""

    def __init__(
        self,
        engine: Engine,
        busy: Callable[[], bool],
        warmup_target: int,
        interval_s: float,
        on_prompt: Callable[[str, float, EngineResult], None] | None = None,
        data_dir: Path | None = None,
        scenario_dir: Path | None = None,
    ) -> None:
        self.engine = engine
        self.busy = busy
        self.warmup_target = warmup_target
        self.interval_s = interval_s
        self.on_prompt = on_prompt
        self.data_dir = data_dir or DATA_DIR
        self.scenario_dir = scenario_dir or SCENARIO_DIR
        self._normal: Iterator[str] = draws(
            load_pool(NORMAL_POOL, self.data_dir), random.Random(f"{SEED}:demo")
        )
        self._lock = threading.Lock()
        self._scenario: str | None = None
        self._rows: list[dict] = []
        self._sent = 0
        self._last_trickle = float("-inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background thread."""
        self._thread = threading.Thread(target=self._loop, name="demo-traffic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread after the prompt it is on, if any."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def start_scenario(self, name: str) -> None:
        """Queue a scenario's seeded sequence; KeyError if unknown, RuntimeError if one runs."""
        if name not in scenario_names(self.scenario_dir):
            raise KeyError(name)
        scenario = load_scenario(self.scenario_dir / f"{name}.yaml")
        pools = {
            p: load_pool(p, self.data_dir) for p in (scenario.normal_pool, scenario.shift_pool) if p
        }
        rows = build_sequence(scenario, pools)
        with self._lock:
            if self._scenario is not None:
                raise RuntimeError(f"scenario {self._scenario} is already running")
            self._scenario, self._rows, self._sent = name, rows, 0

    def status(self) -> dict:
        """Warm-up progress and the running scenario, for the dashboard."""
        stored = storage.total_requests()
        with self._lock:
            scenario = (
                {"name": self._scenario, "sent": self._sent, "total": len(self._rows)}
                if self._scenario
                else None
            )
        return {
            "warmup": {"stored": stored, "target": self.warmup_target}
            if stored < self.warmup_target
            else None,
            "scenario": scenario,
        }

    def _next(self) -> tuple[str, str] | None:
        """The next (prompt, label) to send now, or None if nothing is due yet."""
        with self._lock:
            if self._scenario is not None:
                return self._rows[self._sent]["prompt"], self._scenario
        due = time.monotonic() - self._last_trickle >= self.interval_s
        if due or storage.total_requests() < self.warmup_target:
            self._last_trickle = time.monotonic()
            return next(self._normal), TRICKLE_LABEL
        return None

    def _send(self, prompt: str, label: str) -> None:
        """Answer one prompt and store it as a recorded request."""
        start = time.perf_counter()
        result = self.engine.run(prompt)
        storage.save_request(
            prompt,
            result.reply,
            self.engine.model_name,
            result.hits,
            result.gen_ms,
            result.extract_ms,
            label,
        )
        with self._lock:
            if self._scenario == label:
                self._sent += 1
                if self._sent == len(self._rows):
                    self._scenario, self._rows = None, []
        if self.on_prompt is not None:
            self.on_prompt(label, (time.perf_counter() - start) * 1000, result)

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = None if self.busy() else self._next()
            if item is None:
                self._stop.wait(POLL_S)
                continue
            try:
                self._send(*item)
            except Exception:
                # Keep the demo alive through one bad prompt or a locked database; try later.
                logger.exception("demo prompt failed")
                self._stop.wait(self.interval_s)
