"""The banner's refresh rule and text, run in Node against the real common.js."""

import json
import shutil
import subprocess

import pytest

from backend.app import FRONTEND_DIR

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

# Loads common.js with just enough browser stand-ins for its top level to run (the first
# banner update fails quietly without a server), then evaluates `expression` in its scope.
RUNNER = """
const vm = require("vm");
const fs = require("fs");
const context = {
  fetch: () => Promise.reject(new Error("no server")),
  setTimeout: () => 0,
  document: { getElementById: () => null },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
console.log(JSON.stringify(vm.runInContext(process.argv[2], context)));
"""


def evaluate(expression: str):
    result = subprocess.run(
        ["node", "-e", RUNNER, str(FRONTEND_DIR / "common.js"), expression],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


WARMING = {"enabled": True, "warmup": {"stored": 212, "target": 500}, "scenario": None}
RUNNING = {
    "enabled": True,
    "warmup": None,
    "scenario": {"name": "medical_ramp", "sent": 820, "total": 2000},
}
STEADY = {"enabled": True, "warmup": None, "scenario": None}


@pytest.mark.parametrize(
    ("demo", "expected"),
    [
        (WARMING, "DEMO_LIVE_MS"),
        (RUNNING, "DEMO_LIVE_MS"),
        (STEADY, "REFRESH_MS"),
        (None, "REFRESH_MS"),
    ],
)
def test_banner_checks_often_only_while_a_count_is_moving(demo, expected) -> None:
    assert evaluate(f"noticeDelay({json.dumps(demo)}) === {expected}")
    assert evaluate("DEMO_LIVE_MS < REFRESH_MS")


def test_banner_shows_the_moving_counts() -> None:
    assert "212 / 500" in evaluate(f"demoNotice({json.dumps(WARMING)})")
    assert "820 / 2000" in evaluate(f"demoNotice({json.dumps(RUNNING)})")
    assert "staged" in evaluate(f"demoNotice({json.dumps(RUNNING)})")
    assert evaluate(f"demoNotice({json.dumps({**STEADY, 'enabled': False})})") == ""
