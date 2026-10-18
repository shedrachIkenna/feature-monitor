"""The banner's refresh rule and text, run in Node against the real common.js."""

import json

import pytest

from tests.frontend.node_js import evaluate as run_js
from tests.frontend.node_js import needs_node

pytestmark = needs_node


def evaluate(expression: str):
    return run_js(expression, "common.js")


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
