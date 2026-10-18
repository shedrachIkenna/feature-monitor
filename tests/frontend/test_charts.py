"""Chart end labels never overlap, run in Node against the real common.js."""

import json

from tests.frontend.node_js import evaluate as run_js
from tests.frontend.node_js import needs_node

pytestmark = needs_node


def spread(ys: list[float]) -> list[float]:
    labels = json.dumps([{"text": str(i), "y": y} for i, y in enumerate(ys)])
    return run_js(f"spreadLabels({labels}, LABEL_GAP).map((l) => l.y)", "common.js")


def test_close_labels_are_pushed_apart_by_the_gap() -> None:
    gap = run_js("LABEL_GAP", "common.js")
    # "alarm level" and "SAE features" ended 2 px apart on the live chart.
    assert spread([100, 102]) == [100, 100 + gap]
    assert spread([100, 101, 103]) == [100, 100 + gap, 100 + 2 * gap]


def test_labels_far_apart_stay_where_they_are() -> None:
    assert spread([50, 200]) == [50, 200]
    assert spread([200, 50]) == [50, 200]
