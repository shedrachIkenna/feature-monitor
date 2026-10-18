"""The overview's try box, live feed and recorded replay, run in Node against the real scripts."""

import json

from backend.app import FRONTEND_DIR
from tests.frontend.node_js import evaluate as run_js
from tests.frontend.node_js import needs_node

pytestmark = needs_node


def evaluate(expression: str):
    return run_js(expression, "common.js", "live.js")


def test_rare_features_are_named_rare_and_common_ones_get_a_count() -> None:
    assert evaluate("rateText(0)").startswith("rare in normal traffic")
    assert evaluate("rateText(9.9)").startswith("rare in normal traffic")
    assert evaluate("rateText(312.4)") == "in 312 of every 1,000 normal prompts"


def test_feed_shows_only_unseen_rows_oldest_first() -> None:
    rows = json.dumps([{"id": 9}, {"id": 8}, {"id": 7}])
    assert evaluate(f"newRows({rows}, new Set([7])).map((r) => r.id)") == [8, 9]


def test_feed_tags_demo_traffic_and_staged_scenarios() -> None:
    assert evaluate('feedTag("demo_normal")') == "normal traffic"
    assert evaluate('feedTag("injection_ramp")') == "staged scenario: injection_ramp"
    assert evaluate("feedTag(null)") == "recorded traffic"


def test_recorded_caption_uses_the_committed_numbers() -> None:
    example = json.loads((FRONTEND_DIR / "recorded_example.json").read_text())
    caption = evaluate(f"recordedCaption({json.dumps(example)})")
    first = example["first_above"]
    assert f"first crossed 1 at request {first['sae_shift']}" in caption
    assert f"request {first['embedding_mean']}" in caption
    assert ("never crossed 1" in caption) == (first["sae_no_shift"] is None)
