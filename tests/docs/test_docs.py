import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Pages a reviewer reads first; docs/log.md is history and may name files that are gone.
PAGES = ["README.md", "docs/writeup.md", "docs/usage.md", "docs/deployment.md"]
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)|<img[^>]*\ssrc=\"([^\"]+)\"")


@pytest.mark.parametrize("page", PAGES)
def test_relative_links_and_images_exist(page: str) -> None:
    path = ROOT / page
    targets = [a or b for a, b in LINK.findall(path.read_text(encoding="utf-8"))]
    local = [t for t in targets if not re.match(r"[a-z]+:", t)]
    missing = [t for t in local if not (path.parent / t).exists()]
    assert not missing


@pytest.mark.parametrize("page", PAGES)
def test_no_mermaid_blocks(page: str) -> None:
    # Hugging Face does not render Mermaid, and a broken block shows as an error on GitHub.
    assert "```mermaid" not in (ROOT / page).read_text(encoding="utf-8")


def test_architecture_svg_is_valid_with_its_own_background() -> None:
    svg = ET.parse(ROOT / "docs" / "images" / "architecture.svg").getroot()
    rects = svg.findall("{http://www.w3.org/2000/svg}rect")
    assert rects[0].get("fill") == "#ffffff"
