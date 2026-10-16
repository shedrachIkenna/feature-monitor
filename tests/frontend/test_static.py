from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.app import FRONTEND_DIR, create_app
from tests.backend.test_app import FakeEngine

PAGES = ["index.html", "features.html", "playground.html"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(storage.DB_PATH_ENV, str(tmp_path / "test.db"))
    with TestClient(create_app(FakeEngine)) as test_client:
        yield test_client


class Sections(HTMLParser):
    """Collect each <section>'s h2 text and whether it holds a <canvas>."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[dict] = []
        self.scripts: list[str] = []
        self._in_h2 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            self.sections.append({"title": "", "canvas": False})
        elif tag == "h2":
            self._in_h2 = True
        elif tag == "canvas" and self.sections:
            self.sections[-1]["canvas"] = True
        elif tag == "script":
            self.scripts.append(dict(attrs).get("src") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._in_h2 = False

    def handle_data(self, data: str) -> None:
        if self._in_h2 and self.sections:
            self.sections[-1]["title"] += data


def test_root_serves_the_overview(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Drift score over time" in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/features.html", "text/html"),
        ("/style.css", "text/css"),
        ("/common.js", "text/javascript"),
        ("/overview.js", "text/javascript"),
        ("/features.js", "text/javascript"),
        ("/playground.html", "text/html"),
        ("/playground.js", "text/javascript"),
    ],
)
def test_frontend_files_are_served(client, path, content_type) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)


def test_api_routes_still_win_over_static_files(client) -> None:
    assert client.get("/health").json()["model"] == "fake-gpt2"
    assert client.get("/alerts").json() == []
    assert client.get("/no-such-page").status_code == 404


CHART_PAGES = ["index.html", "features.html"]
CHARTJS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.1/chart.umd.min.js"


@pytest.mark.parametrize("page", CHART_PAGES)
def test_every_chart_has_a_title(page) -> None:
    parser = Sections()
    parser.feed((FRONTEND_DIR / page).read_text())
    charts = [s for s in parser.sections if s["canvas"]]
    assert charts
    assert all(s["title"].strip() for s in charts)


@pytest.mark.parametrize("page", PAGES)
def test_pages_load_only_local_scripts_and_chartjs_from_cdnjs(page) -> None:
    parser = Sections()
    parser.feed((FRONTEND_DIR / page).read_text())
    remote = [s for s in parser.scripts if "://" in s]
    assert remote == ([CHARTJS] if page in CHART_PAGES else [])
    assert all((FRONTEND_DIR / s).is_file() for s in parser.scripts if s not in remote)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_links_to_all_three_pages(page) -> None:
    html = (FRONTEND_DIR / page).read_text()
    for target in ['href="./"', 'href="features.html"', 'href="playground.html"']:
        assert target in html


def test_playground_sends_unrecorded_chat_requests() -> None:
    script = (FRONTEND_DIR / "playground.js").read_text()
    assert "record: false" in script


def test_playground_explains_a_full_queue() -> None:
    script = (FRONTEND_DIR / "playground.js").read_text()
    assert "error.status === 429" in script


def test_how_to_read_panel_has_at_most_four_lines() -> None:
    html = (FRONTEND_DIR / "index.html").read_text()
    panel = html.split('class="card howto"')[1].split("</aside>")[0]
    assert "How to read this" in panel
    assert 1 <= panel.count("<li>") <= 4


@pytest.mark.parametrize("kind", ["sae", "embedding_mean", "embedding"])
def test_every_detector_kind_has_a_name_in_the_frontend(kind) -> None:
    # "embedding" is the old cluster detector, still found in saved databases.
    script = (FRONTEND_DIR / "common.js").read_text()
    assert f"\n  {kind}: {{" in script


@pytest.mark.parametrize("kind", ["sae", "embedding_mean"])
def test_overview_explains_each_live_detector(kind) -> None:
    assert f'id="help-{kind}"' in (FRONTEND_DIR / "index.html").read_text()
