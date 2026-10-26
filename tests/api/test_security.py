"""S13: the pages work with a same-origin content policy and no dev CORS."""

import json
import re
import subprocess
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import pytest
from fastapi.testclient import TestClient

from tests.support.store import app
from phaseeval.launcher import Nonces, Session, launcher_router


@pytest.fixture
def client(tmp_path):
    routes = app.router.routes[:]
    tokens = {"owner": "admin", "owner-rater": "rater"}
    app.include_router(launcher_router(Session(tmp_path / "principals.json", tokens, ()), Nonces(tokens), "test", 18800))

    def broken():
        raise RuntimeError("test failure")

    app.add_api_route("/test-failure", broken)
    app.add_api_route("/api/attempts/test-failure", broken)
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.router.routes[:] = routes


@pytest.mark.parametrize("path", ["/workbench", "/scorecard", "/launcher", "/dashboard", "/api/health", "/missing", "/api/attempts/queue", "/test-failure", "/api/attempts/test-failure"])
def test_every_page_sends_csp_nosniff_and_frame_ancestors(client, path):
    response = client.get(path, follow_redirects=False)
    assert response.status_code in (200, 307, 401, 404, 500, 503)
    assert response.headers.get("Content-Security-Policy") == (
        "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    if path.startswith("/api/attempts"):
        assert response.headers["Cache-Control"] == "no-store"
        assert "Authorization" in response.headers["Vary"]


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_generated_documentation_is_disabled(client, path):
    assert client.get(path).status_code == 404


class PageAssets(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.urls = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        assert tag != "style"
        assert not any(key == "style" or key.startswith("on") for key in attrs)
        if tag == "script":
            assert attrs.get("src"), "CSP blocks inline scripts"
        if tag in ("script", "img", "link"):
            self.urls.append(attrs.get("src") or attrs["href"])


@pytest.mark.parametrize("path", ["/workbench", "/scorecard", "/launcher"])
def test_no_page_loads_a_third_party_origin(client, path):
    page = client.get(path)
    assert page.status_code == 200
    pending = [(path, url) for url in PageAssets(page.text).urls]
    seen = set()
    fonts = set()
    while pending:
        parent, url = pending.pop()
        resolved = urljoin(parent, url)
        assert not urlsplit(resolved).netloc, resolved
        if resolved in seen:
            continue
        seen.add(resolved)
        response = client.get(resolved)
        assert response.status_code == 200, resolved
        content_type = response.headers["Content-Type"]
        if "text/css" in content_type:
            pending.extend((resolved, value) for value in re.findall(r"url\([\"']?([^\s)\"']+)", response.text))
        elif "javascript" in content_type:
            assert not re.search(r"\b(?:style|on\w+)=[\"']", response.text), resolved
        elif "font/woff2" in content_type:
            assert response.content.startswith(b"wOF2")
            fonts.add(resolved)
    if path != "/launcher":
        assert len(fonts) >= 2, "both Plex families must be served locally"


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"])
def test_a_dev_origin_preflight_gets_no_cors_headers(client, origin):
    response = client.options("/api/attempts/queue", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert response.status_code == 405
    assert not any(name.startswith("access-control-") for name in response.headers)
    response = client.get("/api/health", headers={"Origin": origin})
    assert response.status_code == 200
    assert not any(name.startswith("access-control-") for name in response.headers)


@pytest.mark.parametrize("name", ["fonts/LICENSE.txt", "fonts/README.md", "fonts/missing.woff2", "fonts/%2e%2e/app.py", "%2e%2e/app.py"])
def test_the_asset_route_keeps_its_allowlist(client, name):
    assert client.get(f"/workbench/assets/{name}").status_code == 404


@pytest.mark.parametrize("name, media_type", [("presentation.css", "text/css"), ("report.css", "text/css"), ("report.js", "text/javascript")])
def test_the_report_assets_are_served(client, name, media_type):
    response = client.get(f"/workbench/assets/{name}")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith(media_type)


def test_the_scorecard_keeps_its_heatmap_colors_and_bar_widths(client):
    page = client.get("/scorecard")
    source = re.search(r'<script src="([^"]+)"', page.text)
    script = client.get(source[1]).text if source else re.search(r"<script>(.*?)</script>", page.text, re.S)[1]
    run = subprocess.run(["node", "-e", r'''
const fs = require("node:fs"), vm = require("node:vm");
const lead = {
  innerHTML: "", nodes: {},
  querySelectorAll(selector) {
    const key = selector === "[data-tint]" ? "tint" : "width";
    return this.nodes[key] ||= [...this.innerHTML.matchAll(new RegExp(`data-${key}="([^"]+)"`, "g"))]
      .map((m) => ({ dataset: { [key]: m[1] }, style: {} }));
  },
};
const ctx = vm.createContext({ URLSearchParams, location: { search: "" },
  sessionStorage: { getItem() {}, removeItem() {} },
  document: { getElementById(id) { return id === "lead" ? lead : {}; } },
});
const script = fs.readFileSync(0, "utf8");
vm.runInContext(script.slice(0, script.indexOf('document.addEventListener("click"')), ctx);
vm.runInContext(`VIEW = { steps: [{ step: "search", title: "Search" }], episodes: [], failure_codes: [],
  matrix: [{ source: "decided", model: "test", episodes: 4, cells: { search: {
    passed: 1, failed: 2, disputed: 1, no_answer: 0, unknown: 1, denominator: 4,
  } } }] }; drawLead();`, ctx);
process.stdout.write(JSON.stringify({
  colors: lead.querySelectorAll("[data-tint]").map((n) => n.style.backgroundColor),
  widths: lead.querySelectorAll("[data-width]").map((n) => n.style.width),
}));
'''], input=script, capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    shown = json.loads(run.stdout)
    assert shown["colors"] == ["rgba(34,197,94,0.145)"]
    assert shown["widths"] == ["25.0%", "25.0%", "25.0%", "0.0%", "25.0%"]
