import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from phaseeval.api.app import SECURITY_HEADERS, create_app
from phaseeval.workbench.cli import Settings
from tests.support.store import ADD, OWNER, _headers


LIMIT = 512


@pytest.mark.parametrize("framing", ["content-length", "chunked", "underreported"])
@pytest.mark.parametrize("extra", [0, 1])
def test_action_body_limit_precedes_service_writes(bench, monkeypatch, framing, extra):
    app = create_app(Settings(max_body_bytes=LIMIT), store=bench.store,
                     authenticator=bench.authenticator)
    body = json.dumps({"steps": [{"idempotency_key": "bounded", "action": ADD}]}).encode()
    body += b" " * (LIMIT + extra - len(body))
    before = {p.relative_to(bench.store.root): p.read_bytes()
              for p in bench.store.root.rglob("*") if p.is_file()}
    calls = []
    gesture = bench.store.gesture

    def tracked(*args):
        calls.append(args)
        return gesture(*args)

    monkeypatch.setattr(bench.store, "gesture", tracked)
    sent = []

    async def chunks():
        parts = (body[:LIMIT], body[LIMIT:], b" ") if extra else (body[:LIMIT // 2], body[LIMIT // 2:])
        for chunk in parts:
            sent.append(chunk)
            yield chunk

    async def post():
        headers = _headers(OWNER) | {"Content-Type": "application/json"}
        if framing != "chunked":
            headers["Content-Length"] = str(len(body) if framing == "content-length" else 1)
        content = body if framing == "content-length" else chunks()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://testserver") as client:
            return await client.post(f"/api/attempts/{bench.root}/actions",
                                     headers=headers, content=content)

    response = asyncio.run(post())
    assert response.status_code == (413 if extra else 201), response.text
    if extra:
        assert response.json() == {"error": "too large"}
        assert calls == []
        assert before == {p.relative_to(bench.store.root): p.read_bytes()
                          for p in bench.store.root.rglob("*") if p.is_file()}
        if framing != "content-length":
            assert sent == [body[:LIMIT], body[LIMIT:]]
        assert all(response.headers[name] == value for name, value in SECURITY_HEADERS.items())
        assert response.headers["Cache-Control"] == "no-store"
        assert "Authorization" in response.headers["Vary"]
    else:
        assert len(calls) == 1
    with TestClient(app) as client:
        assert client.get(f"/api/attempts/roots/{bench.root}/tree",
                          headers=_headers(OWNER)).status_code == 200


def test_oversized_content_length_is_refused_without_reading(bench):
    app = create_app(Settings(max_body_bytes=LIMIT), store=bench.store,
                     authenticator=bench.authenticator)
    sent = []

    async def chunks():
        sent.append(True)
        yield b"{}"

    async def post():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://testserver") as client:
            return await client.post(f"/api/attempts/{bench.root}/actions", content=chunks(),
                                     headers=_headers(OWNER) | {"Content-Length": str(LIMIT + 1)})

    response = asyncio.run(post())
    assert response.status_code == 413, response.text
    assert response.json() == {"error": "too large"}
    assert sent == []
