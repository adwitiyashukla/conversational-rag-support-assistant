import json

import pytest
from fastapi.testclient import TestClient

from convrag.api import create_app


@pytest.fixture
def client(runtime):
    app = create_app(runtime.cfg, runtime)
    with TestClient(app) as test_client:
        yield test_client


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        kind, data = "", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((kind, json.loads(data)))
    return events


def test_health_ready_and_collections(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["embedding_model"].startswith("hash")
    assert ready["chunks"] > 0
    collections = client.get("/collections").json()["collections"]
    assert collections[0]["collection"] == "dmv" and collections[0]["documents"] == 3


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "convrag" in response.text and "text/html" in response.headers["content-type"]


def test_search_endpoint_and_validation(client):
    response = client.post("/search", json={"query": "renew license", "collection": "dmv"})
    assert response.status_code == 200
    body = response.json()
    assert body["hits"][0]["path"] == "dmv/licenses.md"
    assert {s["name"] for s in body["trace"]["spans"]} >= {"sparse", "dense", "fuse"}
    assert client.post("/search", json={"query": "", "collection": "dmv"}).status_code == 422
    bad_mode = client.post("/search", json={"query": "x", "collection": "dmv", "mode": "magic"})
    assert bad_mode.status_code == 400


def test_chat_json_and_session_lookup(client):
    first = client.post("/chat", json={"message": "my insurance ended", "collection": "dmv"})
    assert first.status_code == 200
    body = first.json()
    assert body["grounded"] and body["answer"].endswith("[1]")
    session_id = body["session_id"]
    second = client.post(
        "/chat",
        json={"message": "how long do I have?", "collection": "dmv", "session_id": session_id},
    )
    assert second.json()["session_id"] == session_id
    session = client.get(f"/sessions/{session_id}").json()
    assert [m["role"] for m in session["messages"]] == ["user", "assistant", "user", "assistant"]
    assert session["messages"][1]["payload"]["sources"]
    assert client.get("/sessions/nope").status_code == 404


def test_chat_stream_sends_sse_events(client):
    with client.stream(
        "POST",
        "/chat",
        json={"message": "dispute a parking ticket", "collection": "dmv", "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    events = parse_sse(body)
    kinds = [k for k, _ in events]
    assert kinds[:3] == ["start", "queries", "sources"]
    assert kinds[-1] == "done"
    done = events[-1][1]
    assert done["sources"][0]["path"] == "parking.html"
    assert done["trace"]["total_ms"] >= 0


def test_ingest_endpoint(client, corpus):
    (corpus / "boats.txt").write_text("Boat registration\n\nRegister your boat yearly.", "utf-8")
    response = client.post("/ingest", json={"folder": str(corpus), "collection": "dmv"})
    assert response.status_code == 200
    assert response.json()["added"] == 1 and response.json()["unchanged"] == 3
    missing = client.post("/ingest", json={"folder": str(corpus / "nope"), "collection": "dmv"})
    assert missing.status_code == 400


def test_traces_and_metrics(client):
    client.post("/search", json={"query": "renew license", "collection": "dmv"})
    client.post("/chat", json={"message": "renew license", "collection": "dmv"})
    traces = client.get("/traces?limit=5").json()["traces"]
    assert traces and traces[0]["kind"] == "chat"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'convrag_requests_total{kind="search"} 1' in metrics.text
    assert 'convrag_requests_total{kind="chat"} 1' in metrics.text


def test_api_key_protects_endpoints(runtime, monkeypatch):
    monkeypatch.setenv(runtime.cfg.server.api_key_env, "secret")
    with TestClient(create_app(runtime.cfg, runtime)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/collections").status_code == 401
        ok = client.get("/collections", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        assert client.get("/collections", headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get("/collections", headers={"X-API-Key": "nope"}).status_code == 401
