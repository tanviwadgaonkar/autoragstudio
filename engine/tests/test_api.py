"""API behaviour: the FastAPI engine surface (6 tests)."""

from .conftest import CAT_DOC, FINANCE_DOC


def _ingest(client, agent_id, docs):
    return client.post(
        f"/agents/{agent_id}/ingest", json={"documents": docs}
    )


def test_health_reports_backends(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["embedder"] == "HashingEmbedder"
    assert body["llm"] == "echo"


def test_ingest_endpoint_indexes_documents(client, agent_id):
    resp = _ingest(client, agent_id, [CAT_DOC])
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_id"] == agent_id
    assert body["chunks_indexed"] >= 1

    stats = client.get(f"/agents/{agent_id}/stats").json()
    assert stats["total_chunks"] == body["total_chunks"]


def test_chat_returns_answer_with_sources(client, agent_id):
    _ingest(client, agent_id, [CAT_DOC])
    resp = client.post(
        f"/agents/{agent_id}/chat", json={"question": "what do kittens chase?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "what do kittens chase?" in body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["metadata"]["doc_title"] == "Cats"


def test_query_unknown_agent_returns_404(client):
    resp = client.post(
        "/agents/ghost-agent/query", json={"question": "anything at all"}
    )
    assert resp.status_code == 404


def test_empty_question_fails_validation(client, agent_id):
    _ingest(client, agent_id, [CAT_DOC])
    resp = client.post(f"/agents/{agent_id}/chat", json={"question": ""})
    assert resp.status_code == 422


def test_agents_are_isolated_through_the_api(client):
    _ingest(client, "api-cats", [CAT_DOC])
    _ingest(client, "api-fin", [FINANCE_DOC])

    resp = client.post(
        "/agents/api-fin/query", json={"question": "cats purr kittens yarn"}
    )
    titles = {s["metadata"]["doc_title"] for s in resp.json()["sources"]}
    assert titles == {"Finance"}
