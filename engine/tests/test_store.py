"""Storage: per-agent ChromaDB collections."""

import pytest

from app.rag.chunker import chunk_text
from app.rag.embeddings import HashingEmbedder
from app.rag.store import AgentIndexNotFound, ChromaAgentStore


EMB = HashingEmbedder()


def _index(store: ChromaAgentStore, agent_id: str, text: str, title: str = "doc"):
    chunks = chunk_text(text, doc_title=title, chunk_size=30, overlap=5)
    store.add_chunks(agent_id, chunks, EMB.embed([c["text"] for c in chunks]))
    return chunks


def test_add_chunks_and_count(store, agent_id):
    chunks = _index(store, agent_id, " ".join(f"w{i}" for i in range(90)))
    assert store.count(agent_id) == len(chunks) > 1


def test_per_agent_isolation(store):
    _index(store, "agent-cats", "cats purr and kittens chase yarn balls", "cats")
    _index(store, "agent-fin", "compound interest grows dividend income", "fin")

    [q] = EMB.embed(["cats purr"])
    hits = store.query("agent-fin", q, top_k=5)
    titles = {h.metadata["doc_title"] for h in hits}
    assert titles == {"fin"}  # the cats document can never leak in


def test_delete_agent_index(store, agent_id):
    _index(store, agent_id, "some words to index for deletion")
    assert store.count(agent_id) > 0
    assert store.delete_agent(agent_id) is True
    assert store.count(agent_id) == 0


def test_query_returns_top_k(store, agent_id):
    _index(store, agent_id, " ".join(f"token{i}" for i in range(200)))
    [q] = EMB.embed(["token1 token2"])
    hits = store.query(agent_id, q, top_k=3)
    assert len(hits) == 3


def test_query_unknown_agent_raises(store):
    [q] = EMB.embed(["anything"])
    with pytest.raises(AgentIndexNotFound):
        store.query("never-ingested", q)


def test_long_agent_ids_do_not_share_a_collection(store):
    # v1 truncated names to 63 chars, so these two agents shared one index.
    base = "customer-support-knowledge-base-for-enterprise-clients-region-"
    _index(store, base + "a", "cats purr and kittens chase yarn balls", "cats")
    _index(store, base + "b", "compound interest grows dividend income", "fin")
    [q] = EMB.embed(["cats purr"])
    assert {h.metadata["doc_title"] for h in store.query(base + "b", q, top_k=5)} == {"fin"}


def test_delete_document_removes_only_that_document(store, agent_id):
    _index(store, agent_id, "cats purr and kittens chase yarn balls", "cats")
    _index(store, agent_id, "compound interest grows dividend income", "fin")
    assert store.delete_document(agent_id, "cats") >= 1
    [q] = EMB.embed(["cats purr"])
    assert {h.metadata["doc_title"] for h in store.query(agent_id, q, top_k=5)} == {"fin"}
