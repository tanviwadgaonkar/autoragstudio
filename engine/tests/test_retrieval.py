"""Retrieval: end-to-end pipeline behaviour."""

import pytest

from app.rag.embeddings import HashingEmbedder
from app.rag.llm import EchoLLM
from app.rag.pipeline import PromptStore, RagPipeline

from .conftest import CAT_DOC, FINANCE_DOC


def test_ingest_reports_counts(pipeline, agent_id):
    result = pipeline.ingest(agent_id, [CAT_DOC, FINANCE_DOC])
    assert result["documents"] == 2
    assert result["chunks_indexed"] >= 2
    assert result["total_chunks"] == result["chunks_indexed"]


def test_relevant_chunk_ranks_first(pipeline, agent_id):
    pipeline.ingest(agent_id, [CAT_DOC, FINANCE_DOC])
    hits = pipeline.retrieve(agent_id, "do cats purr and play with yarn?", top_k=2)
    assert hits[0].metadata["doc_title"] == "Cats"
    assert hits[0].score >= hits[-1].score


def test_sources_carry_document_metadata(pipeline, agent_id):
    pipeline.ingest(agent_id, [CAT_DOC])
    hits = pipeline.retrieve(agent_id, "kittens", top_k=1)
    meta = hits[0].metadata
    assert meta["doc_title"] == "Cats"
    assert meta["chunk_index"] == 0
    assert "word_count" in meta


def test_reingesting_a_shorter_document_drops_stale_chunks(pipeline, agent_id):
    long_doc = {"title": "Policy", "text": " ".join(f"oldword{i}" for i in range(400))}
    first = pipeline.ingest(agent_id, [long_doc])
    assert first["total_chunks"] > 1

    pipeline.ingest(agent_id, [{"title": "Policy", "text": "the new short policy"}])
    assert pipeline.store.count(agent_id) == 1
    hits = pipeline.retrieve(agent_id, "oldword390", top_k=5)
    assert all("oldword" not in h.text for h in hits)


def test_reingest_keeps_other_documents(pipeline, agent_id):
    pipeline.ingest(agent_id, [CAT_DOC, FINANCE_DOC])
    pipeline.ingest(agent_id, [{"title": "Cats", "text": "cats sleep all day"}])
    titles = {h.metadata["doc_title"] for h in pipeline.retrieve(agent_id, "interest", top_k=5)}
    assert titles == {"Cats", "Finance"}


def test_failed_embedding_leaves_previous_version_intact(store, agent_id):
    class FlakyEmbedder(HashingEmbedder):
        fail = False

        def embed(self, texts):
            if self.fail:
                raise ConnectionError("ollama is down")
            return super().embed(texts)

    embedder = FlakyEmbedder()
    pipe = RagPipeline(store=store, embedder=embedder, llm=EchoLLM())
    pipe.ingest(agent_id, [CAT_DOC])
    before = store.count(agent_id)

    embedder.fail = True
    with pytest.raises(ConnectionError):
        pipe.ingest(agent_id, [{"title": "Cats", "text": "replacement text"}])
    assert store.count(agent_id) == before


def test_system_prompt_survives_restart(tmp_path, store, agent_id):
    path = tmp_path / "system_prompts.json"
    pipe = RagPipeline(store=store, embedder=HashingEmbedder(), llm=EchoLLM(), prompts=PromptStore(path))
    pipe.ingest(agent_id, [CAT_DOC], system_prompt="You are a cat expert.")

    restarted = RagPipeline(store=store, embedder=HashingEmbedder(), llm=EchoLLM(), prompts=PromptStore(path))
    assert restarted.prompts.get(agent_id) == "You are a cat expert."

    restarted.delete_agent(agent_id)
    assert PromptStore(path).get(agent_id) is None
