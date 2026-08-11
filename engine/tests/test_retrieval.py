"""Retrieval: end-to-end pipeline behaviour (3 tests)."""

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
