"""Ingestion: chunking behaviour."""

from app.rag.chunker import chunk_text


LONG_TEXT = " ".join(f"word{i}" for i in range(300))


def test_chunks_respect_max_size():
    chunks = chunk_text(LONG_TEXT, chunk_size=100, overlap=20)
    assert all(len(c["text"].split()) <= 100 for c in chunks)
    assert len(chunks) > 1


def test_consecutive_chunks_overlap():
    chunks = chunk_text(LONG_TEXT, chunk_size=100, overlap=20)
    first_tail = chunks[0]["text"].split()[-20:]
    second_head = chunks[1]["text"].split()[:20]
    assert first_tail == second_head


def test_short_text_yields_single_chunk():
    chunks = chunk_text("only a few words here", chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "only a few words here"


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\t  ", chunk_size=100, overlap=20) == []


def test_chunk_ids_and_metadata_are_deterministic():
    a = chunk_text(LONG_TEXT, doc_title="My Doc!", chunk_size=100, overlap=20)
    b = chunk_text(LONG_TEXT, doc_title="My Doc!", chunk_size=100, overlap=20)
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert a[0]["id"].startswith("my-doc-") and a[0]["id"].endswith("::chunk-0")
    assert a[0]["metadata"]["doc_title"] == "My Doc!"
    assert a[1]["metadata"]["chunk_index"] == 1


def test_similar_titles_get_distinct_chunk_ids():
    # "Q&A" and "Q A" slugify identically; their chunks must not overwrite each other.
    a = chunk_text("alpha beta", doc_title="Q&A")
    b = chunk_text("gamma delta", doc_title="Q A")
    assert a[0]["id"] != b[0]["id"]
