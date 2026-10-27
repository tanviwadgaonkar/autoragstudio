"""Ingestion: embedding behaviour."""

from app.rag.embeddings import HashingEmbedder, cosine


def test_embeddings_have_fixed_dimension():
    emb = HashingEmbedder(dim=256)
    vectors = emb.embed(["hello world", "another text", ""])
    assert all(len(v) == 256 for v in vectors)


def test_embeddings_are_deterministic():
    emb = HashingEmbedder()
    [a] = emb.embed(["the same sentence"])
    [b] = emb.embed(["the same sentence"])
    assert a == b


def test_embeddings_separate_topics():
    emb = HashingEmbedder()
    cat, cat2, finance = emb.embed(
        [
            "cats purr and kittens chase yarn",
            "kittens and cats purr near the yarn basket",
            "compound interest and dividend income",
        ]
    )
    assert cosine(cat, cat2) > cosine(cat, finance)
