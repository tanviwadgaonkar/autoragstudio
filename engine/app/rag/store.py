"""Per-agent vector storage.

Every agent gets its own collection (``agent--<id>``), so retrieval is fully
isolated: agent A can never surface agent B's documents. That isolation is
the core design decision of AutoRAG Studio and is covered directly by the
test suite.

Two backends share one interface:

- ``ChromaAgentStore`` — ChromaDB, persistent or in-memory (the real store).
- ``MemoryAgentStore`` — a dependency-free in-process store with the same
  semantics, for offline runs and for platforms where Chroma's native
  bindings are unavailable.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol


class AgentIndexNotFound(KeyError):
    """Raised when querying an agent that has no indexed documents."""


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class AgentStore(Protocol):
    def add_chunks(self, agent_id: str, chunks: list[dict], embeddings: list[list[float]]) -> int: ...
    def delete_document(self, agent_id: str, doc_title: str) -> int: ...
    def delete_agent(self, agent_id: str) -> bool: ...
    def count(self, agent_id: str) -> int: ...
    def query(self, agent_id: str, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]: ...


_LOSSLESS = re.compile(r"[a-z0-9][a-z0-9_-]*[a-z0-9]|[a-z0-9]")
_MAX_NAME = 63


def collection_name(agent_id: str) -> str:
    """Map an agent id to a Chroma collection name, injectively.

    Ids that are already lowercase slugs and fit keep the readable
    ``agent--<id>`` form (unchanged from v1, so existing indexes still load).
    Anything else — too long, uppercase, punctuation — gets a truncated
    readable prefix plus a hash of the *original* id. The hashed form contains
    a ``.``, which a lossless name never can, so the two forms cannot collide;
    two hashed ids could only share a collection on a 64-bit hash collision
    (about 1 in 10^19 per pair).
    """
    readable = f"agent--{agent_id}"
    if _LOSSLESS.fullmatch(agent_id) and len(readable) <= _MAX_NAME:
        return readable

    safe = re.sub(r"[^a-z0-9_-]+", "-", agent_id.strip().lower()).strip("-_")
    if not safe:
        raise ValueError("agent_id must contain at least one alphanumeric character")
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()[:16]
    prefix = f"agent--{safe}"[: _MAX_NAME - len(digest) - 1].rstrip("-_")
    return f"{prefix}.{digest}"


class ChromaAgentStore:
    """One Chroma client, one collection per agent, cosine space."""

    def __init__(self, persist_dir: str | None = None) -> None:
        import chromadb  # imported lazily so the memory backend needs no native deps

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

    def _get(self, agent_id: str):
        try:
            return self._client.get_collection(collection_name(agent_id))
        except Exception:  # Chroma raises different types across versions
            return None

    # -- write -------------------------------------------------------------
    def add_chunks(
        self,
        agent_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0
        collection = self._client.get_or_create_collection(
            name=collection_name(agent_id),
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def delete_document(self, agent_id: str, doc_title: str) -> int:
        collection = self._get(agent_id)
        if collection is None:
            return 0
        ids = collection.get(where={"doc_title": doc_title}, include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def delete_agent(self, agent_id: str) -> bool:
        if self._get(agent_id) is None:
            return False
        self._client.delete_collection(collection_name(agent_id))
        return True

    # -- read --------------------------------------------------------------
    def count(self, agent_id: str) -> int:
        collection = self._get(agent_id)
        return collection.count() if collection is not None else 0

    def query(
        self,
        agent_id: str,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        collection = self._get(agent_id)
        if collection is None or collection.count() == 0:
            raise AgentIndexNotFound(f"agent '{agent_id}' has no indexed documents")

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            RetrievedChunk(text=d, score=1.0 - dist, metadata=dict(m or {}))
            for d, m, dist in zip(docs, metas, dists)
        ]


class MemoryAgentStore:
    """In-process store with the same semantics as ``ChromaAgentStore``.

    Exact (brute-force) cosine search — fine for the document counts a single
    agent holds locally, and it makes retrieval results easy to reason about.
    """

    def __init__(self) -> None:
        # collection name -> chunk id -> (text, metadata, unit vector)
        self._collections: dict[str, dict[str, tuple[str, dict, list[float]]]] = {}

    def add_chunks(self, agent_id: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0
        rows = self._collections.setdefault(collection_name(agent_id), {})
        for chunk, vector in zip(chunks, embeddings):
            rows[chunk["id"]] = (chunk["text"], dict(chunk["metadata"]), _unit(vector))
        return len(chunks)

    def delete_document(self, agent_id: str, doc_title: str) -> int:
        rows = self._collections.get(collection_name(agent_id), {})
        doomed = [cid for cid, (_, meta, _) in rows.items() if meta.get("doc_title") == doc_title]
        for cid in doomed:
            del rows[cid]
        return len(doomed)

    def delete_agent(self, agent_id: str) -> bool:
        return self._collections.pop(collection_name(agent_id), None) is not None

    def count(self, agent_id: str) -> int:
        return len(self._collections.get(collection_name(agent_id), {}))

    def query(self, agent_id: str, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        rows = self._collections.get(collection_name(agent_id))
        if not rows:
            raise AgentIndexNotFound(f"agent '{agent_id}' has no indexed documents")
        q = _unit(query_embedding)
        scored = [
            (sum(a * b for a, b in zip(q, vec)), text, meta)
            for text, meta, vec in rows.values()
        ]
        scored.sort(key=lambda row: row[0], reverse=True)
        return [RetrievedChunk(text=t, score=s, metadata=dict(m)) for s, t, m in scored[:top_k]]


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else list(vector)
