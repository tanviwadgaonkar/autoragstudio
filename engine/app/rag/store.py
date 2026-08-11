"""Per-agent vector storage on ChromaDB.

Every agent gets its own Chroma collection (``agent--<id>``), so retrieval is
fully isolated: agent A can never surface agent B's documents. That isolation
is the core design decision of AutoRAG Studio and is covered directly by the
test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb


class AgentIndexNotFound(KeyError):
    """Raised when querying an agent that has no indexed documents."""


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def _collection_name(agent_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", agent_id.strip().lower()).strip("-")
    if not safe:
        raise ValueError("agent_id must contain at least one alphanumeric character")
    return f"agent--{safe}"[:63]


class ChromaAgentStore:
    """One Chroma client, one collection per agent, cosine space."""

    def __init__(self, persist_dir: str | None = None) -> None:
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

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
            name=_collection_name(agent_id),
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def delete_agent(self, agent_id: str) -> bool:
        try:
            self._client.delete_collection(_collection_name(agent_id))
            return True
        except Exception:
            return False

    # -- read --------------------------------------------------------------
    def count(self, agent_id: str) -> int:
        try:
            return self._client.get_collection(_collection_name(agent_id)).count()
        except Exception:
            return 0

    def query(
        self,
        agent_id: str,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        try:
            collection = self._client.get_collection(_collection_name(agent_id))
        except Exception as exc:
            raise AgentIndexNotFound(
                f"agent '{agent_id}' has no indexed documents"
            ) from exc

        n = max(1, min(top_k, max(collection.count(), 1)))
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            RetrievedChunk(text=d, score=1.0 - dist, metadata=dict(m or {}))
            for d, m, dist in zip(docs, metas, dists)
        ]
