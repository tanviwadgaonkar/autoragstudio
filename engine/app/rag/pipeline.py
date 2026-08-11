"""The end-to-end RAG pipeline: chunk -> embed -> store -> retrieve -> answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunker import chunk_documents
from .embeddings import Embedder
from .llm import LLM
from .store import ChromaAgentStore, RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are a focused assistant. Answer using ONLY the provided context "
    "blocks. If the context does not contain the answer, say you do not know."
)


@dataclass
class Answer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class RagPipeline:
    def __init__(
        self,
        store: ChromaAgentStore,
        embedder: Embedder,
        llm: LLM,
        *,
        chunk_size: int = 120,
        overlap: int = 24,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._system_prompts: dict[str, str] = {}

    # -- ingestion ---------------------------------------------------------
    def ingest(
        self,
        agent_id: str,
        documents: list[dict],
        *,
        system_prompt: str | None = None,
        reset: bool = False,
    ) -> dict:
        """Chunk + embed + upsert ``documents`` into the agent's collection."""
        if reset:
            self.store.delete_agent(agent_id)
        if system_prompt:
            self._system_prompts[agent_id] = system_prompt

        chunks = chunk_documents(
            documents, chunk_size=self.chunk_size, overlap=self.overlap
        )
        embeddings = self.embedder.embed([c["text"] for c in chunks]) if chunks else []
        stored = self.store.add_chunks(agent_id, chunks, embeddings)
        return {
            "agent_id": agent_id,
            "documents": len(documents),
            "chunks_indexed": stored,
            "total_chunks": self.store.count(agent_id),
        }

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self, agent_id: str, question: str, top_k: int = 4
    ) -> list[RetrievedChunk]:
        [query_embedding] = self.embedder.embed([question])
        return self.store.query(agent_id, query_embedding, top_k=top_k)

    # -- generation --------------------------------------------------------
    def answer(self, agent_id: str, question: str, top_k: int = 4) -> Answer:
        sources = self.retrieve(agent_id, question, top_k=top_k)
        context = "\n\n".join(
            f"[source {i + 1} | {c.metadata.get('doc_title', 'unknown')}]\n{c.text}"
            for i, c in enumerate(sources)
        )
        system = self._system_prompts.get(agent_id, DEFAULT_SYSTEM_PROMPT)
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return Answer(answer=self.llm.chat(system, user), sources=sources)
