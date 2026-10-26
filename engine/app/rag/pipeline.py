"""The end-to-end RAG pipeline: chunk -> embed -> store -> retrieve -> answer."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_documents
from .embeddings import Embedder
from .llm import LLM
from .store import AgentStore, RetrievedChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are a focused assistant. Answer using ONLY the provided context "
    "blocks. If the context does not contain the answer, say you do not know."
)


@dataclass
class Answer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


class PromptStore:
    """Per-agent system prompts, optionally persisted to a JSON file.

    The vector index survives a restart, so the agent's persona must too —
    otherwise a restarted engine silently answers with the default prompt.
    Writes go to a temp file and are renamed into place, so a crash mid-write
    never leaves a truncated file.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._prompts: dict[str, str] = {}
        if self._path and self._path.exists():
            self._prompts = json.loads(self._path.read_text(encoding="utf-8"))

    def get(self, agent_id: str) -> str | None:
        return self._prompts.get(agent_id)

    def set(self, agent_id: str, prompt: str) -> None:
        self._prompts[agent_id] = prompt
        self._flush()

    def delete(self, agent_id: str) -> None:
        if self._prompts.pop(agent_id, None) is not None:
            self._flush()

    def _flush(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._prompts, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)


class RagPipeline:
    def __init__(
        self,
        store: AgentStore,
        embedder: Embedder,
        llm: LLM,
        *,
        prompts: PromptStore | None = None,
        chunk_size: int = 120,
        overlap: int = 24,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.prompts = prompts or PromptStore()
        self.chunk_size = chunk_size
        self.overlap = overlap

    # -- ingestion ---------------------------------------------------------
    def ingest(
        self,
        agent_id: str,
        documents: list[dict],
        *,
        system_prompt: str | None = None,
        reset: bool = False,
    ) -> dict:
        """Chunk + embed + index ``documents`` into the agent's collection.

        A document whose title is already indexed is *replaced*, not merged:
        its old chunks are removed first, so an edited (shorter) document
        cannot leave stale chunks behind to be retrieved later.
        """
        chunks = chunk_documents(
            documents, chunk_size=self.chunk_size, overlap=self.overlap
        )
        # Embed before touching the index: if the embedder is down, the
        # previous version of every document is still intact.
        embeddings = self.embedder.embed([c["text"] for c in chunks]) if chunks else []

        if reset:
            self.store.delete_agent(agent_id)
        else:
            for doc in documents:
                self.store.delete_document(agent_id, doc.get("title", "doc"))
        if system_prompt:
            self.prompts.set(agent_id, system_prompt)

        stored = self.store.add_chunks(agent_id, chunks, embeddings)
        return {
            "agent_id": agent_id,
            "documents": len(documents),
            "chunks_indexed": stored,
            "total_chunks": self.store.count(agent_id),
        }

    def delete_agent(self, agent_id: str) -> bool:
        self.prompts.delete(agent_id)
        return self.store.delete_agent(agent_id)

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
        system = self.prompts.get(agent_id) or DEFAULT_SYSTEM_PROMPT
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return Answer(answer=self.llm.chat(system, user), sources=sources)
