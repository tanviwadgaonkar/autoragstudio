"""Engine configuration and shared singletons.

Backends are selected with environment variables so the same code runs in
several modes without edits:

- real:     RAG_EMBEDDER=ollama  RAG_LLM=ollama   (local Ollama server)
- offline:  RAG_EMBEDDER=hash    RAG_LLM=echo     (no model, still functional)
- tests:    same as offline, plus RAG_CHROMA=ephemeral (in-memory Chroma)

``RAG_STORE=memory`` swaps ChromaDB for the dependency-free in-process store
(same isolation semantics, nothing persisted).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .rag.embeddings import HashingEmbedder, OllamaEmbedder
from .rag.llm import EchoLLM, OllamaLLM
from .rag.pipeline import PromptStore, RagPipeline
from .rag.store import ChromaAgentStore, MemoryAgentStore


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    persist_dir = os.environ.get("RAG_CHROMA_DIR", ".chroma")

    prompts = PromptStore()
    if _env("RAG_STORE", "chroma") == "memory":
        store = MemoryAgentStore()
    elif _env("RAG_CHROMA", "persistent") == "ephemeral":
        store = ChromaAgentStore(persist_dir=None)
    else:
        store = ChromaAgentStore(persist_dir=persist_dir)
        prompts = PromptStore(Path(persist_dir) / "system_prompts.json")

    if _env("RAG_EMBEDDER", "ollama") == "hash":
        embedder = HashingEmbedder()
    else:
        embedder = OllamaEmbedder(
            base_url=ollama_url,
            model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )

    if _env("RAG_LLM", "ollama") == "echo":
        llm = EchoLLM()
    else:
        llm = OllamaLLM(
            base_url=ollama_url,
            model=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2"),
        )

    return RagPipeline(store=store, embedder=embedder, llm=llm, prompts=prompts)


def reset_pipeline() -> None:
    """Drop the cached pipeline (used by tests to get a fresh store)."""
    get_pipeline.cache_clear()
