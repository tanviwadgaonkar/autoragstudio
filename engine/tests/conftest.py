"""Shared fixtures. Tests run fully offline: hashing embedder, echo LLM,
ephemeral (in-memory) Chroma. Each test gets a fresh pipeline and unique
agent ids, so nothing leaks between tests.

``RAG_STORE=memory pytest`` runs the same suite against the dependency-free
store; CI runs it against both backends."""

from __future__ import annotations

import os
import uuid

import pytest

os.environ["RAG_EMBEDDER"] = "hash"
os.environ["RAG_LLM"] = "echo"
os.environ["RAG_CHROMA"] = "ephemeral"
os.environ.setdefault("RAG_STORE", "chroma")

from fastapi.testclient import TestClient  # noqa: E402

from app import deps  # noqa: E402
from app.main import app  # noqa: E402
from app.rag.embeddings import HashingEmbedder  # noqa: E402
from app.rag.llm import EchoLLM  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.rag.store import AgentStore, ChromaAgentStore, MemoryAgentStore  # noqa: E402


@pytest.fixture()
def store() -> AgentStore:
    if os.environ["RAG_STORE"] == "memory":
        return MemoryAgentStore()
    return ChromaAgentStore(persist_dir=None)


@pytest.fixture()
def pipeline(store: ChromaAgentStore) -> RagPipeline:
    return RagPipeline(store=store, embedder=HashingEmbedder(), llm=EchoLLM())


@pytest.fixture()
def client() -> TestClient:
    deps.reset_pipeline()
    return TestClient(app)


@pytest.fixture()
def agent_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


CAT_DOC = {
    "title": "Cats",
    "text": (
        "Cats are small carnivorous mammals kept as pets. A cat purrs when "
        "content and grooms its fur daily. Kittens play with yarn and chase "
        "laser pointers around the living room."
    ),
}

FINANCE_DOC = {
    "title": "Finance",
    "text": (
        "Compound interest grows an investment over time. A diversified "
        "portfolio balances equity and debt. Inflation erodes purchasing "
        "power while dividends provide income."
    ),
}
