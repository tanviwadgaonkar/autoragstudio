"""Environment configuration of the model backends."""

import pytest

from app import deps


@pytest.fixture(autouse=True)
def fresh_pipeline():
    deps.reset_pipeline()
    yield
    deps.reset_pipeline()


def test_ollama_timeouts_come_from_env(monkeypatch):
    monkeypatch.setenv("RAG_STORE", "memory")
    monkeypatch.setenv("RAG_EMBEDDER", "ollama")
    monkeypatch.setenv("RAG_LLM", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_TIMEOUT", "600")
    monkeypatch.setenv("OLLAMA_EMBED_TIMEOUT", "90")
    pipe = deps.get_pipeline()
    assert (pipe.llm.timeout, pipe.embedder.timeout) == (600.0, 90.0)


@pytest.mark.parametrize("bad", ["0", "-5", "soon"])
def test_invalid_timeout_fails_at_startup(monkeypatch, bad):
    monkeypatch.setenv("RAG_STORE", "memory")
    monkeypatch.setenv("RAG_LLM", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_TIMEOUT", bad)
    with pytest.raises(ValueError):
        deps.get_pipeline()
