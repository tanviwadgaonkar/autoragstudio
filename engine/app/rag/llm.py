"""LLM backends for answer generation.

- ``OllamaLLM`` — chats with a locally hosted model through Ollama's
  ``/api/chat`` endpoint (default ``llama3.2``). Non-streaming on purpose:
  the engine returns whole answers with their sources.
- ``EchoLLM`` — deterministic stand-in used by the test suite and offline
  mode: it echoes how much context it was given and the question it was
  asked, which lets tests assert the prompt-assembly path end to end
  without a model.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str) -> str: ...


class EchoLLM:
    name = "echo"

    def chat(self, system: str, user: str) -> str:
        context_blocks = user.count("[source ")
        question = user.rsplit("Question:", 1)[-1].strip()
        return (
            f"[echo-llm] Answering '{question}' using {context_blocks} "
            "retrieved context block(s)."
        )


class OllamaLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def chat(self, system: str, user: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
