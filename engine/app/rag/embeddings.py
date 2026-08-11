"""Embedding backends.

Two implementations behind one tiny interface (``embed(texts) -> list[vec]``):

- ``OllamaEmbedder`` — calls a locally hosted Ollama server
  (``/api/embeddings``), default model ``nomic-embed-text``. This is what the
  platform uses in real runs: no data leaves the machine.
- ``HashingEmbedder`` — a dependency-free, deterministic bag-of-words hashing
  embedder. Used by the test suite and as an offline fallback so the whole
  pipeline (chunk -> embed -> store -> retrieve) runs without any model
  downloads. It is intentionally simple, but word overlap still produces
  meaningful cosine similarity, which is all the tests need.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic hashed bag-of-words vectors, L2-normalised."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class OllamaEmbedder:
    """Embeddings from a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dim = -1  # discovered from the first response

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                resp.raise_for_status()
                vector = resp.json()["embedding"]
                if self.dim == -1:
                    self.dim = len(vector)
                out.append(vector)
        return out


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)
