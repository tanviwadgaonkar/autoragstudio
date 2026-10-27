"""Word-window chunking with overlap — written from scratch, no framework.

A chunk is a dict: {"id": str, "text": str, "metadata": {...}}.
Chunk ids are deterministic (doc title + running index). The id carries a
short hash of the exact title, so titles that slugify alike ("Q&A" / "Q A")
never overwrite each other's chunks.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


_WS = re.compile(r"\s+")


def _words(text: str) -> list[str]:
    return [w for w in _WS.split(text.strip()) if w]


def chunk_text(
    text: str,
    *,
    doc_title: str = "doc",
    chunk_size: int = 120,
    overlap: int = 24,
) -> list[dict]:
    """Split ``text`` into overlapping word windows.

    - ``chunk_size``: max words per chunk.
    - ``overlap``: words repeated from the tail of one chunk at the head of
      the next, so sentences cut at a boundary stay retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = _words(text)
    if not words:
        return []

    step = chunk_size - overlap
    doc_key = f"{_slug(doc_title)}-{hashlib.sha256(doc_title.encode('utf-8')).hexdigest()[:8]}"
    chunks: list[dict] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(
            {
                "id": f"{doc_key}::chunk-{index}",
                "text": " ".join(window),
                "metadata": {
                    "doc_title": doc_title,
                    "chunk_index": index,
                    "word_start": start,
                    "word_count": len(window),
                },
            }
        )
        if start + chunk_size >= len(words):
            break
        start += step
        index += 1
    return chunks


def chunk_documents(
    documents: Iterable[dict],
    *,
    chunk_size: int = 120,
    overlap: int = 24,
) -> list[dict]:
    """Chunk many documents: each is {"title": str, "text": str}."""
    out: list[dict] = []
    for doc in documents:
        out.extend(
            chunk_text(
                doc.get("text", ""),
                doc_title=doc.get("title", "doc"),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return out


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "doc"
