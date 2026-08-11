"""HTTP bridge from the studio to the inference engine."""

from __future__ import annotations

import requests
from django.conf import settings

from .models import Document


class EngineError(RuntimeError):
    pass


def push_document(document: Document, *, timeout: float = 120.0) -> dict:
    """Send one document to the engine for chunking + embedding + indexing.

    Updates ``document.status`` to indexed/failed and stores the error text
    on failure so it is visible in the admin.
    """
    url = f"{settings.ENGINE_URL}/agents/{document.agent.slug}/ingest"
    payload = {
        "documents": [{"title": document.title, "text": document.text}],
        "system_prompt": document.agent.system_prompt or None,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        document.status = Document.Status.FAILED
        document.last_error = str(exc)[:300]
        document.save(update_fields=["status", "last_error"])
        raise EngineError(str(exc)) from exc

    document.status = Document.Status.INDEXED
    document.last_error = ""
    document.save(update_fields=["status", "last_error"])
    return resp.json()


def drop_agent_index(agent_slug: str, *, timeout: float = 30.0) -> bool:
    url = f"{settings.ENGINE_URL}/agents/{agent_slug}/index"
    try:
        resp = requests.delete(url, timeout=timeout)
        resp.raise_for_status()
        return bool(resp.json().get("deleted"))
    except requests.RequestException as exc:
        raise EngineError(str(exc)) from exc
