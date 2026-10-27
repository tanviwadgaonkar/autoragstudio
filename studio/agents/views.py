"""Minimal JSON API for the React frontend.

Local-development API (CSRF-exempt JSON endpoints); the admin remains the
authenticated management surface. In a hardened deployment these would sit
behind session or token auth.

CSRF: the endpoints are exempt from Django's token check, so a page on another
site could otherwise POST to a Studio running on localhost with a "simple"
request (e.g. Content-Type: text/plain), which browsers send without a CORS
preflight. Requiring Content-Type: application/json forces a preflight that
this server never approves, so cross-site writes are blocked; DELETE always
needs a preflight.
"""

from __future__ import annotations

import json
from functools import wraps

from django.db.models import Count
from django.http import HttpRequest, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import Agent, Document
from .services import EngineError, push_document


def json_posts_only(view):
    """Reject POSTs that are not application/json (see module docstring: CSRF)."""
    @wraps(view)
    def wrapper(request: HttpRequest, *args, **kwargs):
        if request.method == "POST" and request.content_type != "application/json":
            return JsonResponse({"error": "Content-Type must be application/json"}, status=415)
        return view(request, *args, **kwargs)
    return wrapper


def _json_body(request: HttpRequest) -> dict | None:
    """Parsed JSON object body, or None if the body is not a JSON object."""
    try:
        body = json.loads(request.body or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _agent_dict(agent: Agent) -> dict:
    return {
        "id": agent.id,
        "name": agent.name,
        "slug": agent.slug,
        "description": agent.description,
        "system_prompt": agent.system_prompt,
        "documents": agent.doc_count if hasattr(agent, "doc_count") else agent.documents.count(),
    }


def _doc_dict(doc: Document) -> dict:
    return {
        "id": doc.id,
        "title": doc.title,
        "status": doc.status,
        "last_error": doc.last_error,
        "created_at": doc.created_at.isoformat(),
    }


@csrf_exempt
@require_http_methods(["GET", "POST"])
@json_posts_only
def agents_view(request: HttpRequest) -> JsonResponse:
    if request.method == "GET":
        agents = Agent.objects.annotate(doc_count=Count("documents"))
        return JsonResponse({"agents": [_agent_dict(a) for a in agents]})

    body = _json_body(request)
    if body is None:
        return JsonResponse({"error": "body must be a JSON object"}, status=400)
    name = (body.get("name") or "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)
    if Agent.objects.filter(name__iexact=name).exists():
        return JsonResponse({"error": "agent already exists"}, status=409)

    agent = Agent.objects.create(
        name=name,
        description=(body.get("description") or "").strip(),
        system_prompt=(body.get("system_prompt") or "").strip(),
    )
    return JsonResponse(_agent_dict(agent), status=201)


@csrf_exempt
@require_http_methods(["GET", "POST"])
@json_posts_only
def documents_view(request: HttpRequest, slug: str) -> JsonResponse:
    agent = get_object_or_404(Agent, slug=slug)

    if request.method == "GET":
        return JsonResponse(
            {
                "agent": agent.slug,
                "documents": [_doc_dict(d) for d in agent.documents.all()],
            }
        )

    body = _json_body(request)
    if body is None:
        return JsonResponse({"error": "body must be a JSON object"}, status=400)
    title = (body.get("title") or "").strip()
    text = (body.get("text") or "").strip()
    if not title or not text:
        return JsonResponse({"error": "title and text are required"}, status=400)

    doc, _created = Document.objects.update_or_create(
        agent=agent, title=title, defaults={"text": text}
    )
    try:
        push_document(doc)
    except EngineError:
        pass  # status/last_error already recorded on the document
    return JsonResponse(_doc_dict(doc), status=201)


@csrf_exempt
@require_http_methods(["POST"])
@json_posts_only
def reindex_view(request: HttpRequest, slug: str, doc_id: int) -> JsonResponse:
    doc = get_object_or_404(Document, id=doc_id, agent__slug=slug)
    try:
        result = push_document(doc)
    except EngineError as exc:
        return JsonResponse({"error": str(exc)}, status=502)
    return JsonResponse({"document": _doc_dict(doc), "engine": result})


@csrf_exempt
@require_http_methods(["DELETE"])
def agent_detail_view(request: HttpRequest, slug: str) -> JsonResponse:
    agent = get_object_or_404(Agent, slug=slug)
    agent.delete()  # post_delete signal drops the engine index
    return JsonResponse({"deleted": slug})
