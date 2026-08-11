"""AutoRAG Studio — inference engine (FastAPI).

Endpoints (all agent-scoped; an agent's index is its own Chroma collection):

    GET    /health                     backend + mode info
    POST   /agents/{agent_id}/ingest   chunk + embed + index documents
    POST   /agents/{agent_id}/query    retrieval only (debugging retrieval)
    POST   /agents/{agent_id}/chat     retrieval + LLM answer with sources
    GET    /agents/{agent_id}/stats    chunk count for the agent
    DELETE /agents/{agent_id}/index    drop the agent's index

Run:  uvicorn app.main:app --port 8001 --reload   (from engine/)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .deps import get_pipeline
from .rag.store import AgentIndexNotFound
from .schemas import (
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceOut,
    StatsResponse,
)

app = FastAPI(title="AutoRAG Studio Engine", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipe = get_pipeline()
    return HealthResponse(
        status="ok",
        embedder=type(pipe.embedder).__name__,
        llm=getattr(pipe.llm, "name", type(pipe.llm).__name__),
    )


@app.post("/agents/{agent_id}/ingest", response_model=IngestResponse)
def ingest(agent_id: str, payload: IngestRequest) -> IngestResponse:
    pipe = get_pipeline()
    result = pipe.ingest(
        agent_id,
        [d.model_dump() for d in payload.documents],
        system_prompt=payload.system_prompt,
        reset=payload.reset,
    )
    return IngestResponse(**result)


@app.post("/agents/{agent_id}/query", response_model=QueryResponse)
def query(agent_id: str, payload: QueryRequest) -> QueryResponse:
    pipe = get_pipeline()
    try:
        sources = pipe.retrieve(agent_id, payload.question, top_k=payload.top_k)
    except AgentIndexNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return QueryResponse(
        agent_id=agent_id,
        sources=[SourceOut(text=s.text, score=s.score, metadata=s.metadata) for s in sources],
    )


@app.post("/agents/{agent_id}/chat", response_model=ChatResponse)
def chat(agent_id: str, payload: QueryRequest) -> ChatResponse:
    pipe = get_pipeline()
    try:
        result = pipe.answer(agent_id, payload.question, top_k=payload.top_k)
    except AgentIndexNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatResponse(
        agent_id=agent_id,
        answer=result.answer,
        sources=[
            SourceOut(text=s.text, score=s.score, metadata=s.metadata)
            for s in result.sources
        ],
    )


@app.get("/agents/{agent_id}/stats", response_model=StatsResponse)
def stats(agent_id: str) -> StatsResponse:
    pipe = get_pipeline()
    return StatsResponse(agent_id=agent_id, total_chunks=pipe.store.count(agent_id))


@app.delete("/agents/{agent_id}/index")
def delete_index(agent_id: str) -> dict:
    pipe = get_pipeline()
    deleted = pipe.store.delete_agent(agent_id)
    return {"agent_id": agent_id, "deleted": deleted}
