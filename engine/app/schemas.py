"""Pydantic v2 schemas for the AutoRAG engine API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)
    system_prompt: str | None = None
    reset: bool = False


class IngestResponse(BaseModel):
    agent_id: str
    documents: int
    chunks_indexed: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceOut(BaseModel):
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    agent_id: str
    sources: list[SourceOut]


class ChatResponse(BaseModel):
    agent_id: str
    answer: str
    sources: list[SourceOut]


class StatsResponse(BaseModel):
    agent_id: str
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    embedder: str
    llm: str
