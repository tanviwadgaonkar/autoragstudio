# AutoRAG Studio

A multi-agent RAG platform that runs entirely on your machine. Create agents, give each one its own documents, and chat with them — every agent retrieves only from its own isolated ChromaDB vector store, and generation runs on locally hosted LLMs through [Ollama](https://ollama.com). The chunking → embedding → retrieval pipeline is written from scratch (no LangChain), so every step is inspectable and testable.

## Architecture

```
 React (Vite, :5173)
   │
   ├── /studio/* ──▶  Django "Studio" (:8000)         ── source of truth for
   │                  agents + documents (SQLite,        WHAT exists
   │                  admin, JSON API)
   │                        │  POST /agents/{slug}/ingest
   │                        ▼
   └── /engine/* ──▶  FastAPI "Engine" (:8001)        ── source of truth for
                      chunk → embed → store →            the VECTOR INDEX
                      retrieve → generate
                            │                │
                        ChromaDB          Ollama
                   (one collection     (embeddings +
                     per agent)         chat model)
```

The two services share nothing but the **agent slug** — the Engine keys its Chroma collections by it. That gives the platform its core guarantee: **per-agent isolation**. Agent A can never retrieve agent B's documents; the test suite proves it at both the store layer and through the HTTP API.

## Quickstart

### 1. Engine (FastAPI)

```bash
cd engine
pip install -r requirements.txt

# with local models (recommended):
ollama pull nomic-embed-text && ollama pull llama3.2
uvicorn app.main:app --port 8001 --reload

# or fully offline (no models; hashing embedder + echo LLM):
RAG_EMBEDDER=hash RAG_LLM=echo uvicorn app.main:app --port 8001
```

### 2. Studio (Django)

```bash
cd studio
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin
python manage.py runserver 8000
```

### 3. Frontend (React)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies to :8000 and :8001)
```

Create an agent in the sidebar, paste a document, and ask it a question — the answer comes back with the exact source chunks and similarity scores it was grounded on.

## Tests

```bash
cd engine
pytest -q          # 22 passed
```

The 22-test suite covers **ingestion** (chunk sizing, overlap, deterministic ids; embedding dimensionality and determinism), **retrieval** (per-agent isolation, ranking, top-k, metadata round-trips), and **API behaviour** (ingest/chat/stats endpoints, 404 on unknown agents, 422 validation, isolation through HTTP). Tests run fully offline in under a second: a deterministic hashing embedder and an echo LLM stand in for Ollama, and Chroma runs in-memory — the same interfaces, no network.

## Engine API

| Method | Path                        | Purpose                                  |
| ------ | --------------------------- | ---------------------------------------- |
| GET    | `/health`                   | Active embedder + LLM backends           |
| POST   | `/agents/{id}/ingest`       | Chunk + embed + index documents          |
| POST   | `/agents/{id}/query`        | Retrieval only — for debugging retrieval |
| POST   | `/agents/{id}/chat`         | Retrieve + generate; answer with sources |
| GET    | `/agents/{id}/stats`        | Indexed chunk count                      |
| DELETE | `/agents/{id}/index`        | Drop the agent's index                   |

Interactive docs at `http://localhost:8001/docs`.

## Design notes

**Why build the RAG layer from scratch?** To understand what the frameworks abstract. Chunking is a ~60-line sliding word window with overlap; embeddings and generation sit behind two-method interfaces with an Ollama implementation and an offline implementation each; retrieval is a cosine query against a per-agent collection. When retrieval brings back the wrong document, every knob is in this repo: chunk size, overlap, `top_k`, the embedding model, and the metadata on each chunk.

**Why a separate `/query` endpoint?** Most RAG failures are retrieval failures. `/query` shows exactly which chunks matched and at what score *before* any model touches them, which turns "the answer is wrong" into "chunk 3 outranked chunk 1, and here's why".

**Why offline test doubles?** CI should not depend on a 4 GB model download. The hashing embedder is deterministic bag-of-words — crude, but word overlap still produces meaningful cosine ordering, which is exactly enough to test the pipeline's plumbing end to end.

**Environment variables** (engine): `RAG_EMBEDDER` = `ollama` | `hash` · `RAG_LLM` = `ollama` | `echo` · `RAG_CHROMA` = `persistent` | `ephemeral` · `RAG_CHROMA_DIR` · `OLLAMA_URL` · `OLLAMA_EMBED_MODEL` · `OLLAMA_CHAT_MODEL`. Studio: `ENGINE_URL`, `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`.

## Project structure

```
engine/    FastAPI service + the RAG core (app/rag: chunker, embeddings, store, llm, pipeline) + 22 tests
studio/    Django backend: agent registry, document store, admin, JSON API
frontend/  React (Vite) UI: agent sidebar, document ingest, chat with sources
```

## Roadmap

- Tool-calling agent steps with retry/fallback on failure
- A scripted evaluation loop (fixed question set, prompt A/B comparison, accuracy + latency logging)
- File upload (PDF/DOCX) ingestion in the Studio
