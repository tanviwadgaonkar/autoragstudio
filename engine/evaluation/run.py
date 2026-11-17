"""Run the evaluation against a running engine (or an in-process test client).

    python -m evaluation.run --docs evaluation/docs --questions evaluation/questions.json

The engine must be running with real backends (Ollama) for the result to count.
With the offline hashing embedder and echo LLM the run still works, which is how
the harness itself is tested, but the report says it is not a valid result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evaluation.questions import Question, QuestionFileError, load_documents, load_questions, normalise

DONT_KNOW_PHRASES = ("do not know", "don't know", "dont know", "not in the context",
                     "does not contain", "doesn't contain", "no information", "cannot find",
                     "can't find", "not mentioned", "not provided", "unable to find")

OFFLINE_BACKENDS = ("HashingEmbedder", "echo")


def retrieval_hit(sources: list[dict], question: Question) -> bool:
    """True when a retrieved chunk from the right document holds the evidence phrase."""
    needle = normalise(question.evidence or "")
    return any(s.get("metadata", {}).get("doc_title") == question.doc and needle in normalise(s["text"])
               for s in sources)


def keyword_check(answer: str, question: Question) -> bool:
    """Answerable: every required keyword appears. Unanswerable: the answer admits it does not know."""
    text = normalise(answer)
    if question.kind == "unanswerable":
        return any(phrase in text for phrase in DONT_KNOW_PHRASES)
    return all(normalise(word) in text for word in question.must_contain)


def run_eval(client: Any, docs: dict[str, str], questions: list[Question], *,
             agent_id: str | None = None, top_k: int = 4, base_url: str = "") -> dict:
    """Ingest the documents into a throw-away agent, ask every question, return the raw results."""
    agent_id = agent_id or f"eval-{uuid.uuid4().hex[:8]}"
    health = client.get(f"{base_url}/health").json()
    ingest = client.post(f"{base_url}/agents/{agent_id}/ingest", json={
        "documents": [{"title": name, "text": text} for name, text in docs.items()], "reset": True})
    ingest.raise_for_status()
    records = []
    try:
        for q in questions:
            resp = client.post(f"{base_url}/agents/{agent_id}/chat", json={"question": q.question, "top_k": top_k})
            resp.raise_for_status()
            body = resp.json()
            records.append({
                "id": q.id, "kind": q.kind, "question": q.question,
                "answer": body["answer"], "sources": body["sources"],
                "retrieval_hit": retrieval_hit(body["sources"], q) if q.kind == "answerable" else None,
                "keyword_pass": keyword_check(body["answer"], q),
                "manual_grade": None,
            })
    finally:
        client.delete(f"{base_url}/agents/{agent_id}/index")
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": health["embedder"], "llm": health["llm"], "top_k": top_k,
        "chunks_indexed": ingest.json()["total_chunks"], "documents": sorted(docs),
        "valid_for_resume": not (health["embedder"] == OFFLINE_BACKENDS[0] or health["llm"] == OFFLINE_BACKENDS[1]),
        "records": records,
    }


def summarize(results: dict) -> dict:
    records = results["records"]
    answerable = [r for r in records if r["kind"] == "answerable"]
    unanswerable = [r for r in records if r["kind"] == "unanswerable"]
    graded = [r for r in records if r["manual_grade"] is not None]
    return {
        "questions": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval_hits": sum(bool(r["retrieval_hit"]) for r in answerable),
        "keyword_passes": sum(r["keyword_pass"] for r in records),
        "refusals_ok": sum(r["keyword_pass"] for r in unanswerable),
        "graded": len(graded),
        "manual_correct": sum(r["manual_grade"] == "correct" for r in graded),
    }


def render_report(results: dict) -> str:
    s = summarize(results)
    lines = [
        "# AutoRAG evaluation",
        "",
        f"- Run: {results['run_at']}",
        f"- Embedder: `{results['embedder']}`, LLM: `{results['llm']}`, top_k: {results['top_k']}",
        f"- Documents: {', '.join(results['documents'])} ({results['chunks_indexed']} chunks)",
    ]
    if not results["valid_for_resume"]:
        lines += ["", "> **Not a valid result.** This run used the offline hashing embedder or the echo LLM. "
                      "Run again against a real Ollama model before quoting any number."]
    if results["chunks_indexed"] <= 3 * results["top_k"]:
        lines += ["", f"> **Retrieval hit is not meaningful here.** Only {results['chunks_indexed']} chunks were indexed and "
                      f"{results['top_k']} are retrieved per question, so most chunks come back whatever the question. "
                      "Add more documents (a few thousand words at least) before reading anything into it."]
    lines += [
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Retrieval hit (evidence phrase in top {results['top_k']} chunks) | {s['retrieval_hits']} of {s['answerable']} answerable |",
        f"| Automatic answer check (keywords present) | {sum(r['keyword_pass'] for r in results['records'] if r['kind'] == 'answerable')} of {s['answerable']} answerable |",
        f"| Says it does not know when the documents cannot answer | {s['refusals_ok']} of {s['unanswerable']} unanswerable |",
    ]
    if s["graded"]:
        lines.append(f"| **Manual grade: answered correctly** | **{s['manual_correct']} of {s['graded']}** |")
    else:
        lines.append("| Manual grade | not graded yet: run `python -m evaluation.grade <results file>` |")
    lines += ["", "## Per question", "", "| id | kind | retrieval | auto | manual |", "|---|---|---|---|---|"]
    for r in results["records"]:
        hit = "n/a" if r["retrieval_hit"] is None else ("hit" if r["retrieval_hit"] else "miss")
        lines.append(f"| {r['id']} | {r['kind']} | {hit} | {'pass' if r['keyword_pass'] else 'fail'} | {r['manual_grade'] or '-'} |")
    return "\n".join(lines) + "\n"


def save(results: dict, out_dir: Path, stem: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_report(results), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the AutoRAG engine on your own documents and questions.")
    parser.add_argument("--docs", type=Path, default=Path("evaluation/docs"))
    parser.add_argument("--questions", type=Path, default=Path("evaluation/questions.json"))
    parser.add_argument("--engine", default="http://localhost:8001", help="Base URL of the running engine")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--expected", type=int, default=20, help="Number of questions the file must contain")
    args = parser.parse_args(argv)
    try:
        docs = load_documents(args.docs)
        questions = load_questions(args.questions, docs, expected=args.expected or None)
    except (QuestionFileError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Cannot start:\n{exc}", file=sys.stderr)
        return 2
    try:
        with httpx.Client(base_url=args.engine, timeout=300.0) as client:
            results = run_eval(client, docs, questions, top_k=args.top_k)
    except httpx.HTTPError as exc:
        print(f"The engine at {args.engine} did not answer properly: {exc}\n"
              "Is it running (uvicorn app.main:app --port 8001) and is Ollama up?", file=sys.stderr)
        return 3
    results["questions_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()[:12]
    json_path, md_path = save(results, args.out)
    print(render_report(results))
    print(f"Saved {json_path} and {md_path}")
    if results["valid_for_resume"]:
        print("Next: read every answer and grade it:  python -m evaluation.grade", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
