"""Loading and validating the question file.

Each question is one JSON object::

    {"id": "q01", "kind": "answerable",
     "question": "Which ranker did BM25 beat, and on what measure?",
     "doc": "scour-readme",                    # file stem of the source document
     "evidence": "nDCG@10",                    # short exact phrase that proves the answer
     "must_contain": ["tf-idf", "ndcg"]}       # words a correct answer must include

    {"id": "q18", "kind": "unanswerable",
     "question": "What is the capital of France?"}   # the documents do not say

Validation is strict on purpose: a typo in an evidence phrase would silently
turn a retrieval hit into a miss, so every evidence phrase must be found in
its document before anything runs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WS = re.compile(r"\s+")

EXPECTED_QUESTIONS = 20
MIN_UNANSWERABLE = 3


def normalise(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


@dataclass
class Question:
    id: str
    kind: str                       # "answerable" or "unanswerable"
    question: str
    doc: str | None = None
    evidence: str | None = None
    must_contain: list[str] = field(default_factory=list)


class QuestionFileError(ValueError):
    """The question file is wrong; the message says which question and why."""


def load_documents(directory: Path) -> dict[str, str]:
    """Read every .txt and .md file in ``directory``; the key is the file stem."""
    docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(directory.glob("*"))
            if p.suffix.lower() in {".txt", ".md"} and p.name.lower() != "readme.md"}
    if not docs:
        raise QuestionFileError(f"No .txt or .md documents found in {directory}. Add your own documents first.")
    return docs


def load_questions(path: Path, docs: dict[str, str], *, expected: int | None = EXPECTED_QUESTIONS) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise QuestionFileError("The question file must be a JSON list.")
    problems: list[str] = []
    questions: list[Question] = []
    seen: set[str] = set()
    normalised_docs = {name: normalise(text) for name, text in docs.items()}

    for position, item in enumerate(raw, start=1):
        where = f"question {position}"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object")
            continue
        qid = str(item.get("id", "")).strip()
        where = f"question {position} ({qid or 'no id'})"
        if not qid:
            problems.append(f"{where}: missing id")
        elif qid in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(qid)
        kind = item.get("kind")
        text = str(item.get("question", "")).strip()
        if kind not in {"answerable", "unanswerable"}:
            problems.append(f"{where}: kind must be 'answerable' or 'unanswerable'")
        if not text:
            problems.append(f"{where}: missing question text")
        q = Question(id=qid, kind=str(kind), question=text, doc=item.get("doc"),
                     evidence=item.get("evidence"), must_contain=list(item.get("must_contain") or []))
        if kind == "answerable":
            if q.doc not in docs:
                problems.append(f"{where}: doc {q.doc!r} is not one of {sorted(docs)}")
            if not q.evidence or not q.evidence.strip():
                problems.append(f"{where}: answerable questions need an evidence phrase")
            elif len(q.evidence.split()) > 12:
                problems.append(f"{where}: evidence phrase is longer than 12 words; use a short exact phrase")
            elif q.doc in docs and normalise(q.evidence) not in normalised_docs[q.doc]:
                problems.append(f"{where}: evidence phrase {q.evidence!r} does not appear in {q.doc}")
            if not q.must_contain:
                problems.append(f"{where}: answerable questions need must_contain keywords")
        questions.append(q)

    if expected is not None and len(questions) != expected:
        problems.append(f"expected {expected} questions, found {len(questions)}")
    if expected is not None and sum(q.kind == "unanswerable" for q in questions) < MIN_UNANSWERABLE:
        problems.append(f"include at least {MIN_UNANSWERABLE} unanswerable questions "
                        "(ones your documents cannot answer), to test that the agent says so")
    if problems:
        raise QuestionFileError("\n".join(f"- {p}" for p in problems))
    return questions
