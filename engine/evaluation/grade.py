"""Grade the answers by hand. The manual grade is the only number that goes on a resume.

    python -m evaluation.grade evaluation/results/<run>.json

For each question you see the answer and the retrieved sources and press
c (correct), w (wrong) or s (skip for now). Grades are saved after every
answer, so you can stop and continue later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.run import render_report, summarize

VALID = {"correct", "wrong"}


def apply_grade(results: dict, question_id: str, grade: str) -> None:
    if grade not in VALID:
        raise ValueError(f"grade must be one of {sorted(VALID)}")
    for record in results["records"]:
        if record["id"] == question_id:
            record["manual_grade"] = grade
            return
    raise KeyError(question_id)


def save(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(render_report(results), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade evaluation answers by hand.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--regrade", action="store_true", help="Also go back over questions already graded")
    args = parser.parse_args(argv)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    if not results.get("valid_for_resume"):
        print("This run used the offline hashing embedder or echo LLM. Grading it would mean nothing.")
        return 2
    for record in results["records"]:
        if record["manual_grade"] and not args.regrade:
            continue
        print("\n" + "=" * 70)
        print(f"{record['id']} ({record['kind']}): {record['question']}")
        print(f"\nANSWER:\n{record['answer']}")
        for i, src in enumerate(record["sources"][:2], start=1):
            print(f"\n  source {i} [{src['metadata'].get('doc_title')}]: {src['text'][:300]}")
        print(f"\n  retrieval: {record['retrieval_hit']}   automatic check: {record['keyword_pass']}")
        while True:
            choice = input("Is the answer correct? [c]orrect / [w]rong / [s]kip: ").strip().lower()
            if choice in {"c", "w", "s"}:
                break
        if choice != "s":
            apply_grade(results, record["id"], "correct" if choice == "c" else "wrong")
            save(results, args.results)
    s = summarize(results)
    save(results, args.results)
    print(f"\nGraded {s['graded']} of {s['questions']}. Correct: {s['manual_correct']} of {s['graded']}.")
    if s["graded"] == s["questions"]:
        print(f"For the resume: answered {s['manual_correct']} of {s['questions']} evaluation questions correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
