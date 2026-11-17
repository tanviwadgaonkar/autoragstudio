"""The evaluation harness: it must refuse bad inputs and never inflate a result."""
import json
from pathlib import Path

import pytest

from evaluation import grade, run
from evaluation.questions import Question, QuestionFileError, load_documents, load_questions

EXAMPLE = Path(__file__).resolve().parent.parent / "evaluation" / "example"


@pytest.fixture
def docs():
    return load_documents(EXAMPLE / "docs")


def write(tmp_path, items):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(items))
    return path


def good(**overrides):
    item = {"id": "q1", "kind": "answerable", "question": "When does the fish market start?",
            "doc": "harbour-guide", "evidence": "half past six", "must_contain": ["six"]}
    item.update(overrides)
    return item


# --- question file validation ------------------------------------------------

def test_example_questions_load(docs):
    questions = load_questions(EXAMPLE / "questions.json", docs, expected=6)
    assert [q.kind for q in questions] == ["answerable"] * 3 + ["unanswerable"] * 3


def test_evidence_must_appear_in_the_document(tmp_path, docs):
    path = write(tmp_path, [good(evidence="a phrase that is not in the document")])
    with pytest.raises(QuestionFileError, match=r"q1.*does not appear in harbour-guide"):
        load_questions(path, docs, expected=None)


def test_evidence_matching_ignores_case_and_line_breaks(tmp_path, docs):
    path = write(tmp_path, [good(evidence="HALF PAST   six")])
    assert load_questions(path, docs, expected=None)[0].id == "q1"


@pytest.mark.parametrize("change, message", [
    ({"doc": "other-file"}, "is not one of"),
    ({"kind": "maybe"}, "kind must be"),
    ({"question": " "}, "missing question text"),
    ({"must_contain": []}, "need must_contain"),
    ({"evidence": " ".join(["word"] * 13)}, "longer than 12 words"),
    ({"id": ""}, "missing id"),
])
def test_bad_questions_are_named_and_refused(tmp_path, docs, change, message):
    path = write(tmp_path, [good(**change)])
    with pytest.raises(QuestionFileError, match=message):
        load_questions(path, docs, expected=None)


def test_duplicate_ids_are_refused(tmp_path, docs):
    with pytest.raises(QuestionFileError, match="duplicate id"):
        load_questions(write(tmp_path, [good(), good()]), docs, expected=None)


def test_exactly_twenty_questions_with_three_unanswerable_are_required(tmp_path, docs):
    with pytest.raises(QuestionFileError, match="expected 20 questions, found 1"):
        load_questions(write(tmp_path, [good()]), docs)
    items = [good(id=f"q{i}") for i in range(20)]
    with pytest.raises(QuestionFileError, match="at least 3 unanswerable"):
        load_questions(write(tmp_path, items), docs)
    items[:3] = [{"id": f"u{i}", "kind": "unanswerable", "question": f"Unknown {i}?"} for i in range(3)]
    assert len(load_questions(write(tmp_path, items), docs)) == 20


def test_no_documents_is_a_clear_error(tmp_path):
    (tmp_path / "README.md").write_text("instructions only")
    with pytest.raises(QuestionFileError, match="Add your own documents"):
        load_documents(tmp_path)


# --- automatic checks -----------------------------------------------------------

def q(**kw):
    base = dict(id="q", kind="answerable", question="?", doc="d", evidence="half past six", must_contain=["six"])
    base.update(kw)
    return Question(**base)


def source(text, doc="d"):
    return {"text": text, "score": 0.5, "metadata": {"doc_title": doc}}


def test_retrieval_hit_needs_the_right_document_and_the_phrase():
    assert run.retrieval_hit([source("Selling starts at HALF  past\nsix daily")], q())
    assert not run.retrieval_hit([source("Selling starts at half past six", doc="other")], q())
    assert not run.retrieval_hit([source("Selling starts at seven")], q())
    assert not run.retrieval_hit([], q())


def test_keyword_check_needs_every_keyword():
    question = q(must_contain=["six", "market"])
    assert run.keyword_check("The market opens at six.", question)
    assert not run.keyword_check("It opens at six.", question)


def test_unanswerable_passes_only_when_the_agent_admits_it():
    question = q(kind="unanswerable")
    assert run.keyword_check("I do not know; the context does not say.", question)
    assert not run.keyword_check("A ferry ticket costs ten pounds.", question)


# --- a full run through the real API (offline backends) --------------------------

def test_full_run_on_the_example_is_flagged_not_valid(client, docs):
    questions = load_questions(EXAMPLE / "questions.json", docs, expected=6)
    results = run.run_eval(client, docs, questions, agent_id="eval-test", top_k=2)
    assert results["valid_for_resume"] is False
    assert len(results["records"]) == 6
    assert results["records"][3]["retrieval_hit"] is None
    assert "Not a valid result" in run.render_report(results)
    assert client.get("/agents/eval-test/stats").json()["total_chunks"] == 0   # throw-away agent removed


def test_retrieval_is_actually_measured(client, docs):
    questions = load_questions(EXAMPLE / "questions.json", docs, expected=6)
    results = run.run_eval(client, docs, questions, agent_id="eval-hits", top_k=1)
    answerable = [r for r in results["records"] if r["kind"] == "answerable"]
    # With one chunk per question and the bag-of-words embedder the lexical questions land on the right chunk.
    assert all(r["retrieval_hit"] is not None for r in answerable)
    assert run.summarize(results)["retrieval_hits"] == sum(r["retrieval_hit"] for r in answerable)


# --- grading and report -------------------------------------------------------------

def results_with(grades):
    records = [{"id": f"q{i}", "kind": "answerable", "question": "?", "answer": "a", "sources": [],
                "retrieval_hit": True, "keyword_pass": True, "manual_grade": g}
               for i, g in enumerate(grades)]
    return {"run_at": "now", "embedder": "OllamaEmbedder", "llm": "ollama:llama3.2", "top_k": 4,
            "chunks_indexed": 9, "documents": ["d"], "valid_for_resume": True, "records": records}


def test_manual_grades_drive_the_headline_number():
    results = results_with([None, None, None])
    grade.apply_grade(results, "q0", "correct")
    grade.apply_grade(results, "q1", "wrong")
    summary = run.summarize(results)
    assert (summary["graded"], summary["manual_correct"]) == (2, 1)
    assert "**1 of 2**" in run.render_report(results)


def test_ungraded_report_says_so_and_quotes_no_number():
    report = run.render_report(results_with([None, None]))
    assert "not graded yet" in report and "Manual grade: answered correctly" not in report


def test_invalid_grades_and_unknown_questions_are_refused():
    results = results_with([None])
    with pytest.raises(ValueError):
        grade.apply_grade(results, "q0", "maybe")
    with pytest.raises(KeyError):
        grade.apply_grade(results, "nope", "correct")


def test_grading_an_offline_run_is_refused(tmp_path, capsys):
    results = results_with([None])
    results["valid_for_resume"] = False
    path = tmp_path / "r.json"
    path.write_text(json.dumps(results))
    assert grade.main([str(path)]) == 2
    assert "mean nothing" in capsys.readouterr().out


# --- command line ---------------------------------------------------------------------

def test_cli_refuses_a_bad_question_file_before_touching_the_engine(tmp_path, capsys):
    bad = write(tmp_path, [good(evidence="not in the document")])
    code = run.main(["--docs", str(EXAMPLE / "docs"), "--questions", str(bad), "--expected", "0"])
    assert code == 2
    assert "does not appear in harbour-guide" in capsys.readouterr().err


def test_cli_reports_an_unreachable_engine(capsys):
    code = run.main(["--docs", str(EXAMPLE / "docs"), "--questions", str(EXAMPLE / "questions.json"),
                     "--expected", "0", "--engine", "http://127.0.0.1:9"])
    assert code == 3
    assert "Is it running" in capsys.readouterr().err


def test_tiny_corpus_gets_a_retrieval_warning_and_a_big_one_does_not():
    small = results_with([None])
    small.update(chunks_indexed=6, top_k=4)
    assert "Retrieval hit is not meaningful" in run.render_report(small)
    large = results_with([None])
    large.update(chunks_indexed=60, top_k=4)
    assert "Retrieval hit is not meaningful" not in run.render_report(large)
