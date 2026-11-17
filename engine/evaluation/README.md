# Evaluating AutoRAG on your own documents

This measures how reliably the engine answers questions about documents you chose, with the real local models. It takes about an hour. **The numbers in the final report come from your run and your grading, so write the questions yourself.**

## 1. Add documents

Put 3 to 6 `.txt` or `.md` files in `evaluation/docs/` (see the README there), a few thousand words in total. Public material you own only. With too little text the engine returns almost every chunk for every question, and the report warns you that the retrieval number means nothing.

## 2. Write 20 questions

Create `evaluation/questions.json`: a list of 20 objects. At least 3 must be unanswerable (the documents do not say), to check that the agent admits it does not know instead of inventing an answer.

```json
{"id": "q01", "kind": "answerable",
 "question": "When does the fish market start selling?",
 "doc": "harbour-guide",
 "evidence": "half past six",
 "must_contain": ["half past six"]}

{"id": "q18", "kind": "unanswerable",
 "question": "How much does a ticket for the ferry cost?"}
```

- `doc` is the file name without the extension.
- `evidence` is a short exact phrase (at most 12 words) copied from that document that proves the answer. The harness checks that it really appears in the document and refuses to run if it does not.
- `must_contain` lists words a correct answer has to include.
- Write questions a person would really ask: some direct lookups, some that need two sentences, some with numbers. Do not write them to be easy.

`evaluation/example/` has a complete tiny example (6 questions; run it with `--expected 6`) so you can see the format.

## 3. Run it with the real models

```bash
ollama pull nomic-embed-text && ollama pull llama3.2
cd engine
uvicorn app.main:app --port 8001          # real backends, not RAG_EMBEDDER=hash
python -m evaluation.run                  # in another terminal, from engine/
```

The run ingests the documents into a throw-away agent, asks all 20 questions, saves `evaluation/results/<time>.json` and `.md`, and deletes the agent. A run on the offline hashing embedder or echo model still works but is marked **not a valid result**.

## 4. Grade every answer yourself

```bash
python -m evaluation.grade evaluation/results/<time>.json
```

Read each answer next to its sources and press `c` (correct) or `w` (wrong). A partly right answer is wrong. For an unanswerable question, correct means it said it does not know. The report is rewritten after every grade.

The two automatic checks (evidence phrase retrieved, keywords present) are only guards: a lucky keyword can pass a wrong answer. **Only the manual grade is the "N of 20" you may quote.**

## 5. Publish the evidence

Commit the `.json` and `.md` from your graded run, and paste the table into the main README under "Evaluation" with the date, the models and the number of questions. If the result is poor, say so and say what you would change; a bad number with an explanation reads better than a missing one.

## What this does not measure

Twenty questions on a handful of documents is a small sample. It says nothing about other document types, long documents or other models, and the grader is the author. Say so if asked.
