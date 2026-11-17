"""A small, honest evaluation harness for the AutoRAG engine.

You bring your own documents and 20 questions; the harness ingests the
documents into a throw-away agent, asks every question through the engine's
HTTP API, and reports three things:

* retrieval hit rate: did a retrieved chunk contain the evidence phrase?
* automatic answer check: does the answer contain the required keywords (or,
  for questions the documents cannot answer, does it admit it does not know)?
* manual grade: you read each answer and mark it right or wrong.

Only the manual grade is a statement about answer quality. The two automatic
checks are cheap, repeatable guards that catch obvious failures.
"""
