# Sample data

Small, synthetic, safe-to-share files so every exercise has something real to
chew on. Nothing here describes a real person.

| File | Used by | What it is |
|---|---|---|
| `notes/*.md` | 08, 09 | Personal notes — the RAG corpus |
| `pantry.txt` | 10, 12 | What's in the kitchen, for the meal-planner agent |
| `expenses.csv` | 10, 12 | Two months of spending, for the budget agent |
| `inbox.jsonl` | 12 | A fake inbox, for the triage capstone |
| `eval_questions.json` | 09 | Questions over `notes/`, for the walkthrough |
| `eval_questions_docs.json` | 09 | 22 questions over `docs/` (~400 chunks), for the real measurement |

### Why two evaluation sets?

`notes/` is 8 chunks — fine for showing the pipeline, useless for measuring it.
Retrieving 3 of 8 scores well by accident. So notebook 09 evaluates against the
workshop's own `docs/` folder instead, about 400 chunks, where recall@3 is
informative and tuning actually moves it.

Every expected source in `eval_questions_docs.json` was checked against the real
text before it was written down. An evaluation set whose right answers are wrong
is worse than no evaluation at all.

**Swap these for your own files.** The exercises get dramatically more
interesting when the assistant is answering questions about *your* notes. That
is also the moment to re-read `docs/guides/privacy-and-safety.md`: running the
model locally is exactly what makes using your real data reasonable.
