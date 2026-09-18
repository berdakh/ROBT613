# Sample data

Small, synthetic, safe-to-share files so every exercise has something real to
chew on. Nothing here describes a real person.

| File | Used by | What it is |
|---|---|---|
| `notes/*.md` | 08, 09 | Personal notes — the RAG corpus |
| `pantry.txt` | 10, 12 | What's in the kitchen, for the meal-planner agent |
| `expenses.csv` | 10, 12 | Two months of spending, for the budget agent |
| `inbox.jsonl` | 12 | A fake inbox, for the triage capstone |
| `eval_questions.json` | 09 | Questions + expected sources, for measuring retrieval |

**Swap these for your own files.** The exercises get dramatically more
interesting when the assistant is answering questions about *your* notes. That
is also the moment to re-read `docs/guides/privacy-and-safety.md`: running the
model locally is exactly what makes using your real data reasonable.
