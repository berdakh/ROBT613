---
title: Capstone ideas
parent: Guides
nav_order: 7
---

# Capstone ideas
{: .no_toc }

The best capstone scratches your own itch — you will test it far more honestly.
These are starting points, not a menu.

1. TOC
{:toc}

---

## The requirements, again

All four are required:

1. A **local open-weight model** — runs with no API key.
2. **≥3 tools, or RAG** over your own documents.
3. An **agent loop or an explicit chain**, with a trace of a multi-step run.
4. An **honest evaluation** — at least 10 test cases, with a score.

Plus a README: what it does, how to run it, **what it gets wrong**, what you
would do next.

---

## ⭐⭐ Kitchen assistant

*"What can I cook tonight, and what should I buy?"*

**Tools:** `list_pantry`, `expiring_soon`, `search_recipes`, `shopping_list`,
`calculate`.

**The interesting part:** prioritising by expiry date. The naive version lists
what you have; the good version tells you the milk goes tomorrow.

**Evaluation:** 10 pantry states → does it suggest a meal using the
soonest-expiring items? Does it refuse to invent ingredients you do not have?

**Extension:** photograph your fridge and use Qwen3-VL to read it.

---

## ⭐⭐ Budget coach

*"Where does my money actually go, and what should I change?"*

**Tools:** `spending_by_category`, `largest_expenses`, `compare_months`,
`calculate`, `search_notes` (for your stated budget target).

**The interesting part:** getting arithmetic right. Small models are bad at
mental maths, so every number must come from a tool. Your evaluation should
check that.

**Evaluation:** 10 questions with known correct answers computed in pandas. Any
mismatch is a failure — this is one of the few capstones where you can grade
automatically and exactly.

**Extension:** anomaly detection — "this month's utilities are 40% above your
average".

---

## ⭐⭐⭐ Inbox triage

*"What actually needs my attention today?"*

**Tools:** `read_inbox`, `classify_message`, `draft_reply`, `send_message`
(confirmation-gated).

**The interesting part:** this is the capstone where **prompt injection is
real**. `data/inbox.jsonl` contains a phishing email. Does your assistant flag
it, or follow it? Make that a test case.

**Evaluation:** label the 10 sample emails yourself (urgent / action needed /
FYI / spam), then measure agreement. Report the confusion matrix, not just
accuracy.

{: .warning }
> If you add a real send capability, gate it behind explicit confirmation. See
> `send_message` in `demo_tools.py` for the pattern.

---

## ⭐⭐⭐ Study buddy

*"Quiz me on my lecture notes."*

**Tools:** RAG over your notes, `generate_question`, `check_answer`,
`track_progress`.

**The interesting part:** generating questions whose answers are *actually in
the notes*, and grading free-text answers fairly. Both are harder than they
sound and both are measurable.

**Evaluation:** generate 20 questions, check each answer appears in the source.
Then grade 10 deliberately wrong answers and 10 correct ones — does it
distinguish them?

**Extension:** spaced repetition, tracking which topics you keep failing.

---

## ⭐⭐⭐ Trip planner

*"Plan me a weekend in Almaty for under 50,000 tenge."*

**Tools:** `search_notes` (past trips), `calculate`, `current_date`,
`days_between`, `packing_list`, `budget_check`.

**The interesting part:** multi-constraint planning. Budget, dates and
preferences interact, and the agent must revise when a plan exceeds the budget.
Good test of multi-step behaviour.

**Evaluation:** 10 briefs with different budgets. Does the total ever exceed the
stated budget? (It will. Measure how often.)

---

## ⭐⭐⭐⭐ Lab notebook assistant

*"What did I try last time, and what happened?"*

**Tools:** RAG over experiment logs, `plot_results`, `compare_runs`,
`extract_parameters`.

**The interesting part:** structured extraction from semi-structured notes, and
answering comparative questions across documents — which is where naive RAG
struggles, because the answer is in no single chunk.

**Evaluation:** questions whose answers span two or more documents. Measure how
often it finds both.

---

## ⭐⭐⭐⭐ Code reviewer

*"Does this diff follow our conventions?"*

**Tools:** `read_file`, `get_diff`, `search_codebase`, `check_style_rules`.
Use **Qwen2.5-Coder-1.5B/7B** rather than a general model.

**The interesting part:** grounding the review in *your* written conventions
(a RAG corpus of your style guide), not the model's generic opinions.

**Evaluation:** 10 diffs with known issues you planted. What fraction does it
find? How many false positives? Report both — a reviewer that flags everything
is useless.

---

## Bring your own

The best projects at every workshop are the ones nobody suggested. If you have a
recurring annoyance in your own week, that is your capstone. Ask:

1. **Is there a real task?** Something you do repeatedly and dislike.
2. **Can tools supply the facts?** If it needs knowledge the model lacks and you
   cannot retrieve, pick something else.
3. **Can you tell right from wrong?** If you cannot write 10 test cases with
   known answers, you cannot evaluate it — and you will not know if it works.
4. **Is failure harmless?** For a 4-hour project, it must be. Nothing medical,
   legal, or financially binding.

---

## Scoping advice

You have roughly **3 hours of build time**. That means:

| Do | Don't |
|---|---|
| 3 tools that work | 10 tools that half-work |
| Qwen3-0.6B or 4B | Wait 40 minutes for a 14B download |
| A hardcoded test set | A dynamic evaluation framework |
| Terminal output | A polished web UI |
| One thing done well and measured | Three features and no evaluation |

{: .tip }
> Build the tools first and test them **without any model**. Most agent bugs are
> tool bugs. If `spending_by_category("2026-01")` returns the wrong number, no
> amount of prompt engineering will save you.
