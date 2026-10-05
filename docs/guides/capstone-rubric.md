---
title: Capstone rubric
parent: Guides
nav_order: 8
---

# Capstone rubric
{: .no_toc }

For instructors marking the day-4 capstone, and for students who want to know
exactly what is being judged.

1. TOC
{:toc}

---

{: .note }
> Exercise and checkpoint answers for the other notebooks live in
> [`solutions/`](https://github.com/berdakh/ROBT613/tree/master/solutions).

## How to use this

**Students:** this is not a surprise. Read it on day 1, build against it, and
self-assess before you demo. The last column of each row tells you what full
marks looks like.

**Instructors:** 100 points across five sections. The weighting is deliberate —
**evaluation and honesty together are worth more than the implementation**,
because that is the part this workshop exists to teach. A beautiful demo with no
measurement should not beat a modest one that is properly evaluated.

---

## Section 1 — It runs (20 points)

| Points | Criterion |
|:---:|:---|
| 8 | **It runs from a clean clone.** Follow the README on a machine that has never seen the project. It works, or it does not. |
| 6 | **Local open-weight model.** No API key, no hosted service in the required path. |
| 4 | **A multi-step run is demonstrated**, not just a single call. |
| 2 | **Dependencies are declared.** `requirements.txt` or equivalent is complete. |

{: .note }
> Mark the first row by actually doing it. "It works on my machine" is the thing
> this row exists to catch.

---

## Section 2 — Engineering (25 points)

| Points | Criterion |
|:---:|:---|
| 8 | **Three or more tools, or RAG over their own documents.** Tools have validated arguments and docstrings a model can use. |
| 6 | **An agent loop or an explicit chain**, with a trace of a multi-step run included. |
| 5 | **A step limit exists and was tested.** Ask them to show it being hit. |
| 4 | **Tools fail safely.** Errors return as text the model can act on; nothing crashes the loop. |
| 2 | **No `eval`, `exec`, shell injection, or unconstrained file access.** |

**Automatic deduction:** `eval()` on model output costs the whole section,
however well the rest is built. This is stated plainly on day 2.

---

## Section 3 — Evaluation (30 points)

The heaviest section. This is the discriminator.

| Points | Criterion |
|:---:|:---|
| 10 | **At least 10 test cases**, covering the happy path, an edge case, and something the system should refuse. |
| 8 | **Scores are reported**, not vibes. A table or printed summary. |
| 7 | **A before/after comparison.** They changed one thing — prompt, model, `max_steps`, a tool description — and measured the effect. |
| 5 | **The test set was written before tuning.** Ask. The answer is usually honest and usually informative. |

A capstone with no evaluation **cannot exceed 70/100**, regardless of how
impressive the demo is.

---

## Section 4 — Honesty (15 points)

| Points | Criterion |
|:---:|:---|
| 8 | **Three specific failure modes named**, from observation rather than speculation. "It sometimes gets things wrong" scores 0; "it picks `calculate` when it needs `search_notes` on questions phrased as commands" scores full. |
| 4 | **Limits of the evaluation acknowledged** — small test set, one model, no real users. |
| 3 | **Nothing is overclaimed.** No "production-ready" on a four-hour project. |

Reward students who show you something broken. They are doing the harder and
more useful thing.

---

## Section 5 — Communication (10 points)

| Points | Criterion |
|:---:|:---|
| 4 | **README** covers what it does, how to run it, what it gets wrong, what is next. |
| 3 | **The demo runs** in the five minutes allotted (a recording is an acceptable backup). |
| 3 | **They can explain their own architecture** when asked why they chose it. |

---

## Not marked

State these up front so nobody optimises for them:

- **Model size.** A 0.6B project evaluated well beats an 8B one that is not.
- **Framework choice.** A hand-written loop is worth as much as LangChain.
- **UI polish.** Terminal output is fine. Gradio earns no points by itself.
- **Feature count.** One thing done well and measured beats four half-built.
- **Whether the results are good.** A strategy that honestly reports poor
  performance is a success. Hiding it is the failure.

---

## Grade boundaries

| Range | What it means |
|---|---|
| 90–100 | Runs, measured, honest about limits. Would survive a code review. |
| 75–89 | Solid. Usually loses points on evaluation depth or thin failure analysis. |
| 60–74 | Works, but under-measured. The most common outcome — and the most fixable. |
| 40–59 | Runs partly, or no meaningful evaluation. |
| < 40 | Does not run from a clean clone, or there is nothing to demonstrate. |

---

## A one-page version for the day

```
RUNS           /20   clean clone · local model · multi-step · deps
ENGINEERING    /25   tools · loop · step limit · safe errors · no eval()
EVALUATION     /30   10+ cases · scores · before/after · written first
HONESTY        /15   3 specific failures · limits stated · no overclaiming
COMMUNICATION  /10   README · demo · can explain it
                     ----
                     /100     (no evaluation -> capped at 70)
```

---

## Marking notes from experience

- **Run it yourself before the demo.** Students present their best path; the
  clean-clone test finds what the demo hides.
- **Ask "what surprised you?"** The answer separates people who watched their
  system from people who assembled it.
- **Ask to see a failure.** A student who can produce one on demand understands
  their system. One who cannot usually has not looked.
- **Be generous with partial credit on evaluation.** Five honest test cases are
  far closer to the point than twenty invented after the fact.
