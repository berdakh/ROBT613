---
title: "Day 3 · Knowledge & agents"
parent: Daily plan
nav_order: 3
---

# Day 3 — Your documents, and a model that acts
{: .no_toc }

**Goal:** by this evening, the model answers questions about *your* notes with
citations, and an agent chains tools together on its own.

1. TOC
{:toc}

---

## Schedule

| Time | Activity | Notebook |
|---|---|---|
| 0:00–0:15 | Recap; share the tools you designed last night | — |
| 0:15–1:15 | Embeddings and semantic search | `08_embeddings_and_search` |
| 1:15–2:45 | RAG over your own documents, with evaluation | `09_rag_over_your_notes` |
| 2:45–3:00 | *Break* | — |
| 3:00–4:45 | Agents from scratch | `10_agents_from_scratch` |
| 4:45–5:30 | Failure modes, prompt injection, guardrails | `10` (second half) |
| 5:30–6:00 | Lab: point RAG at your own documents | — |

## The one idea

> **Fine-tuning teaches a model how to speak. RAG tells it what to say.
> Tools let it do things. An agent decides which to do next.**

Most real systems need the middle two and rarely the first.

## What to pay attention to

**In notebook 08**, the "invoice 88213" search. There is no such note, and the
index returns confident-looking results anyway. A vector index always returns
its *k* nearest neighbours; "nearest" does not mean "relevant". Without a score
threshold you have built a hallucination machine.

**In notebook 09**, the separation of retrieval evaluation from answer
evaluation. They fail for different reasons and have different fixes. If
recall@3 is low, no amount of prompt engineering will help — the model cannot
answer from text it never received.

**In notebook 10**, three things:
- the 30-line agent loop. Write it yourself before using the library version.
  Every framework you meet afterwards is that loop plus conveniences.
- the `stop=["Observation:"]` experiment. Remove it and watch the model invent
  its own tool results. It is the most instructive bug in the workshop.
- the prompt-injection demonstration. Whatever your model does on the day,
  **do not conclude you are safe.**

## The exercise that matters most

In notebook 09: *write 10 evaluation questions about your own documents, with
expected sources. Measure recall@3. Change one thing. Measure again.*

This single exercise teaches more than any amount of prompt tweaking by feel,
and it is the skill that separates people who can ship LLM systems from people
who can only demo them.

## Bring your own documents

Today is when the workshop stops being an exercise. Replace `data/notes/` with:

- your lecture notes or slides, converted to text or markdown;
- documentation for a project you work on;
- your own journal, reading notes, or recipe collection.

Anything you genuinely want to search. The privacy argument for open weights
stops being abstract the moment your real notes are in the index — nothing
leaves your machine.

{: .note }
> PDFs need converting first. `pip install pymupdf4llm` then
> `python -c "import pymupdf4llm; print(pymupdf4llm.to_markdown('slides.pdf'))" > slides.md`
> works well for text-based PDFs. Scanned pages need OCR.

## Checkpoint

1. What does an embedding model output, and how does it differ from a chat model?
2. Give three query types where keyword search beats embeddings.
3. What are the four stages of RAG?
4. What does recall@k measure, and what do you do when it is low?
5. Write the agent loop from memory, in four steps.
6. Name the five agent failure modes.
7. Give two defences against prompt injection that do **not** rely on the model
   being sensible.

## Optional reading

- [Agentic AI](../guides/agentic-ai.html) — the long-form guide. Read it tonight
  if you intend to build an agent for your capstone.
- [Yao et al., *ReAct*](https://arxiv.org/abs/2210.03629) — the original paper.
  Short and readable.
