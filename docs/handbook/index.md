---
title: Handbook
nav_order: 3
has_children: true
---

# The handbook
{: .no_toc }

Background reading. The notebooks teach you to *build*; this explains what you
are building on — where these models come from, how they are made, and why the
tools around them exist.

It is written to be read straight through in an evening, or dipped into. It
assumes nothing beyond the ability to read code, and it does not require you to
run anything.

---

## Why this exists

You can use an LLM productively without knowing how it was trained, in the same
way you can drive without knowing how an engine works. But the moment something
goes wrong — the model invents a citation, ignores your instruction, costs ten
times what you expected — the people who can diagnose it are the ones who know
what is happening underneath.

This handbook is that layer of understanding. It is also, deliberately, a map of
the **ecosystem**: there are hundreds of tools with confident names and similar
descriptions, and most of the confusion in this field comes from not knowing
which layer a tool sits in or which problem it was born to solve.

## The chapters

| | Chapter | What you get |
|:---|:---|:---|
| 1 | [How we got here](how-we-got-here.html) | Why each piece was invented, in order. The shortest useful history. |
| 2 | [How models work](how-models-work.html) | Tokens, embeddings, attention, context windows — the mechanism, without the maths degree. |
| 3 | [How LLMs are trained](how-llms-are-trained.html) | Every phase: data, pre-training, post-training, RLHF, DPO, reasoning, and what *you* can do. |
| 4 | [The model landscape](the-model-landscape.html) | Base vs instruct vs reasoning. Foundation vs frontier. How to read a model card. |
| 5 | [The tool ecosystem](the-tool-ecosystem.html) | Every tool worth knowing: what problem it solves, why it was built, when to use it. |
| 6 | [Protocols and standards](protocols-and-standards.html) | The OpenAI API as lingua franca, tool-calling formats, and MCP in depth. |

## How it relates to the workshop

The handbook is standalone — you can read it without touching a notebook. But
each chapter points at the notebook where you do the thing it describes.

| Handbook | Notebook |
|---|---|
| How models work | 01 (tokenization), 03 (generation) |
| How LLMs are trained | 11 (LoRA fine-tuning) |
| The model landscape | 02 (choosing a model) |
| The tool ecosystem | 05 (runtimes), 06 (serving) |
| Protocols and standards | 07 (tool calling), 10 (agents) |

## A note on dates and numbers

Specific figures — parameter counts, token budgets, training costs, release
dates — are a **snapshot**, and this field moves faster than any document can.
They are here to give you a sense of scale, not to be quoted.

The *reasoning* should outlast them. When a number here disagrees with a model
card, believe the model card.

{: .note }
> Anything asserted about a specific model should be checked against its own
> documentation. Anything asserted about *why* a technique exists is much more
> durable — that is what this handbook is really for.
