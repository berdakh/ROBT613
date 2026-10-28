---
title: "Optional · Agents in depth"
parent: Daily plan
nav_order: 6
---

# The agents track — notebooks 17 to 20
{: .no_toc }

**Goal:** go from "I can write an agent loop" to "I can build one that
remembers, plans, delegates, and that I can prove works."

1. TOC
{:toc}

---

## Who this is for

Notebook 10 builds an agent from nothing and is complete in itself. What it
leaves you with is an agent that answers one question and forgets everything —
which is most of what an agent is, and not enough for anything you would use
twice.

This track adds the rest, and it adds the part nobody teaches: **how to tell
whether any of it works**. Take it if you are building something with an agent
in it, or if you want the vocabulary to evaluate the claims in an agent
framework's README.

**Skip it** if your goal is retrieval and structured output. Notebooks 01–12
are complete without this, and prompting, RAG and tools take you further than
agents in most products.

## What it costs you

| | |
|---|---|
| **Time** | About 6 hours, so one more day |
| **Hardware** | A laptop. **No server, no GPU, no model downloads.** |
| **Prerequisites** | Notebook 10. Notebook 07 helps for the tool material. |

That second row is unusual and deliberate. Every notebook in this track runs
against a **scripted model** — `qwen_workshop.scripted`, which replays a
written-down script in place of a real one. Agent logic is ordinary Python, it
deserves ordinary tests, and you cannot debug a loop whose failures do not
reproduce. The technique is the lesson as much as the convenience.

## Schedule

| Time | Activity | Notebook |
|---|---|---|
| 0:00–1:30 | Sandboxed tools; four memory strategies; the context budget | [`17_tools_and_memory`](../notebooks/17_tools_and_memory.html) |
| 1:30–3:00 | Plans as graphs; waves; failure, blocking and replanning | [`18_planning_and_decomposition`](../notebooks/18_planning_and_decomposition.html) |
| 3:00–4:30 | Routers, orchestrators, critic loops — and when not to | [`19_multi_agent_systems`](../notebooks/19_multi_agent_systems.html) |
| 4:30–6:00 | Outcome vs trajectory; lucky passes; a suite that runs in CI | [`20_evaluating_agents`](../notebooks/20_evaluating_agents.html) |

## The notebooks

### 17 · Tools and memory

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT406LLM/blob/master/notebooks/17_tools_and_memory.ipynb)

A triage agent for a messy downloads folder. Path containment that actually
holds, then four memory strategies measured against the same conversation.

**The idea you take away:** memory is not a model feature, it is your code
deciding what goes in the next request — which makes it a budget problem with
an arithmetic answer. Every strategy here drops a constraint the user stated
in turn three, except the one where somebody decided it was a rule.

<img src="../assets/diagrams/ag-memory-strategies.svg" alt="Four memory strategies and what each one loses" width="100%">

### 18 · Planning and decomposition

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT406LLM/blob/master/notebooks/18_planning_and_decomposition.ipynb)

A research agent over this workshop's own documentation. Plans as dependency
graphs, validated before they run, executed in waves, and recovered by
replanning when a step fails.

**The idea you take away:** a plan is an artifact you can inspect, price and
refuse *before* anything happens. That is where your guardrails want to be —
the notebook catches a `Delete everything` task in eleven lines, before it ran.

### 19 · Multi-agent systems

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT406LLM/blob/master/notebooks/19_multi_agent_systems.ipynb)

A trip-planning team: router, orchestrator, critic loop, and the handoff costs
of each, counted rather than assumed.

**The idea you take away:** most multi-agent systems should have been one agent
with more tools. There are exactly three good reasons for a team, one of which
— keeping tool permissions apart — nothing else can provide.

<img src="../assets/diagrams/ag-team-patterns.svg" alt="Router, orchestrator, pipeline and critic loop, with the cost of each" width="100%">

### 20 · Evaluating agents

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT406LLM/blob/master/notebooks/20_evaluating_agents.ipynb)

The notebook the rest of the track is for. Outcome versus trajectory, the
lucky pass, LLM-as-judge and its three traps, pass@k versus pass^k, and a
regression suite that runs on every commit without a GPU.

**The idea you take away:** an agent that only gets scored on its answers is an
agent you cannot trust. The suite in this notebook reports 100% outcome and 40%
trajectory on the same five cases — and one of the "passes" sent an email
nobody approved.

<img src="../assets/diagrams/ag-eval-quadrants.svg" alt="Outcome versus trajectory, and the lucky-pass quadrant" width="100%">

## If you only do one

**Notebook 20.** Memory, planning and teams are options you may never need.
Knowing whether your agent works is not optional, and almost nobody checks the
trajectory.

## What this track deliberately does not cover

* **A framework.** LangGraph, CrewAI and Qwen-Agent are each a variation on
  what you build here. Learn the mechanism and the frameworks become
  unremarkable, which is the point.
* **MCP in code.** [Chapter 6 of the handbook](../handbook/protocols-and-standards.html)
  covers what it is and why it exists; writing a server is a different
  afternoon.
* **Computer-use and browser agents.** Same loop, far more dangerous tools, and
  the safety story deserves more room than a section.
* **Agents that are actually reliable.** Nobody has those yet. What you can
  have is an agent whose failure modes you can name, reproduce and test for —
  and that is what this track gives you.
