# %% [markdown]
# # 12 · Capstone — build your daily-life assistant
#
# **Day 4 · the rest of the workshop**
#
# Everything so far has been preparation. Now you build one thing, end to end,
# that you would actually use.
#
# **The brief:** an assistant that helps with one real, recurring task in your
# own life, running entirely on your own machine.
#
# This notebook gives you a working reference implementation, then hands you
# the scaffolding to build your own.

# %% [markdown]
# ## 12.1 · The requirements
#
# Your capstone must include **all four**:
#
# | Requirement | Where you learned it | Evidence needed |
# |---|---|---|
# | 1. A local open-weight model | 03, 05, 06 | it runs with no API key |
# | 2. At least 3 tools, or RAG over your own documents | 07, 08, 09 | tool schemas or an index |
# | 3. An agent loop or an explicit chain | 10 | a trace of a multi-step run |
# | 4. An honest evaluation | 09 | ≥10 test cases, with a score |
#
# Plus a short **README** covering: what it does, how to run it, what it gets
# wrong, and what you would do with another week.
#
# ### Judged on
#
# - **It works.** A demo that runs beats a design that does not.
# - **You measured it.** A number, however humble, beats "it seems good".
# - **You know its limits.** Naming a failure honestly scores higher than
#   pretending it does not exist.
# - **Sensible engineering.** Safe tools, a step limit, no `eval`.
#
# Not judged on: model size, framework choice, or UI polish.

# %% [markdown]
# ## 12.2 · Project ideas
#
# Pick one, or bring your own. The best capstone is one that scratches your own
# itch — you will test it far more honestly.
#
# | Project | Tools it needs | Difficulty |
# |---|---|---|
# | **Kitchen assistant** — what can I cook tonight? | pantry, expiry dates, recipe notes | ⭐⭐ |
# | **Budget coach** — where does my money go? | CSV analysis, arithmetic, notes | ⭐⭐ |
# | **Inbox triage** — what needs my attention? | read inbox, classify, draft replies | ⭐⭐⭐ |
# | **Study buddy** — quiz me on my lecture notes | RAG over notes, question generation, scoring | ⭐⭐⭐ |
# | **Trip planner** — plan a weekend away | notes, budget, dates, checklist | ⭐⭐⭐ |
# | **Lab notebook assistant** — search and summarise your experiments | RAG, plotting, structured extraction | ⭐⭐⭐⭐ |
# | **Code reviewer** — critique a diff against your team's conventions | file reading, Qwen2.5-Coder | ⭐⭐⭐⭐ |
#
# 📖 More detail on each: [`docs/guides/capstone-ideas.md`](../docs/guides/capstone-ideas.md)

# %% [markdown]
# ## 12.3 · Reference implementation: the kitchen + budget assistant
#
# Read this, run it, then replace it with yours.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_workshop.agent import Agent
from qwen_workshop.client import get_client, require_backend
from qwen_workshop.demo_tools import daily_life_registry

BACKEND, MODEL = "ollama", "qwen3:0.6b"   # try qwen3:4b if you have the RAM
require_backend(BACKEND)
client, backend = get_client(BACKEND, model=MODEL)

ASSISTANT_SYSTEM = """You are a practical daily-life assistant for a student in Kazakhstan.

You have tools for the user's pantry, expenses, notes and inbox. Rules:
- ALWAYS use a tool for facts about the user's own life. Never guess an amount,
  a date or an item.
- Use `current_date` before any reasoning about "today", "soon" or "this month".
- Use `calculate` for arithmetic. Do not do sums in your head.
- Prices are in tenge (₸). Write them with a thousands separator.
- Answer in at most 6 lines. Be concrete and actionable.
- If the tools cannot answer, say so plainly instead of inventing something."""

tools = daily_life_registry()
assistant = Agent(client, backend, tools, system=ASSISTANT_SYSTEM, max_steps=6)

print(f"{len(tools)} tools ready:")
print(tools.describe())

# %%
run = assistant.run("What should I cook tonight? Use what needs eating first.")
print(run.trace())
print("\n" + "=" * 72)
print(run.answer)

# %%
run = assistant.run("Am I overspending? Compare my January total with my budget target.")
print(run.trace())
print("\n" + "=" * 72)
print(run.answer)

# %% [markdown]
# ## 12.4 · Add a tool of your own
#
# The fastest way to make the assistant yours. Here is a worked example:
# a shopping list built from the gap between recipes and the pantry.

# %%
from qwen_workshop.tools import ToolError, tool


@tool
def suggest_shopping_list(meals: int = 3, budget_kzt: int = 8000) -> dict:
    """Suggest what to buy to cook a number of meals within a budget.

    Args:
        meals: How many meals to plan for.
        budget_kzt: Maximum to spend, in tenge.
    """
    if meals < 1 or meals > 14:
        raise ToolError("meals must be between 1 and 14")
    if budget_kzt < 500:
        raise ToolError("budget must be at least 500 KZT")

    from qwen_workshop.demo_tools import list_pantry

    have = {item["item"].split("(")[0].strip().lower() for item in list_pantry()}
    # A deliberately simple staples model - your version should be better.
    staples = {
        "onions": 400, "carrots": 350, "potatoes": 600, "eggs": 1200,
        "milk": 700, "chicken thighs": 2800, "rice (basmati)": 1500,
        "red lentils": 900, "tomatoes (tinned)": 1400, "bread": 350,
        "yoghurt": 800, "cheddar": 1900,
    }

    missing, spent = [], 0
    for item, price in sorted(staples.items(), key=lambda kv: kv[1]):
        if item.split("(")[0].strip().lower() in have:
            continue
        if spent + price > budget_kzt:
            continue
        missing.append({"item": item, "approx_kzt": price})
        spent += price

    return {
        "meals_planned": meals,
        "buy": missing,
        "estimated_total_kzt": spent,
        "budget_kzt": budget_kzt,
        "note": "Prices are rough estimates, not live data.",
    }


tools.add(suggest_shopping_list)
assistant = Agent(client, backend, tools, system=ASSISTANT_SYSTEM, max_steps=6)

run = assistant.run("Plan 3 meals and tell me what to buy for under 6000 tenge.")
print(run.trace())
print("\n" + "=" * 72)
print(run.answer)

# %% [markdown]
# > **Exercise 1.** Notice what `suggest_shopping_list` does *badly*: the
# > prices are hardcoded, it ignores `meals` entirely, and it has no idea what
# > recipes are possible. Improve one of those three. That is your first real
# > contribution to the system.

# %% [markdown]
# ## 12.5 · Evaluate it — this is the part people skip
#
# Write your test cases **before** you tune anything. Otherwise you will tune
# until the last example you happened to try looks good.

# %%
from dataclasses import dataclass


@dataclass
class TestCase:
    question: str
    must_use_tools: list[str]      # tools that must appear in the trace
    must_mention: list[str]        # strings the answer must contain
    must_not_mention: list[str]    # e.g. invented numbers


TEST_CASES = [
    TestCase("What is expiring in my kitchen soon?", ["expiring_soon"], ["milk"], []),
    TestCase("How much did I spend in January 2026?", ["spending_by_category"], ["154"], []),
    TestCase("What is my budget target?", ["search_notes", "read_note"], ["180"], []),
    TestCase("What is my biggest single expense?", ["largest_expenses"], [], []),
    TestCase("What day is it today?", ["current_date"], [], []),
    TestCase("How many days until 2026-12-31?", ["days_between", "current_date"], [], []),
    TestCase("What is in my pantry?", ["list_pantry"], ["rice"], []),
    TestCase("Do I have any email about the midterm?", ["read_inbox"], ["midterm"], []),
    TestCase("What is 15% of 154000?", ["calculate"], ["23100", "23,100"], []),
    TestCase("What is the population of Mars?", [], [], ["billion", "million"]),
]


def evaluate(agent, cases: list[TestCase], verbose: bool = True) -> dict:
    """Score an agent on tool use and answer content."""
    results = {"tool_use": 0, "content": 0, "clean": 0, "total": len(cases)}

    for case in cases:
        run = agent.run(case.question)
        used = {r.name for step in run.steps for r in step.tool_results}
        answer = run.answer.lower()

        tool_ok = (not case.must_use_tools) or bool(used & set(case.must_use_tools))
        content_ok = (not case.must_mention) or any(
            m.lower() in answer for m in case.must_mention
        )
        clean_ok = not any(m.lower() in answer for m in case.must_not_mention)

        results["tool_use"] += tool_ok
        results["content"] += content_ok
        results["clean"] += clean_ok

        if verbose:
            marks = f"{'T' if tool_ok else '-'}{'C' if content_ok else '-'}{'S' if clean_ok else '-'}"
            print(f"  [{marks}] {case.question[:48]:<50} used: {','.join(sorted(used)) or 'none'}")

    print(f"\n  tool use : {results['tool_use']}/{results['total']}")
    print(f"  content  : {results['content']}/{results['total']}")
    print(f"  clean    : {results['clean']}/{results['total']}")
    return results


print("Evaluating (this makes ~10 agent runs, so it takes a minute)...\n")
baseline_scores = evaluate(assistant, TEST_CASES)

# %% [markdown]
# **Record these numbers.** Now change one thing — the system prompt, the
# model, `max_steps`, a tool description — and re-run. Report the delta in your
# README. That before/after pair is the single most valuable artefact of your
# capstone.

# %%
# Example: does a stricter system prompt improve tool use?
STRICTER_SYSTEM = ASSISTANT_SYSTEM + (
    "\n\nCRITICAL: You must call at least one tool before answering any question "
    "about the user's pantry, spending, notes, calendar or email. If you answer "
    "such a question without calling a tool, your answer is wrong by definition."
)

strict_assistant = Agent(client, backend, tools, system=STRICTER_SYSTEM, max_steps=6)
print("With a stricter system prompt:\n")
strict_scores = evaluate(strict_assistant, TEST_CASES)

print(f"\ntool use: {baseline_scores['tool_use']} -> {strict_scores['tool_use']}")

# %% [markdown]
# ## 12.6 · A user interface (optional, 20 minutes)
#
# A UI is not required, but it makes your demo far more convincing and takes
# almost no time with Gradio.
#
# ```bash
# pip install gradio
# ```

# %%
def launch_ui(agent, share: bool = False):
    """Launch a chat UI for the agent. Call launch_ui(assistant) to try it."""
    import gradio as gr

    def respond(message, history):
        run = agent.run(message)
        used = {r.name for step in run.steps for r in step.tool_results}
        footer = f"\n\n---\n*tools used: {', '.join(sorted(used)) or 'none'} · " \
                 f"{len(run.steps)} step(s)*"
        return run.answer + footer

    return gr.ChatInterface(
        fn=respond,
        title="Daily Life Assistant",
        description="Running entirely on your machine with Qwen3. Nothing leaves this computer.",
        examples=[
            "What should I cook tonight?",
            "Am I overspending this month?",
            "What is expiring soon?",
            "Do I have any urgent email?",
        ],
    ).launch(share=share, inbrowser=False)


print("Run  launch_ui(assistant)  in a new cell to start the web UI.")
print("Showing the tools used in the footer is not decoration - it is how a")
print("user knows whether the answer was grounded or invented.")

# %% [markdown]
# ## 12.7 · Your turn — the scaffolding
#
# Fill this in. Delete the reference implementation above if it helps you think.

# %%
# ============================================================================
# MY CAPSTONE
# ============================================================================
# Project name : ...........................................................
# What it does : ...........................................................
# Who it is for: ...........................................................
# ============================================================================

MY_SYSTEM_PROMPT = """TODO: describe the assistant's role, its rules, and what
it must never do. Be specific about when it must use a tool."""


# --- My tools -------------------------------------------------------------
# At least three. Each one: validated arguments, a clear docstring, a
# ToolError for expected failures. No eval, no shell, no unbounded file access.

@tool
def my_first_tool(argument: str) -> str:
    """TODO: one line the model can understand.

    Args:
        argument: TODO what this is and what form it takes.
    """
    raise NotImplementedError("build me")


# --- My agent -------------------------------------------------------------
# from qwen_workshop.tools import ToolRegistry
# my_tools = ToolRegistry().add(my_first_tool, ...)
# my_agent = Agent(client, backend, my_tools, system=MY_SYSTEM_PROMPT, max_steps=6)


# --- My evaluation --------------------------------------------------------
MY_TEST_CASES = [
    # TestCase("a question a user would really ask", ["expected_tool"], ["expected text"], []),
    # ... at least 10, covering: the happy path, an edge case, and something
    # the assistant SHOULD refuse.
]

# scores = evaluate(my_agent, MY_TEST_CASES)

print("Scaffolding ready. Start with the tools - they define what is possible.")

# %% [markdown]
# ## 12.8 · Write your README
#
# Copy this into `capstone/README.md` in your own repository:
#
# ```markdown
# # <Project name>
#
# ## What it does
# One paragraph. What problem, for whom.
#
# ## Demo
# A screenshot or a transcript of a real run.
#
# ## How to run it
# Exact commands, starting from a fresh clone. Test them on a classmate's laptop.
#
# ## Architecture
# Which model, which tools, chain or agent, and why.
#
# ## Evaluation
# The test cases, the scores, and the before/after of one change you made.
#
# ## What it gets wrong
# Be specific and honest. Name three real failure modes you observed.
#
# ## What I would do next
# The thing you ran out of time for.
# ```
#
# ## 12.9 · Presenting
#
# Five minutes. Suggested shape:
#
# 1. **The problem** (30s) — something a human in the room recognises.
# 2. **Live demo** (2m) — run it. Have a recording as a backup.
# 3. **How it works** (1m) — one diagram: model, tools, loop.
# 4. **Numbers** (1m) — your evaluation, including what failed.
# 5. **Honest limits** (30s) — what you would not trust it with.
#
# The fifth point is the one that distinguishes an engineer from a demo.

# %% [markdown]
# ## Where to go after this workshop
#
# You now have the foundations. The natural next steps:
#
# - **Multimodal** — Qwen3-VL reads images, screenshots and documents. Your
#   receipt parser could take photographs instead of text.
# - **MCP** — package your tools as a Model Context Protocol server and use
#   them from any client.
# - **Better retrieval** — rerankers, hybrid search, query rewriting at scale.
# - **Evaluation** — LLM-as-judge, adversarial test sets, regression suites.
# - **Efficiency** — speculative decoding, KV-cache quantisation, batching.
# - **Read the model cards and technical reports.** They are more accurate and
#   more current than any tutorial, including this one.
#
# 📖 [`docs/guides/where-next.md`](../docs/guides/where-next.md)
#
# ---
#
# **You can now take an open-weight model, run it on your own hardware, give it
# tools and your own documents, wrap it in an agent, measure whether it works,
# and say honestly where it does not.**
#
# That is the whole job. Go build something.
