# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/18_planning_and_decomposition.ipynb)
#
# *Click the badge to run this notebook in Google Colab - no install required. The first code cell sets everything up.*

# %% [markdown]
# # 18 · Planning and decomposition
#
# **Agents track · ~90 minutes · no server needed**
#
# The agent in notebook 10 decides one step at a time. That is ReAct, it is
# flexible, and it has a specific blind spot: **nothing ever checks that the
# steps add up to the task.** A ReAct agent can take six sensible-looking
# steps and answer a question nobody asked.
#
# The alternative is to write the plan down first. By the end you will be able
# to:
#
# 1. say when plan-then-execute beats one-step-at-a-time, and when it loses;
# 2. model a plan as a dependency graph, and validate it before running it;
# 3. parse a plan out of real model output, including the malformed kind;
# 4. execute independent steps together and dependent steps in order;
# 5. distinguish **failed** from **blocked**, and recover by replanning;
# 6. use the plan as the place a human intervenes.
#
# **Everything runs with a scripted model.** The working example is a
# **research agent over this workshop's own documentation** — a question whose
# answer is genuinely spread across several files, which is exactly the shape
# that rewards planning.

# %%
# --- Setup: works on your laptop AND in Google Colab ------------------------
import os
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    if not Path("/content/ROBT613").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/ROBT613.git",
                        "/content/ROBT613"], check=True)
    os.chdir("/content/ROBT613")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], check=True)

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "src" / "qwen_workshop").is_dir())
sys.path.insert(0, str(REPO_ROOT / "src"))
print("repo root:", REPO_ROOT)

# %%
import json
import re
import textwrap

from qwen_workshop.planning import Plan, PlanError, Task, topological_order

# %% [markdown]
# ## 18.1 · What one-step-at-a-time cannot see
#
# ReAct chooses the next action from what it knows *now*. That is its strength
# and its limit:
#
# | | ReAct | Plan-then-execute |
# |---|---|---|
# | Decides | one step at a time | all steps, up front |
# | Adapts to findings | immediately | only at a replan |
# | Can run steps in parallel | no — it does not know what is next | yes, where independent |
# | Reviewable before it runs | no | **yes** |
# | Cost of a wrong turn | one step | the whole plan |
# | Knows if it is done | it decides it is | checkable against the plan |
#
# Row four is the one that matters in practice. A plan is an **artifact**: you
# can print it, price it, show it to a person, and refuse it — all before
# anything irreversible happens. With ReAct the first time you see the shape
# of the work is in the trace afterwards.
#
# Row six is the subtle one. A ReAct agent stops when the model feels finished,
# which is not the same as the task being complete, and there is nothing to
# compare against. With a plan, "did every task reach `done`?" is a property
# of a data structure.
#
# The cost is real: **a plan written before step 1 ran is a plan written in
# ignorance.** That is what replanning (18.6) is for, and why serious systems
# loop plan → execute → replan rather than picking a side.

# %% [markdown]
# ## 18.2 · A plan is a graph, not a list
#
# The thing people get wrong is modelling a plan as an ordered list. It is a
# **dependency graph**, and the difference shows up the moment anything fails
# or anything could run in parallel.

# %%
plan = Plan.from_dicts(
    "How does this workshop teach RAG, and what does it warn you about?",
    [
        {"id": "find_rag", "description": "Search the docs for the RAG pipeline"},
        {"id": "find_warn", "description": "Search the docs for RAG failure modes"},
        {"id": "read_rag", "description": "Read the best RAG hit", "depends_on": ["find_rag"]},
        {"id": "read_warn", "description": "Read the best warning hit", "depends_on": ["find_warn"]},
        {"id": "answer", "description": "Write the answer",
         "depends_on": ["read_rag", "read_warn"]},
    ],
)
print(plan.render())
print("\nwaves, straight from the graph:", topological_order(plan.tasks))

# %% [markdown]
# Five tasks, **three waves**. Wave 0 holds two searches that have nothing to
# do with each other, so they can run together. That number — the wave count —
# is the shortest the plan can possibly take, and you know it before paying
# for anything.
#
# ### Validate before you run
#
# A plan comes from a model, so it can be nonsense. Catch it here, loudly,
# rather than three steps in.

# %%
for rows, label in [
    ([{"id": "a", "description": "x", "depends_on": ["b"]},
      {"id": "b", "description": "y", "depends_on": ["a"]}], "a cycle"),
    ([{"id": "a", "description": "x", "depends_on": ["ghost"]}], "a dependency that does not exist"),
    ([], "no tasks at all"),
    ([{"id": "a"}], "a task with no description"),
]:
    try:
        Plan.from_dicts("goal", rows)
        print(f"  {label:<36} NOT CAUGHT")
    except PlanError as exc:
        print(f"  {label:<36} {exc}")

# %% [markdown]
# Each of those is a real thing models produce. The cycle in particular: ask
# for a plan where step 3 "verifies step 5" and you will get one.
#
# > **Exercise 1.** `topological_order` raises on a cycle. Write
# > `find_cycle(tasks)` that returns one offending cycle as a list of ids
# > (e.g. `["a", "b", "a"]`) so the error can name it, and use it to improve
# > the message. Then say why reporting *one* cycle is more useful to a model
# > trying to fix its plan than reporting all of them.

# %% [markdown]
# ## 18.3 · The domain: research over the workshop's own docs
#
# Two tools. Plain substring search — no embeddings, so this runs instantly
# and the results are easy to check by eye. Notebook 09 covers doing it
# properly; here the retrieval is not the lesson.

# %%
DOCS = sorted((REPO_ROOT / "docs").rglob("*.md"))
print(f"{len(DOCS)} markdown files")


def search_docs(query: str, limit: int = 4) -> list[dict]:
    """Find docs mentioning `query`, best first by number of mentions."""
    hits = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        count = text.lower().count(query.lower())
        if count:
            hits.append({"path": str(path.relative_to(REPO_ROOT)), "mentions": count})
    hits.sort(key=lambda h: -h["mentions"])
    return hits[:limit]


def read_doc(path: str, around: str = "", chars: int = 400) -> str:
    """Read a doc, centred on `around` if given."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    if around and (index := text.lower().find(around.lower())) != -1:
        start = max(0, index - chars // 3)
        return text[start:start + chars]
    return text[:chars]


for hit in search_docs("recall@"):
    print(" ", hit)

# %% [markdown]
# ## 18.4 · Getting a plan out of a model
#
# You ask for JSON. You get JSON wrapped in enthusiasm, usually in a code
# fence, sometimes with a trailing apology.

# %%
PLANNER_PROMPT = """Break this question into independent research tasks.

QUESTION: {question}

Reply with ONLY a JSON object:
{{"tasks": [{{"id": "short_id", "description": "what to do",
              "depends_on": ["ids this needs first"]}}]}}

Rules:
- Searches that look for different things do not depend on each other.
- A task that reads a result depends on the search that found it.
- The final task writes the answer and depends on everything it uses."""

# What a model actually sends back. Note the fence and the chatter.
MODEL_REPLY = """Sure! Here's how I'd break that down:

```json
{"tasks": [
  {"id": "s_rag",   "description": "Search docs for 'retrieval' to find the RAG pipeline"},
  {"id": "s_eval",  "description": "Search docs for 'recall@' to find how retrieval is measured"},
  {"id": "r_rag",   "description": "Read the top RAG hit", "depends_on": ["s_rag"]},
  {"id": "r_eval",  "description": "Read the top evaluation hit", "depends_on": ["s_eval"]},
  {"id": "answer",  "description": "Write the final answer",
   "depends_on": ["r_rag", "r_eval"]}
]}
```

Let me know if you'd like me to adjust the scope!"""

QUESTION = "How does this workshop do RAG, and how does it measure whether retrieval worked?"
plan = Plan.from_model_output(QUESTION, MODEL_REPLY)
print(plan.render())

# %% [markdown]
# `from_model_output` handles the fence and the chatter, accepts a bare list
# or a `{"tasks": ...}` wrapper, and **raises if there is no plan in there at
# all** — rather than returning an empty plan that silently does nothing,
# which is the failure that takes longest to notice.

# %%
try:
    Plan.from_model_output(QUESTION, "I'm not sure how to break that down, sorry.")
except PlanError as exc:
    print("refused to proceed:", str(exc).splitlines()[0])

# %% [markdown]
# ## 18.5 · Executing the plan
#
# `execute` repeatedly takes every task whose dependencies have **succeeded**,
# runs it, and records the result. The runner is yours; it gets the task and
# the results of its dependencies.

# %%
def runner(task: Task, inputs: dict[str, str]) -> str:
    """Does one task. Raise to fail it - a half-worked step must not look fine."""
    if task.id.startswith("s_"):
        # Take the query from the task description. Hardcoding it here would
        # mean a replanned task silently re-runs the original search, and the
        # recovery in 18.6 would look like it worked when it had not.
        match = re.search(r"'([^']+)'", task.description)
        if not match:
            raise ValueError(f"task {task.id!r} has no quoted search term")
        query = match.group(1)
        hits = search_docs(query)
        if not hits:
            raise LookupError(f"nothing found for {query!r}")
        return json.dumps(hits[:2])

    if task.id.startswith("r_"):
        hits = json.loads(next(iter(inputs.values())))
        return read_doc(hits[0]["path"], around="recall", chars=240)

    # The final task: a real system asks the model here. We join, so the
    # output is reproducible and the plumbing is what you are looking at.
    return " ".join(f"[{k}] {v[:90]}…" for k, v in inputs.items())


plan.execute(runner, on_task=lambda t: print(f"  {t.status:<7} {t.id}"))
print("\ncomplete:", plan.complete, "| failed:", [t.id for t in plan.failed])
print("\nANSWER:\n" + textwrap.fill(plan.by_id["answer"].result, 76))

# %% [markdown]
# Look at the order those printed in: `s_rag` and `s_eval` both ran before
# either `r_` task, because they were in the same wave. A ReAct agent cannot
# do that — it does not know the second search exists until it has finished
# thinking about the first.
#
# > **Exercise 2.** `execute` runs a wave sequentially. Replace the inner loop
# > with a `concurrent.futures.ThreadPoolExecutor` so one wave runs in
# > parallel, and time both on this plan. Then name the thing that becomes
# > unsafe the moment tasks run concurrently, given that `runner` here touches
# > the filesystem — and what you would change about the runner contract.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT613/master/docs/assets/diagrams/ag-plan-graph.svg" alt="A plan as a dependency graph in three waves; one failed search blocks its descendants while the parallel branch completes" width="100%">
#
# ## 18.6 · When a step fails
#
# This is where the dependency graph earns its place. Watch what happens when
# one search finds nothing.

# %%
broken = Plan.from_model_output(QUESTION, MODEL_REPLY)


def flaky_runner(task: Task, inputs: dict[str, str]) -> str:
    if task.id == "s_eval":
        raise LookupError("search index unavailable")
    return runner(task, inputs)


broken.execute(flaky_runner)
print(broken.render())
print("\nfailed :", [t.id for t in broken.failed])
print("blocked:", [t.id for t in broken.blocked])
print("kept   :", list(broken.results()))

# %% [markdown]
# Three things to notice, and each is a bug if you get it wrong:
#
# 1. **`s_rag` and `r_rag` still ran.** They had nothing to do with the
#    failure. Abandoning the whole run would have thrown away good work.
# 2. **`r_eval` and `answer` are `blocked`, not `failed`.** They did not fail —
#    they never ran. Conflating the two makes a trace unreadable and sends you
#    debugging the wrong task.
# 3. **The partial results survive.** Whatever happens next starts from here,
#    not from scratch.
#
# ### Replanning
#
# Now ask the planner for a way around the failure. The completed work is kept.

# %%
def replanner(current: Plan, stuck: list[Task]) -> list[dict]:
    """Stands in for a model call. A real one is given the plan, the failures
    and the error messages, and asked for replacement tasks."""
    print("  stuck:", [(t.id, t.error) for t in stuck])
    return [
        {"id": "s_eval2", "description": "Search for 'evaluation' instead"},
        {"id": "r_eval2", "description": "Read the top hit", "depends_on": ["s_eval2"]},
        {"id": "answer2", "description": "Write the final answer",
         "depends_on": ["r_rag", "r_eval2"]},
    ]


broken.replan_after_failure(replanner)
print("\nafter replanning:")
print(broken.render())
broken.execute(runner)
print("\ncomplete:", broken.complete, "| replans:", broken.replans)

# %% [markdown]
# `r_rag` kept its result and was never re-run — that is the point of
# replanning rather than restarting. A system that starts over on every
# failure is slower and, with a flaky dependency, can loop forever.
#
# The guard rail is worth seeing. A replacement task may depend on **completed
# work**, never on the wreckage it is replacing:

# %%
doomed = Plan.from_dicts("g", [{"id": "a", "description": "the one that fails"}])
doomed.execute(lambda t, i: (_ for _ in ()).throw(OSError("nope")))
try:
    doomed.replan_after_failure(lambda p, stuck: [
        {"id": "b", "description": "depends on the failure", "depends_on": ["a"]}])
except PlanError as exc:
    print("refused:", exc)

# %% [markdown]
# And **giving up is a legitimate replan.** A planner that returns `[]` is
# saying "this cannot be done with these tools", which is a far better answer
# than inventing a step.

# %%
doomed.replan_after_failure(lambda p, stuck: [])
print("tasks remaining:", doomed.tasks, "| complete:", doomed.complete)

# %% [markdown]
# > **Exercise 3.** Add a `max_replans` cap to the plan–execute–replan loop and
# > write the loop itself: execute, and while there are failures and replans
# > remain, replan and execute again. Test it with a runner that fails the
# > *first* two times a task runs and then succeeds. What should happen when
# > the cap is hit — and why is "keep going" the wrong answer even though the
# > task would eventually succeed?

# %% [markdown]
# ## 18.7 · The honest comparison
#
# Planning is not better. It is a different trade, and the literature
# overstates it because plans look impressive in a paper.
#
# **Plan-then-execute wins when:**
#
# * the subtasks are genuinely independent (parallelism is free speedup);
# * someone needs to approve the work before it happens;
# * the task is long enough that a wrong turn is expensive;
# * you need to report progress — a plan has a denominator.
#
# **ReAct wins when:**
#
# * what to do next genuinely depends on what you just found;
# * the task is two or three steps, so planning overhead dominates;
# * the environment is unpredictable and plans go stale immediately.
#
# **Neither helps when the model cannot decompose the problem.** A bad plan
# executed perfectly is still a bad answer, and planning *adds* a place to
# fail. If your 0.6B model writes a four-task plan where three tasks are the
# same search, the problem is not the executor.
#
# A useful instinct: **the plan should be shorter than the work.** If writing
# the plan costs as much as doing the task, do the task.

# %%
print(f"{'':<22}{'model calls':>13}{'can parallelise':>18}{'reviewable':>12}")
for label, model_calls, parallel, reviewable in [
    ("ReAct, 5 steps", "5", "no", "no"),
    ("Plan, 5 tasks", "1 + 5", "yes", "yes"),
    ("Plan + 1 replan", "2 + 8", "yes", "yes"),
]:
    print(f"{label:<22}{model_calls:>13}{parallel:>18}{reviewable:>12}")

# %% [markdown]
# Planning costs an extra model call up front and buys parallelism and
# reviewability. On five independent searches that is a clear win. On a
# two-step task it is pure overhead.

# %% [markdown]
# ## 18.8 · The plan is where a human belongs
#
# This is the practical reason to plan, and it is underrated: **a plan is a
# cheap review point.** Before it runs, nothing has happened yet.

# %%
def approve(plan: Plan, *, allow: set[str], max_tasks: int = 8) -> list[str]:
    """Check a plan before any of it runs. Returns the reasons to refuse."""
    problems = []
    if len(plan.tasks) > max_tasks:
        problems.append(f"{len(plan.tasks)} tasks exceeds the limit of {max_tasks}")
    for task in plan.tasks:
        verb = task.description.split()[0].lower()
        if verb not in allow:
            problems.append(f"task {task.id!r} starts with {verb!r}, which is not allowed")
    return problems


READ_ONLY = {"search", "read", "write", "summarise", "compare"}
print("research plan :", approve(plan, allow=READ_ONLY) or "approved")

dangerous = Plan.from_dicts("tidy up", [
    {"id": "a", "description": "Search for old files"},
    {"id": "b", "description": "Delete everything older than a year", "depends_on": ["a"]},
])
print("tidy-up plan  :", approve(dangerous, allow=READ_ONLY))

# %% [markdown]
# That check is eleven lines and it caught a `Delete` task **before it ran**.
# With ReAct the equivalent check has to happen at every step, under time
# pressure, with no view of what comes next — and by the time you see the
# delete, the agent is already asking to do it.
#
# The same point in one sentence: **planning moves your guardrails earlier,
# where they are cheaper and see more.**
#
# > **Exercise 4.** `approve` checks the first word of the description, which
# > is trivially defeated by "Carefully delete everything". Rewrite it to
# > check the *tools* each task would use instead of its prose — which means
# > changing the plan format so each task declares its tool. Then say which
# > property of 10.1 that change strengthens, and what flexibility it costs.
#
# ## Checkpoint
#
# 1. What can a plan check that ReAct structurally cannot?
# 2. Why is a plan a graph rather than a list? Give a consequence of each.
# 3. What does the wave count tell you before you run anything?
# 4. Name three invalid plans a model will actually produce.
# 5. Distinguish `failed` from `blocked`, and say why conflating them hurts.
# 6. Why does replanning keep completed tasks?
# 7. Give two situations where ReAct is the better choice.
# 8. Why is a plan a better place for a human check than a step?
#
# ➡️ **Next:** [`19_multi_agent_systems.ipynb`](19_multi_agent_systems.ipynb) —
# several agents, and whether you needed them.
