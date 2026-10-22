# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/17_tools_and_memory.ipynb)
#
# *Click the badge to run this notebook in Google Colab - no install required. The first code cell sets everything up.*

# %% [markdown]
# # 17 · Tools and memory
#
# **Agents track · ~90 minutes · no server, no GPU, no model**
#
# Notebook 10 built an agent that answers one question and forgets everything.
# That is most of what an agent is, and it is not enough for anything you
# would use twice.
#
# This notebook adds the two things that turn a loop into an assistant:
# **tools that can safely touch the real world**, and **memory**, which is a
# budget problem dressed up as a cognitive one.
#
# By the end you will be able to:
#
# 1. build a sandboxed tool set that cannot escape its directory;
# 2. explain why "memory" is your code's job and never the model's;
# 3. choose between four memory strategies, and say what each one loses;
# 4. measure the context budget instead of guessing at it;
# 5. use a scratchpad to stop an agent repeating a failed call;
# 6. expose memory *as tools*, and recognise what goes wrong when you do.
#
# **Everything here runs with a scripted model.** No server required.
#
# The working example is new: a **triage agent for a messy downloads folder**.
# It is deliberately not the pantry — the agent loop does not care about the
# domain, and seeing that is part of the point.

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
import shutil
import tempfile

from qwen_workshop.agent import Agent
from qwen_workshop.scripted import ScriptedModel, calls, says
from qwen_workshop.tools import ToolError, ToolRegistry, tool

# %% [markdown]
# ## 17.1 · Memory is your problem, not the model's
#
# A model has **no memory**. Every call is stateless. The only thing it knows
# is what is in the messages you send, and when the call returns, it is gone.
#
# So "the agent remembers" always means **your code put something in the next
# request**. That is worth saying flatly, because it converts a vague
# aspiration into an engineering decision with a hard constraint:
#
# > The context window is finite, and you pay for every token in it twice —
# > once in latency, once in money.
#
# Which makes memory a **budget problem**. Four strategies, and the right
# answer is usually the first one:
#
# | Strategy | Keeps | Loses | Survives restart |
# |---|---|---|---|
# | `ConversationBuffer` | everything | nothing, until it overflows | no |
# | `WindowedMemory` | the last N tokens | the beginning, silently | no |
# | `SummarisingMemory` | a summary + recent | detail, chosen by the summariser | no |
# | `FactStore` | what you chose | everything you did not choose | **yes** |
#
# The last column is why the fourth row is a different kind of thing. The
# first three are ways to fit a conversation into a window. Only `FactStore`
# is memory in the sense people mean when they say an assistant "knows them".

# %% [markdown]
# ## 17.2 · The domain: a downloads folder nobody has tidied
#
# Let's build something to be an agent *about*. Fifteen files, the usual mess.

# %%
SANDBOX = Path(tempfile.mkdtemp(prefix="triage-"))

FILES = {
    "invoice_2026_03_acme.pdf": "INVOICE\nAcme Ltd\nAmount due: 1,240.00\nDue: 2026-04-01",
    "invoice_2026_02_acme.pdf": "INVOICE\nAcme Ltd\nAmount due: 980.00\nPaid 2026-03-02",
    "lecture_notes_week3.md": "# Week 3\nAttention is a weighted average.\nSee also week 2.",
    "lecture_notes_week4.md": "# Week 4\nResidual streams and layer norm.",
    "tax_return_2025_FINAL_v3_really_final.docx": "TAX RETURN 2025\nSubmitted 2026-01-30",
    "dataset_sample.csv": "id,label\n1,cat\n2,dog\n3,cat",
    "notes.txt": "buy milk\ncall the landlord\nrenew passport",
    "paper_attention_is_all_you_need.pdf": "Attention Is All You Need. Vaswani et al. 2017.",
    "contract_lease_signed.pdf": "LEASE AGREEMENT\nTerm: 12 months from 2026-01-01",
    "budget_2026.xlsx": "monthly target: 250000",
}
# These are genuinely not text. `peek_file` must fail on them for real, not
# because we told it to - a fake failure teaches a fake lesson.
BINARY = {
    "IMG_4417.jpg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x02\xff\xdb",
    "IMG_4418.jpg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x02\xff\xdb",
    "Screenshot 2026-03-14 at 11.02.31.png": b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe",
    "setup-x86_64.exe": b"MZ\x90\x00\x03\x00\x00\x00\xff\xfe\xfd\xfc",
    "random_download.zip": b"PK\x03\x04\x14\x00\x00\x00\x08\x00\xff\xfe",
}
for name, body in FILES.items():
    (SANDBOX / name).write_text(body, encoding="utf-8")
for name, blob in BINARY.items():
    (SANDBOX / name).write_bytes(blob)

for folder in ("_sorted/finance", "_sorted/study", "_sorted/photos"):
    (SANDBOX / folder).mkdir(parents=True, exist_ok=True)

print(f"sandbox: {SANDBOX}")
print(f"{len(FILES) + len(BINARY)} files to triage "
      f"({len(BINARY)} of them not text)")

# %% [markdown]
# ## 17.3 · Tools that cannot escape
#
# The agent is about to be given a tool that **moves files**. That is the
# first genuinely dangerous tool in this workshop, so before anything else:
# containment.
#
# The rule is simple and the implementation is three lines, and almost every
# path-traversal bug is a missing version of those three lines.

# %%
def safe_path(name: str) -> Path:
    """Resolve `name` inside SANDBOX, or refuse.

    `resolve()` collapses `..` and follows symlinks *before* the check, which
    is the whole trick. Checking the string for ".." instead is the classic
    broken version: it misses symlinks, absolute paths and url-encoding.
    """
    candidate = (SANDBOX / name).resolve()
    if not candidate.is_relative_to(SANDBOX.resolve()):
        raise ToolError(
            f"refused: {name!r} resolves outside the sandbox. "
            "Pass a plain file name, not a path."
        )
    return candidate


for attempt in ["notes.txt", "../../../etc/passwd", "/etc/passwd",
                "_sorted/finance/x.pdf", "a/../../../../tmp/evil"]:
    try:
        safe_path(attempt)
        print(f"  allowed  {attempt}")
    except ToolError as exc:
        print(f"  REFUSED  {attempt:<28} {str(exc)[:46]}")

# %% [markdown]
# Note that `_sorted/finance/x.pdf` is allowed — it is inside. Containment is
# not "no slashes", it is "resolves under the root". Getting that distinction
# wrong produces a tool that is simultaneously too strict and still exploitable.
#
# Now the tools, every one of them going through `safe_path`.

# %%
@tool
def list_downloads(pattern: str = "") -> list[dict]:
    """List files waiting to be triaged in the downloads folder.

    Use this first, before any other tool, to see what is there.

    Args:
        pattern: Optional case-insensitive substring to filter names by.
            Leave empty for everything.
    """
    rows = []
    for path in sorted(SANDBOX.iterdir()):
        if path.is_dir() or path.name.startswith("_"):
            continue
        if pattern and pattern.lower() not in path.name.lower():
            continue
        rows.append({"name": path.name, "bytes": path.stat().st_size,
                     "suffix": path.suffix or "(none)"})
    return rows


@tool
def peek_file(name: str, max_chars: int = 200) -> str:
    """Read the first part of a text file to work out what it is.

    Use this when the file name is not enough to classify it.

    Args:
        name: File name exactly as `list_downloads` reported it.
        max_chars: How much to read. Keep small; you only need the gist.
    """
    path = safe_path(name)
    if not path.is_file():
        raise ToolError(f"no such file: {name!r}. Call list_downloads to see what exists.")
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except UnicodeDecodeError:
        raise ToolError(f"{name!r} is not a text file - classify it from its name and suffix.")


@tool
def file_to(name: str, category: str) -> str:
    """Move a file into one of the sorted folders. This changes the disk.

    Args:
        name: File name exactly as `list_downloads` reported it.
        category: One of: finance, study, photos.
    """
    allowed = {"finance", "study", "photos"}
    if category not in allowed:
        raise ToolError(f"unknown category {category!r}. Valid: {', '.join(sorted(allowed))}")
    source = safe_path(name)
    if not source.is_file():
        raise ToolError(f"no such file: {name!r}")
    target = safe_path(f"_sorted/{category}/{source.name}")
    shutil.move(str(source), str(target))
    return f"moved {source.name} -> _sorted/{category}/"


triage_tools = ToolRegistry().add(list_downloads, peek_file, file_to)
print(triage_tools.describe())

# %% [markdown]
# Three tools, one intent each, descriptions that say *when* to use them, and
# errors that tell the model how to retry. That is 10.5 applied.
#
# Let's watch it work. The script is what a competent model would decide.

# %%
model = ScriptedModel([
    calls("list_downloads", {"pattern": "invoice"}),
    calls("file_to", {"name": "invoice_2026_03_acme.pdf", "category": "finance"},
          ("file_to", {"name": "invoice_2026_02_acme.pdf", "category": "finance"})),
    says("Filed both Acme invoices under finance."),
])
print(Agent(*model.as_client(), triage_tools).run("File my invoices.").trace())

# %% [markdown]
# > **Exercise 1.** Script a run where the model tries
# > `file_to(name="../../../etc/passwd", category="finance")` after reading a
# > file whose contents tell it to. Confirm the tool refuses and that the
# > refusal reaches the model as an observation rather than killing the run.
# > Then answer: if `safe_path` did not exist, which *other* guardrail from
# > notebook 10.10 would have stopped this, and which would not?

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT613/master/docs/assets/diagrams/ag-memory-strategies.svg" alt="Four memory strategies as rows: keep everything, keep the newest, compress the old, and write it down; with what each one loses" width="100%">
#
# ## 17.4 · Four memory strategies, measured
#
# Now the conversation. Our user triages in sessions, coming back repeatedly,
# and each exchange costs context.

# %%
from qwen_workshop.memory import (
    ConversationBuffer,
    FactStore,
    Scratchpad,
    SummarisingMemory,
    WindowedMemory,
    estimate_tokens,
    format_messages,
)

# Note where the constraint is: turn 3. People state their rules early, in
# the same breath as the task - which is exactly what a sliding window drops.
SESSION = [
    ("user", "I need to tidy my downloads folder. Finance first, please."),
    ("assistant", "Filed invoice_2026_03_acme.pdf and invoice_2026_02_acme.pdf under finance."),
    ("user", "One rule: never move the .exe or the .zip. I'll check those myself."),
    ("assistant", "Understood - setup-x86_64.exe and random_download.zip stay where they are."),
    ("user", "Good. Also the lease."),
    ("assistant", "Filed contract_lease_signed.pdf under finance."),
    ("user", "My lecture notes should go to study."),
    ("assistant", "Filed lecture_notes_week3.md and lecture_notes_week4.md under study."),
    ("user", "Photos next."),
    ("assistant", "Filed IMG_4417.jpg, IMG_4418.jpg and the screenshot under photos."),
    ("user", "What's left?"),
    ("assistant", "notes.txt, dataset_sample.csv, the tax return, the paper and budget_2026.xlsx."),
]

SYSTEM = "You are a file triage assistant."

buffer = ConversationBuffer(system=SYSTEM)
for role, content in SESSION:
    buffer.add(role, content)

print(f"full conversation: {len(buffer)} messages, ~{buffer.token_count()} tokens")

# %% [markdown]
# ### Strategy 1: keep everything
#
# The right default, and people skip past it looking for something cleverer.
# Most assistants never reach their window. **Add complexity when you measure
# a problem, not before.**
#
# ### Strategy 2: a sliding window

# %%
window = WindowedMemory(system=SYSTEM, budget=120)
for role, content in SESSION:
    window.add(role, content)

rendered = window.render()
print(format_messages(rendered))
print(f"\nkept {len(rendered) - 1} of {len(SESSION)} messages, dropped {window.dropped}")

# %% [markdown]
# Cheap, bounded, and look at **what it dropped**: the user's original
# instruction, and the line saying to leave the `.exe` alone. The agent will
# now happily file the `.exe`, because as far as it can see nobody ever said
# not to.
#
# That is the defining property of a window: it forgets the **oldest**, and
# the oldest is where people state their requirements. Recency is a poor
# proxy for importance.
#
# The standard mitigation is to pin the opening turns:

# %%
pinned = WindowedMemory(system=SYSTEM, budget=120, keep_first=1)
for role, content in SESSION:
    pinned.add(role, content)
print(format_messages(pinned.render()))
print(f"\ndropped {pinned.dropped} from the middle instead")

# %% [markdown]
# Better — the task statement survives. The `.exe` instruction still does not,
# because it was said in the middle, and nothing about position tells you a
# constraint was stated. Hold that thought until 17.7.
#
# ### Strategy 3: summarise the old

# %%
def fake_summariser(messages: list[dict]) -> str:
    """Stands in for a model call, so this cell is reproducible.

    A real one sends `messages` to the model asking for a summary. Note that
    `messages` includes the *previous* summary when there is one, so a real
    prompt must say "fold this in" - otherwise the summary grows forever and
    compaction starts costing more context than it saves.

    This stand-in returns a fixed-length summary of the filing actions. It
    does not mention the .exe rule, and that is not me cheating: a summariser
    has no way to know that one line mattered more than the others.
    """
    return "[earlier: filed invoices and lease to finance, notes to study, photos to photos]"


summarising = SummarisingMemory(system=SYSTEM, budget=120, keep_recent=4,
                                summarise=fake_summariser)
for role, content in SESSION:
    summarising.add(role, content)
    summarising.compact()

print(format_messages(summarising.render()))
print(f"\ncompactions: {summarising.compactions}, tokens: {summarising.token_count()}")

# %% [markdown]
# This is what "context compaction" means in the tools you use. It buys a much
# longer conversation and costs you **detail you did not know you would need**
# — chosen by whatever wrote the summary.
#
# Notice the `.exe` instruction again. My summariser did not mention it. A real
# one might not either, because it does not know that line mattered more than
# the others. **A summary is lossy, and the loss is not under your control.**

# %%
print(f"{'strategy':<22}{'tokens':>8}{'messages':>10}  what it lost")
for label, memory, lost in [
    ("ConversationBuffer", buffer, "nothing"),
    ("WindowedMemory", window, "the opening AND the .exe rule"),
    ("+ keep_first=1", pinned, "the .exe rule"),
    ("SummarisingMemory", summarising, "detail, incl. the .exe rule"),
]:
    print(f"{label:<22}{memory.token_count():>8}{len(memory.render()):>10}  {lost}")

# %% [markdown]
# > **Exercise 2.** Add a 13th turn where the user says *"actually, go ahead
# > and file the exe under finance"* and re-run all four strategies. Which ones
# > now have both the original prohibition and the reversal, and which have
# > only one of them? Then explain why having only the *prohibition* is safer
# > than having only the *reversal*, and what that implies about which end of a
# > conversation you should drop.

# %% [markdown]
# ## 17.5 · The budget, in numbers
#
# Guessing at context cost is how you discover the limit in production. It is
# arithmetic; do the arithmetic.

# %%
TOOL_SCHEMA_TOKENS = estimate_tokens(json.dumps(triage_tools.schemas()))
SYSTEM_TOKENS = estimate_tokens(SYSTEM)

PER_EXCHANGE = estimate_tokens(SESSION[0][1] + SESSION[1][1])

print(f"system prompt      {SYSTEM_TOKENS:>6} tokens")
print(f"3 tool schemas     {TOOL_SCHEMA_TOKENS:>6} tokens   <- paid on EVERY request")
print(f"one exchange       {PER_EXCHANGE:>6} tokens")

print("\nschema cost scales with the registry, and never goes away:")
per_tool = TOOL_SCHEMA_TOKENS / len(triage_tools)
for n_tools in (3, 10, 30, 100):
    cost = int(per_tool * n_tools)
    print(f"  {n_tools:>3} tools: {cost:>6} tokens/request"
          f"  = {cost / 4096:>5.0%} of a 4k window, before a single word is said")

print()
for window_size in (4096, 32768):
    for n_tools in (3, 30):
        fixed = SYSTEM_TOKENS + int(per_tool * n_tools)
        turns = (window_size - fixed) // PER_EXCHANGE
        print(f"{window_size:>6}-token window, {n_tools:>2} tools: "
              f"{fixed:>5} fixed, room for ~{turns} exchanges")

# %% [markdown]
# Two things fall out of that, and both surprise people:
#
# 1. **Tool schemas are not free, and you pay for them every single request.**
#    Thirty tools is a permanent tax on every call in the conversation. This is
#    the real argument against a giant registry, and it is a better argument
#    than "the model gets confused" because you can measure it.
# 2. **A bigger window fixes capacity, not cost.** At 32k, conversation length
#    stops being the constraint — but the schema tax is per *request*, so you
#    still pay it on every single turn, and a request carrying 30k tokens of
#    history is slower and dearer than one carrying 3k. Capacity and cost are
#    different problems and only one of them is solved by a bigger number.
#
# > **Exercise 3.** `estimate_tokens` falls back to 4 characters per token
# > without a tokenizer. Load Qwen3's tokenizer (notebook 01) and re-measure
# > the tool schemas. How far off was the heuristic? Then measure a Kazakh or
# > Russian sentence of the same length and report both numbers — that gap is
# > a real budgeting bug for non-English users.

# %% [markdown]
# ## 17.6 · The scratchpad: working memory within one run
#
# Everything above is *conversation* memory. There is a second, shorter-lived
# kind: what the agent learned **during the task it is doing right now**.
#
# "I already tried `peek_file` on the .exe and it is not text" is worth
# remembering for the next three steps and worthless next week. Putting it in
# a `FactStore` is the mistake `Scratchpad` exists to prevent.

# %%
pad = Scratchpad()
run = Agent(*ScriptedModel([
    calls("peek_file", {"name": "setup-x86_64.exe"}),
    calls("peek_file", {"name": "setup-x86_64.exe", "max_chars": 50}),  # same mistake again
    says("I can't read setup-x86_64.exe - it isn't text. I've left it in place."),
]).as_client(), triage_tools).run("Sort setup-x86_64.exe")
print(run.trace())

for step in run.steps:
    for result in step.tool_results:
        if not result.ok:
            pad.note(f"{result.name}({result.arguments}) failed: {result.content[:50]}")
print("\n" + pad.as_message()["content"])

# %% [markdown]
# The agent burned a step repeating a call it had already been told would
# fail — the error even said what to do instead ("classify it from its name
# and suffix") and it asked again anyway. Feed the scratchpad back into the
# prompt and it has the information it needs not to.
#
# This is the cheapest useful intervention in the whole notebook: **write down
# what failed, and show it to the model.** It costs a few tokens and removes
# an entire failure mode.

# %% [markdown]
# ## 17.7 · Memory as tools
#
# So far memory has been something *your code* manages. The other design hands
# it to the model: give it `remember` and `recall` as tools and let it decide.

# %%
facts = FactStore()


@tool
def remember(key: str, value: str) -> str:
    """Store a durable fact about the user's filing preferences.

    Use this ONLY for standing rules that apply to future sessions, such as
    "never move .exe files" or "invoices go to finance". Do NOT use it for
    what you did in this session.

    Args:
        key: Short identifier, e.g. 'exe files'.
        value: The rule, in one sentence.
    """
    return facts.remember(key, value)


@tool
def recall(query: str) -> dict:
    """Look up previously stored filing rules before acting.

    Use this at the start of a session, and whenever you are unsure whether
    the user has a standing rule about a file type.

    Args:
        query: Word to look for among stored rules.
    """
    return facts.search(query)


memory_tools = ToolRegistry().add(list_downloads, peek_file, file_to, remember, recall)

run = Agent(*ScriptedModel([
    calls("remember", {"key": "exe files", "value": "Never move .exe or .zip; the user checks those."}),
    calls("remember", {"key": "invoices", "value": "Invoices go to finance."}),
    says("Noted both rules."),
]).as_client(), memory_tools).run(
    "Remember: never touch exe or zip files, and invoices always go to finance.")
print(run.trace())
print("\nstored:", json.dumps(facts.facts, indent=2))

# %% [markdown]
# Now a **new session** — a fresh agent, an empty conversation, no history at
# all. The only thing that carries over is the fact store.

# %%
run = Agent(*ScriptedModel([
    calls("recall", {"query": "exe"}),
    calls("list_downloads", {"pattern": ".exe"}),
    says("You have one .exe, setup-x86_64.exe. Your standing rule says not to "
         "move it, so I've left it where it is."),
]).as_client(), memory_tools).run("Tidy up whatever is left.")
print(run.trace())

# %% [markdown]
# The constraint that every windowing strategy in 17.4 lost is now **the one
# thing that survived**, because somebody decided it was a rule rather than a
# remark.
#
# ### And here is the bill
#
# Handing memory to the model means the model decides what is true about you,
# and it is not careful.

# %%
careless = FactStore()
for key, value in [
    ("user mood", "The user seemed irritated about the mess."),
    ("exe files", "The user might not want exe files moved, probably."),
    ("exe files", "Actually the user said to file everything."),  # overwrites!
    ("session", "Filed 8 files on 2026-03-14."),                  # not durable
]:
    careless.remember(key, value)

print(careless.as_system_message()["content"])

# %% [markdown]
# Four problems in four lines, and every one shows up in real systems:
#
# 1. **An inference recorded as a fact.** "Seemed irritated" is the model's
#    guess, and it will be in the prompt next March.
# 2. **Hedged facts.** "Probably" is not something the next session can act on,
#    but it will try.
# 3. **Silent overwrite.** The second `exe files` replaced the first. No
#    conflict, no flag, no history — just a reversed rule.
# 4. **Episodic noise stored as durable fact.** "Filed 8 files on 2026-03-14"
#    belongs in a log, not in every future prompt.
#
# Mitigations, in order of how much they help:
#
# | Fix | Why |
# |---|---|
# | **Bound the store** | `FactStore(max_facts=...)` raises instead of growing forever |
# | **Confirm writes with the user** | A fact is a durable change; treat it like one |
# | **Version instead of overwrite** | Keep the old value and the timestamp |
# | **Review the store** | It is a JSON file. Print it. Read it. Delete things. |
#
# That last one is the real advantage of a fact store over a summary: **you
# can read it and correct it.** Try that with a compacted context.
#
# > **Exercise 4.** `FactStore` currently overwrites silently. Add
# > `remember(key, value, *, confirm_overwrite=False)` that raises a
# > `ToolError` naming the existing value when the key exists and
# > `confirm_overwrite` is False. Script a run where the model overwrites a
# > rule, reads the refusal, and re-calls with the flag. Then say why this is
# > the same pattern as `send_message(confirmed="yes")` from notebook 10.

# %% [markdown]
# ## 17.8 · What to store where
#
# One table, and it settles most arguments:
#
# | Information | Where | Why |
# |---|---|---|
# | "What did we just say?" | conversation buffer | needed now, not later |
# | "I already tried X and it failed" | `Scratchpad` | this task only |
# | "Invoices go to finance" | `FactStore` | a rule, durable, few |
# | "Filed 8 files on 14 March" | a log file | audit, not prompt |
# | The contents of 400 documents | a vector index (notebook 09) | too big for any prompt |
#
# The failure is almost always putting something one row too high: episodic
# detail in the fact store, or documents in the conversation. Both work for a
# week and then quietly stop.
#
# ### The honest summary
#
# There is no "long-term memory" feature here, and there is none in the
# products that advertise it either. There is a fact store, a summariser, and
# retrieval — three ordinary mechanisms, each with a failure mode you can
# name. Anyone describing it as the model remembering is selling something.

# %%
shutil.rmtree(SANDBOX, ignore_errors=True)
print("sandbox removed")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is memory the application's job rather than the model's?
# 2. Why does `safe_path` call `resolve()` before checking, and what breaks if
#    you check the string for `..` instead?
# 3. A sliding window forgets the oldest turns. Why is that the wrong thing to
#    forget, and what does `keep_first` fix?
# 4. What does summarisation cost, and who chooses what is lost?
# 5. You pay for tool schemas on every request. What follows about registry size?
# 6. Give one thing that belongs in a scratchpad and one that belongs in a fact
#    store, and say what goes wrong if you swap them.
# 7. Name three ways a model-written fact store goes bad.
#
# ➡️ **Next:** [`18_planning_and_decomposition.ipynb`](18_planning_and_decomposition.ipynb) —
# deciding all the steps before running any of them.
