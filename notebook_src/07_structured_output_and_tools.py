# %% [markdown]
# # 07 · Structured output and tool calling
#
# **Day 2 · ~90 minutes**
#
# This notebook is the bridge from "chatbot" to "software component". A model
# that returns prose is a demo. A model that returns **validated JSON** and
# **calls your functions** is a system you can build on.
#
# By the end you will be able to:
#
# 1. get reliable JSON out of a small model — three techniques, ranked;
# 2. validate model output with Pydantic and handle failures;
# 3. define tools and let Qwen call them;
# 4. read what a tool call looks like on the wire;
# 5. explain why tools beat prompting for anything factual.

# %%
import json
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_workshop.client import complete, get_client, is_up, require_backend

BACKEND, MODEL = "ollama", "qwen3:0.6b"
require_backend(BACKEND)
client, backend = get_client(BACKEND, model=MODEL)
print(backend)

# %% [markdown]
# ## 7.1 · The problem
#
# Ask nicely for JSON and you get JSON… most of the time. "Most of the time" is
# not a foundation for software.

# %%
naive = complete(
    client, backend,
    [{"role": "user", "content":
      "Extract the date, merchant and amount from: 'Small Magnum, 12 March, 4200 KZT'. "
      "Reply with JSON."}],
    temperature=0.7, max_tokens=200, enable_thinking=False,
)
raw = naive.choices[0].message.content
print(repr(raw))

try:
    json.loads(raw)
    print("\nparsed fine this time")
except json.JSONDecodeError as exc:
    print(f"\nJSONDecodeError: {exc}")
    print("-> typical culprits: ```json fences, a preamble sentence, trailing commas")

# %% [markdown]
# Run that cell a few times. You will see markdown fences, "Sure, here is the
# JSON:", or a trailing explanation. Three fixes, from weakest to strongest:

# %% [markdown]
# ## 7.2 · Technique 1 — prompt harder (weakest)

# %%
STRICT_PROMPT = """Extract fields from the receipt text.

Output ONLY a JSON object, with no markdown fences and no explanation.
Schema:
{"date": "YYYY-MM-DD", "merchant": "string", "amount_kzt": number, "category": "string"}

Receipt: Small Magnum, 12 March 2026, 4200 KZT"""

reply = complete(client, backend, [{"role": "user", "content": STRICT_PROMPT}],
                 temperature=0.0, max_tokens=200, enable_thinking=False)
print(reply.choices[0].message.content)

# %% [markdown]
# Better, but still a request, not a guarantee. Always pair it with a tolerant
# parser:

# %%
# This lives in the package (with tests) so we import it rather than keeping a
# second copy that can drift. Read it - it is the parser you will reuse.
import inspect

from qwen_workshop.parsing import extract_json

print(inspect.getsource(extract_json))

# %%
# Prove it on the messy shapes models actually produce.
for sample in [
    '{"a": 1}',
    'Sure! Here is the JSON:\n```json\n{"a": 2}\n```\nLet me know if you need more.',
    'The result is {"a": 3, "note": "a } inside a string"} - hope that helps!',
    "no json at all",
]:
    print(f"{sample[:52]!r:<56} -> {extract_json(sample)}")

# %% [markdown]
# ## 7.3 · Technique 2 — validate with Pydantic (essential)
#
# Parsing tells you it is JSON. **Validation tells you it is the *right* JSON**
# — right fields, right types, sensible values.

# %%
try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydantic"], check=True)
    from pydantic import BaseModel, Field, ValidationError, field_validator

from typing import Literal


class Expense(BaseModel):
    """One parsed receipt. This class is simultaneously documentation,
    a prompt (via its JSON schema) and a runtime guarantee."""

    date: str = Field(description="ISO date, YYYY-MM-DD")
    merchant: str = Field(min_length=1, description="Shop or service name")
    amount_kzt: float = Field(gt=0, description="Amount in tenge, positive")
    category: Literal["groceries", "transport", "eating out", "utilities",
                      "study", "fun", "health", "other"]

    @field_validator("date")
    @classmethod
    def check_date(cls, value: str) -> str:
        import datetime

        datetime.date.fromisoformat(value)  # raises if malformed
        return value


print(json.dumps(Expense.model_json_schema(), indent=2)[:600], "...")

# %%
def parse_expense(text: str, retries: int = 2) -> Expense | None:
    """Extract an Expense, feeding validation errors back to the model.

    This retry loop is the pattern to remember: a validation error is not a
    failure, it is a *better prompt*. The model is told exactly what was wrong
    and usually fixes it on attempt two.
    """
    schema = json.dumps(Expense.model_json_schema())
    messages = [
        {"role": "system", "content":
         "You extract structured data. Output ONLY a JSON object matching the schema. "
         "No markdown, no explanation."},
        {"role": "user", "content": f"Schema:\n{schema}\n\nReceipt text: {text}"},
    ]

    for attempt in range(1, retries + 2):
        reply = complete(client, backend, messages, temperature=0.0,
                         max_tokens=300, enable_thinking=False)
        content = reply.choices[0].message.content or ""
        data = extract_json(content)

        if data is not None:
            try:
                return Expense.model_validate(data)
            except ValidationError as exc:
                problem = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
        else:
            problem = "the response was not valid JSON"

        print(f"  attempt {attempt} failed -> {problem}")
        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": f"That was invalid: {problem}. "
                                        f"Output only corrected JSON."},
        ]
    return None


for receipt in [
    "Small Magnum, 12 March 2026, 4200 KZT",
    "paid 1500 tenge at Coffee Boom yesterday (today is 2026-03-12)",
    "Yandex Go ride 2026-03-09 — 1,800₸",
]:
    print(f"\n{receipt!r}")
    expense = parse_expense(receipt)
    print(f"  -> {expense!r}" if expense else "  -> gave up after retries")

# %% [markdown]
# ## 7.4 · Technique 3 — constrained decoding (strongest)
#
# The previous techniques *ask* for valid JSON. Constrained decoding makes
# invalid JSON **impossible**: at each step the sampler masks out every token
# that could not continue a valid document under your schema.
#
# | Server | How |
# |---|---|
# | Ollama | `format=<json schema>` in the native API, or `response_format` |
# | vLLM | `extra_body={"guided_json": schema}` |
# | llama.cpp | `--grammar-file` or `json_schema` in the request |
# | transformers | the `outlines` or `xgrammar` libraries |
#
# With this, retries become unnecessary — the output is valid by construction.

# %%
schema = Expense.model_json_schema()

try:
    constrained = client.chat.completions.create(
        model=backend.model,
        messages=[{"role": "user", "content":
                   "Extract the expense: 'Galmart, 2026-03-11, 7350 KZT, weekly shop'"}],
        temperature=0.0,
        max_tokens=300,
        extra_body={
            "format": schema,                 # Ollama
            "guided_json": schema,            # vLLM
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    content = constrained.choices[0].message.content
    print("raw:", content)
    print("validated:", Expense.model_validate_json(content))
except Exception as exc:  # noqa: BLE001
    print(f"Constrained decoding not available on this server: {type(exc).__name__}: {exc}")
    print("Fall back to technique 2 - it works everywhere.")

# %% [markdown]
# > **Exercise 1.** Build a `CalendarEvent` model (title, date, start time,
# > duration minutes, location, attendees list) and extract events from three
# > free-text sentences. Which field does the small model get wrong most often?

# %% [markdown]
# <img src="../docs/assets/diagrams/tool-calling.svg" alt="The tool-calling round trip: the model requests, your code executes, the result returns" width="100%">

# %% [markdown]
# ## 7.5 · Tool calling
#
# Structured output lets the model *return* data. Tool calling lets it
# *request an action*. The flow:
#
# ```
# 1. You send: messages + tool schemas
# 2. Model replies: "call get_weather(city='Astana')"   <- it cannot run it
# 3. YOUR CODE runs the function
# 4. You send: messages + the tool's result
# 5. Model replies: the final answer in plain language
# ```
#
# **Step 3 is the one that matters.** The model never executes anything. It
# emits a request; your code decides whether to honour it. Every security
# property of an agent lives in that decision.

# %%
from qwen_workshop.tools import ToolError, ToolRegistry, tool


@tool
def count_letters(word: str, letter: str) -> int:
    """Count how many times a letter appears in a word.

    Args:
        word: The word to inspect.
        letter: The single letter to count.
    """
    return word.lower().count(letter.lower())


@tool
def calculate(expression: str) -> float:
    """Evaluate a arithmetic expression such as '(3 + 4) * 12'.

    Args:
        expression: An arithmetic expression using + - * / ( ) and numbers only.
    """
    # NEVER use eval() on model output. This parses an AST and permits only
    # arithmetic nodes - see section 7.8.
    import ast
    import operator

    allowed = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ToolError(f"unsupported expression element: {ast.dump(node)[:60]}")

    try:
        return evaluate(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError:
        raise ToolError("division by zero")
    except SyntaxError:
        raise ToolError(f"could not parse {expression!r} as arithmetic")


@tool
def current_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    import datetime

    return datetime.date.today().isoformat()


registry = ToolRegistry().add(count_letters, calculate, current_date)
print(registry.describe())

# %% [markdown]
# ### The schema the model sees
#
# Note that **your docstring became the description**. The model chooses tools
# by reading these. A lazy docstring is a lazy prompt.

# %%
print(json.dumps(registry.schemas()[1], indent=2))

# %% [markdown]
# ### One full round trip, done manually

# %%
messages = [{"role": "user", "content": "How many r's are in 'strawberry'? Use your tools."}]

first = complete(client, backend, messages, tools=registry.schemas(),
                 temperature=0.3, max_tokens=400, enable_thinking=False)
message = first.choices[0].message
print("finish_reason:", first.choices[0].finish_reason)
print("content      :", repr(message.content))
print("tool_calls   :", message.tool_calls)

# %%
if message.tool_calls:
    # Step 3: OUR code executes it.
    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ],
    })

    for call in message.tool_calls:
        result = registry.call(call.function.name, call.function.arguments)
        print(f"executed {result.name}({result.arguments}) -> {result.content}")
        messages.append(result.to_message(call_id=call.id))

    # Step 5: back to the model for a natural-language answer.
    final = complete(client, backend, messages, temperature=0.3,
                     max_tokens=200, enable_thinking=False)
    print("\nFinal answer:", final.choices[0].message.content)
else:
    print("The model answered without tools. Small models sometimes skip them —")
    print("try making the system prompt insist, or use a larger model.")

# %% [markdown]
# ## 7.6 · What it looks like on the wire
#
# When the server does not parse tool calls for you, Qwen3 emits them as text
# in Hermes format. Worth seeing once, so the abstraction is not magic:
#
# ```
# <tool_call>
# {"name": "count_letters", "arguments": {"word": "strawberry", "letter": "r"}}
# </tool_call>
# ```
#
# The server's "tool parser" turns that into the structured `tool_calls` field.
# For vLLM you must enable it explicitly:
#
# ```bash
# vllm serve Qwen/Qwen3-0.6B --enable-auto-tool-choice --tool-call-parser hermes
# ```

# %%
from qwen_workshop.chat import extract_tool_calls

sample = ('I will count them.\n<tool_call>\n'
          '{"name": "count_letters", "arguments": {"word": "strawberry", "letter": "r"}}\n'
          '</tool_call>')
print(extract_tool_calls(sample))

# %% [markdown]
# ## 7.7 · Tools beat prompting — measured
#
# The letter-counting task from notebook 01, both ways.

# %%
WORDS = [("strawberry", "r", 3), ("bookkeeper", "e", 3), ("Mississippi", "s", 4),
         ("committee", "t", 2), ("possession", "s", 4)]


def ask_directly(word: str, letter: str) -> str:
    reply = complete(client, backend,
                     [{"role": "user", "content":
                       f"How many '{letter}' are in '{word}'? Reply with only the number."}],
                     temperature=0.0, max_tokens=30, enable_thinking=False)
    return (reply.choices[0].message.content or "").strip()


print(f"{'word':<14}{'expected':>9}{'no tool':>10}{'with tool':>11}")
print("-" * 44)
direct_hits = 0
for word, letter, expected in WORDS:
    guess = ask_directly(word, letter)
    tooled = registry.call("count_letters", {"word": word, "letter": letter}).content
    direct_hits += str(expected) in guess
    print(f"{word:<14}{expected:>9}{guess[:8]:>10}{tooled:>11}")

print(f"\nwithout tools: {direct_hits}/{len(WORDS)} correct")
print(f"with tools   : {len(WORDS)}/{len(WORDS)} correct, and it always will be")

# %% [markdown]
# This is the central lesson of the agentic day. **Do not fine-tune a model to
# do arithmetic. Give it a calculator.**

# %% [markdown]
# ## 7.8 · Security: tools are your attack surface
#
# The moment a model can trigger code, everything it reads becomes potentially
# hostile input — including web pages, emails and documents (*prompt
# injection*).
#
# Rules, in order of importance:
#
# 1. **Never `eval()` or `exec()` model output.** Not with a filter, not "just
#    for the demo". Note how `calculate` above parses an AST and allows only
#    arithmetic nodes.
# 2. **Never build SQL or shell strings from model output.** Parameterise.
# 3. **Allow-list, never deny-list.** Enumerate what is permitted.
# 4. **Constrain the blast radius.** A file tool gets one directory. A network
#    tool gets one domain. Read-only unless writing is the actual job.
# 5. **Require confirmation for irreversible actions** — sending, paying,
#    deleting. A human in the loop for anything you cannot undo.
# 6. **Treat retrieved text as data, never instructions.** A document saying
#    "ignore previous instructions" is an attack, not a command.

# %%
# A deliberately safe file tool. Read the guard clause carefully.
SAFE_ROOT = (REPO_ROOT / "data").resolve()


@tool
def read_note(filename: str) -> str:
    """Read one of the user's personal notes.

    Args:
        filename: File name inside the notes folder, e.g. "cooking.md".
    """
    candidate = (SAFE_ROOT / "notes" / filename).resolve()
    # The containment check. Without it, "../../../etc/passwd" works.
    if not candidate.is_relative_to(SAFE_ROOT):
        raise ToolError("access denied: path escapes the notes folder")
    if not candidate.is_file():
        available = sorted(p.name for p in (SAFE_ROOT / "notes").glob("*.md"))
        raise ToolError(f"no such note. Available: {', '.join(available)}")
    return candidate.read_text(encoding="utf-8")[:4000]


registry.add(read_note)

print(registry.call("read_note", {"filename": "cooking.md"}).content[:120], "...\n")
print(registry.call("read_note", {"filename": "../../etc/passwd"}).content)
print(registry.call("read_note", {"filename": "nonexistent.md"}).content)

# %% [markdown]
# > **Exercise 2.** Add a `search_notes(query)` tool that greps the notes
# > folder. Then ask the model a question that needs both `read_note` and
# > `search_notes`. Does the 0.6B model chain them correctly?
#
# ## Checkpoint
#
# 1. Rank the three JSON techniques and say when each is appropriate.
# 2. In the tool-calling flow, which step executes the function?
# 3. Why is a good docstring a good prompt?
# 4. Name three rules for writing a safe tool.
#
# ➡️ **Next:** [`08_embeddings_and_search.ipynb`](08_embeddings_and_search.ipynb) —
# teach the model to find things.
