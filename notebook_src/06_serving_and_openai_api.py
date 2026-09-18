# %% [markdown]
# # 06 · Serving Qwen behind an API
#
# **Day 2 · ~60 minutes**
#
# So far the model has lived inside the notebook. Real applications need it
# behind a network boundary, so that a web app, a phone, or three of your
# classmates can all use one GPU.
#
# The key insight: **vLLM, Ollama, llama.cpp and LM Studio all speak the
# OpenAI API.** Write your application against that interface and you can swap
# model and runtime by changing two strings.
#
# By the end you will be able to:
#
# 1. start a local OpenAI-compatible server;
# 2. call it with the official `openai` Python SDK;
# 3. stream responses and count tokens;
# 4. explain when to choose vLLM over Ollama;
# 5. write code that runs unchanged against either.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_workshop.client import BACKENDS, is_up

for name, spec in BACKENDS.items():
    status = "RUNNING" if is_up(name) else "not running"
    print(f"{name:<12}{spec['base_url']:<34}{status}")

# %% [markdown]
# <img src="../docs/assets/diagrams/serving.svg" alt="One OpenAI-compatible API served interchangeably by Ollama, vLLM, llama.cpp or LM Studio" width="100%">

# %% [markdown]
# ## 6.1 · Start a server
#
# Pick **one**. Run it in a terminal (not in this notebook — it needs to stay
# running), then come back.
#
# ### Option A — Ollama (recommended for this workshop)
#
# ```bash
# ollama serve              # usually already running after install
# ollama pull qwen3:0.6b
# ```
#
# ### Option B — vLLM (needs an NVIDIA GPU)
#
# ```bash
# pip install vllm
# vllm serve Qwen/Qwen3-0.6B --max-model-len 8192 --port 8000
# ```
#
# ### Option C — llama.cpp
#
# ```bash
# llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q4_K_M -c 8192 --port 8080 --jinja
# ```
#
# Then re-run the cell above until one says `RUNNING`.

# %%
BACKEND = "ollama"   # change to "vllm" or "llamacpp" to match what you started
MODEL = "qwen3:0.6b" # for vLLM use "Qwen/Qwen3-0.6B"

from qwen_workshop.client import require_backend

try:
    require_backend(BACKEND)
    print(f"{BACKEND} is up")
except RuntimeError as exc:
    print(exc)

# %% [markdown]
# ## 6.2 · Your first API call
#
# Note what is happening here: we are using **the official OpenAI SDK**, but
# every byte stays on your machine. The `api_key` is required by the SDK and
# ignored by the server.

# %%
try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai"], check=True)
    from openai import OpenAI

from qwen_workshop.client import get_client

client, backend = get_client(BACKEND, model=MODEL)
print(backend)

# %%
response = client.chat.completions.create(
    model=backend.model,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "In one sentence: why run a model locally?"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
print()
print(f"prompt tokens    : {response.usage.prompt_tokens}")
print(f"completion tokens: {response.usage.completion_tokens}")
print(f"finish reason    : {response.choices[0].finish_reason}")

# %% [markdown]
# `finish_reason` matters more than students expect:
#
# - `stop` — the model finished naturally. Good.
# - `length` — **it was cut off by `max_tokens`**. Your JSON is truncated, your
#   sentence is half-written. Always check this in production code.
# - `tool_calls` — it wants to call a tool (notebook 07).

# %%
truncated = client.chat.completions.create(
    model=backend.model,
    messages=[{"role": "user", "content": "Count from 1 to 100, one number per line."}],
    max_tokens=20,
)
print(f"finish_reason = {truncated.choices[0].finish_reason}")
print(repr(truncated.choices[0].message.content))
print("\n-> if you parsed that as a complete answer, you would be wrong")

# %% [markdown]
# ## 6.3 · Streaming over the API

# %%
stream = client.chat.completions.create(
    model=backend.model,
    messages=[{"role": "user", "content": "Write a haiku about cold weather and hot tea."}],
    stream=True,
    max_tokens=120,
)

pieces = []
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        pieces.append(delta)
        print(delta, end="", flush=True)
print(f"\n\n[{len(pieces)} chunks received]")

# %% [markdown]
# Note: with `stream=True` you do not get a `usage` block by default. Request
# it with `stream_options={"include_usage": True}` if your server supports it.

# %% [markdown]
# ## 6.4 · Thinking mode over the API
#
# Each server exposes Qwen3's thinking switch differently — an annoying but
# real portability wrinkle. Our helper sets both known forms and lets the
# server ignore the one it does not recognise.

# %%
from qwen_workshop.client import complete

for thinking in (False, True):
    result = complete(
        client, backend,
        [{"role": "user", "content": "If a train leaves at 14:20 and takes 3h45m, when does it arrive?"}],
        temperature=0.6, max_tokens=800, enable_thinking=thinking,
    )
    content = result.choices[0].message.content or ""
    # Some servers split reasoning into a separate field instead of <think> tags.
    reasoning = getattr(result.choices[0].message, "reasoning_content", None)
    print(f"--- enable_thinking={thinking} ---")
    if reasoning:
        print(f"[reasoning: {len(reasoning.split())} words, in a separate field]")
    print(content.strip()[:400])
    print(f"({result.usage.completion_tokens} completion tokens)\n")

# %% [markdown]
# (The train arrives at 18:05.)

# %% [markdown]
# ## 6.5 · The portability payoff
#
# Here is the entire point of this notebook. One function, any backend.

# %%
def ask(question: str, backend_name: str = BACKEND, model: str | None = None) -> str:
    """Ask a question to whichever local runtime is available."""
    client, backend = get_client(backend_name, model=model)
    reply = complete(client, backend, [{"role": "user", "content": question}],
                     temperature=0.3, max_tokens=120, enable_thinking=False)
    return (reply.choices[0].message.content or "").strip()


question = "Name one advantage of running an AI model on your own computer."

for name in ("ollama", "vllm", "llamacpp"):
    if is_up(name):
        print(f"[{name}] {ask(question, name, BACKENDS[name]['model'])}\n")
    else:
        print(f"[{name}] skipped - not running\n")

# %% [markdown]
# Nothing changed except the backend name. **This is why you should never write
# application code against a specific model's Python API.** Put the OpenAI
# interface in the middle and stay free to change your mind.

# %% [markdown]
# ## 6.6 · Concurrency: the real reason for vLLM
#
# `transformers` handles one request at a time. vLLM uses continuous batching
# and PagedAttention to serve many concurrent users on one GPU, at
# dramatically higher total throughput.
#
# Let's measure concurrency on whatever you have running.

# %%
import time
from concurrent.futures import ThreadPoolExecutor

QUESTIONS = [
    "Name a vegetable that stores well in winter.",
    "What is a good beginner bicycle maintenance task?",
    "Suggest one way to reduce electricity use at home.",
    "What should I pack for an overnight train?",
    "Name a cheap source of protein.",
    "How often should I back up my laptop?",
    "Suggest a 20-minute study technique.",
    "What is a reasonable indoor humidity level?",
]


def timed_sequential(questions):
    start = time.perf_counter()
    for q in questions:
        complete(client, backend, [{"role": "user", "content": q}],
                 max_tokens=50, temperature=0.7, enable_thinking=False)
    return time.perf_counter() - start


def timed_concurrent(questions, workers=8):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(
            lambda q: complete(client, backend, [{"role": "user", "content": q}],
                               max_tokens=50, temperature=0.7, enable_thinking=False),
            questions,
        ))
    return time.perf_counter() - start


if is_up(BACKEND):
    sequential = timed_sequential(QUESTIONS)
    concurrent = timed_concurrent(QUESTIONS)
    print(f"sequential ({len(QUESTIONS)} requests): {sequential:.1f}s")
    print(f"concurrent (8 at once)              : {concurrent:.1f}s")
    print(f"speed-up                            : {sequential / concurrent:.1f}x")
    print("\nOn vLLM with a real GPU expect 4-10x. On Ollama/CPU expect much less -")
    print("it queues requests rather than truly batching them.")
else:
    print("Start a server first.")

# %% [markdown]
# ## 6.7 · Serving checklist for a real deployment
#
# Before you let anyone else use your endpoint:
#
# - [ ] **Bind carefully.** `--host 0.0.0.0` exposes it to your whole network.
#       Default to localhost; use a reverse proxy with auth if you need remote access.
# - [ ] **Set `--max-model-len`.** Unbounded context lets one user exhaust your VRAM.
# - [ ] **Rate-limit.** An open LLM endpoint is a free compute service for whoever finds it.
# - [ ] **Log prompts and latencies** — but think hard first about whether you
#       are allowed to log user content. Often you are not.
# - [ ] **Pin versions.** Model, runtime and tokenizer. Silent upgrades change outputs.
# - [ ] **Health checks.** `GET /v1/models` is a cheap liveness probe.
# - [ ] **Cap `max_tokens` server-side.** Otherwise one request can run for an hour.
#
# 📖 More: [`docs/guides/deployment.md`](../docs/guides/deployment.md)
#
# ## Checkpoint
#
# 1. Why does the OpenAI SDK work against a local server?
# 2. What does `finish_reason == "length"` mean and why must you check it?
# 3. What does vLLM give you that Ollama does not?
# 4. What are the first two things you would lock down before exposing a server?
#
# ➡️ **Next:** [`07_structured_output_and_tools.ipynb`](07_structured_output_and_tools.ipynb) —
# make the model return data your program can trust.
