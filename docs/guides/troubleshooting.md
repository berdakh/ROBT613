---
title: Troubleshooting
parent: Guides
nav_order: 8
---

# Troubleshooting
{: .no_toc }

Start here when something breaks. Organised by what you see.

1. TOC
{:toc}

---

## First, run this

```bash
python scripts/check_env.py
```

It reports your Python, hardware, installed packages, cache location and
available runtimes. Most problems are visible in that output.

---

## Import and environment

### `ModuleNotFoundError: No module named 'qwen_workshop'`

The `src` directory is not on the path. Every notebook starts with a cell that
fixes this — you probably skipped it:

```python
import sys
from pathlib import Path
REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
```

If you are running a script rather than a notebook, either `pip install -e .`
or run from the repository root.

### A package is installed but Python cannot import it

Jupyter is almost certainly running a different interpreter from your terminal.
Check both:

```python
import sys; print(sys.executable)
```

Fix by registering your virtual environment as a kernel:

```bash
python -m ipykernel install --user --name qwen-workshop
```

Then **Kernel → Change kernel** in Jupyter.

### `ImportError` mentioning `transformers` and a version

```bash
pip install -U transformers accelerate
```

Qwen3 support needs a reasonably recent `transformers`. If you see
`KeyError: 'qwen3'`, your version predates the architecture.

---

## Downloads

### Download is extremely slow, or stalls

```bash
export HF_ENDPOINT=https://hf-mirror.com     # a mirror
```

Or use ModelScope:

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
```

Or use Ollama, which has its own CDN: `ollama pull qwen3:0.6b`.

Downloads resume, so retrying does not restart from zero.

### `401 Client Error` / `Repository not found`

A gated or private repo. Accept the licence on the model page, then:

```bash
huggingface-cli login
```

If the repo is public and you still get 404, check the spelling — `Qwen/Qwen3-0.6B`
is case-sensitive.

### `No space left on device`

```bash
huggingface-cli scan-cache       # see what is large
huggingface-cli delete-cache     # interactive cleanup
export HF_HOME=/bigger/disk/hf   # or move the cache entirely
```

### Windows: `OSError` about long file names

Enable long paths, or set `HF_HOME` to something short like `C:\hf`.

---

## Memory

### `torch.cuda.OutOfMemoryError`

In order of what to try:

1. **Restart the kernel.** Previously loaded models are still resident. This
   fixes it more often than anything else.
2. **Use a smaller model** — Qwen3-0.6B or 1.7B.
3. **Quantize:** `load_model(..., quantize_4bit=True)`.
4. **Shorten the context.** The KV cache grows with conversation length.
5. **Reduce batch size** to 1.
6. **Fall back to CPU:** `load_model(..., device="cpu")`. Slow, but it works.

To free memory inside a notebook:

```python
import gc, torch
del model
gc.collect()
torch.cuda.empty_cache()
```

### It ran fine, then OOMed after several cells

You loaded a second model without releasing the first. This is the single most
common notebook memory bug. Use the `free_memory()` helper in notebook 03.

### The process is killed with no error (CPU)

The OS out-of-memory killer. You ran out of system RAM. Use a smaller model, or
close other applications.

---

## Generation quality

### The model repeats itself endlessly

Almost always the wrong sampling preset for the mode:

| Mode | temperature | top_p | top_k |
|---|---|---|---|
| Thinking | 0.6 | 0.95 | 20 |
| Non-thinking | 0.7 | 0.8 | 20 |

Also: **never use greedy decoding in thinking mode** — it loops. And check you
did not set `temperature=0` with `do_sample=True`.

### Output is truncated mid-sentence

`max_new_tokens` / `max_tokens` is too low. Check `finish_reason`: if it says
`"length"`, you were cut off.

Thinking mode needs a *lot* of headroom — 1024 tokens or more, because the
reasoning block is counted too.

### The model echoes my prompt back

You forgot to slice the output. `generate()` returns prompt + completion:

```python
new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
```

### The model ignores my system prompt

- Print the templated prompt and confirm the system message is actually in it.
- Check you are not using a `-Base` checkpoint — base models have no chat
  template and do not follow instructions.
- Small models genuinely drift from long system prompts. Shorten and be specific.

### Answers are rambling and unfocused

Lower the temperature, be specific about the output shape (length, format), and
consider putting your instruction *after* the data for long inputs.

### The model rambles and ignores questions entirely

You downloaded a `-Base` model. Get the instruct version — for Qwen3 that is the
plain name, e.g. `Qwen/Qwen3-8B`, not `Qwen/Qwen3-8B-Base`.

---

## Servers and tools

### `No server answering at http://localhost:11434/v1`

```bash
ollama serve          # start it
ollama list           # confirm a model is pulled
curl http://localhost:11434/v1/models
```

On Windows, Ollama runs as a background service — check the system tray.

### The model never calls my tools

1. **Check the schemas are being sent.** Print `registry.schemas()`.
2. **For vLLM**, tool calling must be enabled:
   `--enable-auto-tool-choice --tool-call-parser hermes`.
3. **Improve the descriptions.** Say *when* to use the tool, not just what it does.
4. **Strengthen the system prompt:** "You must call a tool before answering
   questions about the user's data."
5. **Try a larger model.** 0.6B is genuinely weak at tool selection; 4B and 8B
   are much better. This is a real capability difference, not a bug.

### Tool calls have malformed arguments

Use `Literal` types for closed sets so they become enums. Keep argument counts
low. Lower the temperature. And make sure your `ToolError` messages explain how
to fix the call — the model reads them.

### `tool_calls` is always `None` but the text contains `<tool_call>`

The server's tool parser is not enabled, so it is passing the raw text through.
Enable it (see above), or parse it yourself with
`qwen_workshop.chat.extract_tool_calls`.

---

## RAG

### Retrieval returns irrelevant chunks

- Check the chunks look sensible: `print(chunks[0].text)`.
- Vocabulary mismatch — try query rewriting (notebook 09, §9.7).
- Try hybrid search (notebook 08, §8.5).
- Measure recall@k properly before changing anything else.

### The model answers from its own knowledge instead of the context

- Strengthen the prompt: "Use ONLY the context. If it is not there, say so."
- Lower the temperature to 0.2 or below.
- Reduce the amount of context — small models get lost in long prompts.
- Verify citations with `cited_sources()` to detect it automatically.

### It answers questions it should refuse

You have no score threshold. A vector index always returns *k* results. Add
`min_score` (0.35–0.4 is a reasonable starting point).

---

## Fine-tuning

### Loss does not decrease

Learning rate too low (try 2e-4 for LoRA), or too few steps, or the data format
is wrong. Print one formatted training example and check it looks like your
inference input.

### Loss goes to zero immediately

You have memorised a tiny dataset. Expected on 12 examples; on a real dataset,
stop and add data.

### The fine-tuned model is worse at everything

Catastrophic forgetting. Fewer steps, lower learning rate, lower LoRA rank, or
mix general instruction data into your training set.

### `bitsandbytes` fails on Windows

Support is patchy. Use WSL2, or use GGUF via Ollama for quantized inference.

---

## Still stuck?

1. Read the **full** error message, including the last few lines of the traceback.
2. Search the exact error text — someone has usually hit it.
3. Reduce to the smallest cell that reproduces it.
4. Open an issue on the repository with:
   - the full output of `python scripts/check_env.py`
   - the complete traceback
   - the smallest code that reproduces it
