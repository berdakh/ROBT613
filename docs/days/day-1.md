---
title: "Day 1 · Foundations"
parent: Daily plan
nav_order: 1
---

# Day 1 — Foundations and your first local model
{: .no_toc }

**Goal:** by this evening, a language model runs on your machine and you
understand what it is doing.

1. TOC
{:toc}

---

## Schedule

| Time | Activity | Notebook |
|---|---|---|
| 0:00–0:30 | Setup check, introductions | — |
| 0:30–1:30 | What an LLM actually is; tokenization | `01_llm_foundations` |
| 1:30–2:15 | Your hardware; where models come from | `02_environment_and_hardware` |
| 2:15–2:30 | *Break — start your downloads now* | — |
| 2:30–4:00 | First generation, streaming, conversation | `03_first_generation` |
| 4:00–5:15 | Decoding parameters, thinking mode, prompting | `04_decoding_and_prompting` |
| 5:15–6:00 | Lab: the message-rewriter, and discussion | — |

## The one idea

> A language model takes a sequence of tokens and outputs a probability
> distribution over the next token. Everything else is built on repeating that.

Hold onto this. When something surprising happens later in the week —
the model miscounts letters, forgets your name, contradicts itself mid-answer —
the explanation almost always reduces to this sentence.

## What to pay attention to

**In notebook 01**, the chars-per-token table. English gets ~4 characters per
token; Kazakh and Russian get ~2. The same document costs twice the context
window. This is not a detail — it changes what is affordable.

**In notebook 02**, your own numbers. Write down your RAM, your VRAM if any,
and which model the recommender picked. You will need them all week.

**In notebook 03**, the prompt-token count growing every conversational turn.
That growth *is* the model's "memory", and it is why long chats get slow and
eventually break.

**In notebook 04**, the two runs at `temperature=0.1` versus `temperature=1.3`.
Sampling settings are not a footnote; they are the difference between a useful
assistant and a random text generator.

## Start your downloads early

The first substantial download happens in notebook 03. Start it during the
first break so you are not waiting later:

```bash
python scripts/download_models.py --set core
```

That fetches Qwen3-0.6B and the embedding model — about 3 GB.

{: .tip }
> On a slow or filtered connection, set `export HF_ENDPOINT=https://hf-mirror.com`
> before running, or use ModelScope. See [Troubleshooting](../guides/troubleshooting.html).

## Common day-1 problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: qwen_workshop` | You skipped the `sys.path.insert` cell at the top of the notebook |
| Jupyter uses the wrong Python | `python -m ipykernel install --user --name qwen-workshop` |
| Download stalls at 0% | Network or region — use a mirror |
| Generation is extremely slow | You are on CPU. Expected. Use Qwen3-0.6B. |
| Model repeats itself endlessly | Wrong sampling preset for the mode — see notebook 04 |

## Checkpoint

Before you stop for the day, answer these without looking:

1. What does a language model output at each step?
2. Why does the same prompt give different answers?
3. Why is counting the letters in "strawberry" hard, and what is the correct fix?
4. What does the chat template do?
5. What is your machine's tokens/second for Qwen3-0.6B?
6. What is the difference between "open weights" and "open source"?

## Optional reading

- [Open-weight models](../guides/open-weight-models.html) — the full picture
- [The Qwen family](../guides/qwen-family.html) — what else is available
- [Andrej Karpathy, *Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) —
  two hours, builds a small transformer from nothing. The best use of an evening
  if you want the mechanism rather than the API.
