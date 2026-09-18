---
title: The Qwen family
parent: Guides
nav_order: 3
---

# The Qwen family — which model for which job
{: .no_toc }

1. TOC
{:toc}

{: .warning }
> The Qwen line-up changes every few months. This page is a **snapshot** from
> when the workshop was written. Always confirm sizes, context lengths and
> licences on the model card at [huggingface.co/Qwen](https://huggingface.co/Qwen).

---

## What Qwen is

Qwen (通义千问, *Tongyi Qianwen*) is Alibaba Cloud's model family. It is used in
this workshop because of a rare combination: strong quality, a permissive
licence on the main line (Apache 2.0), sizes from 0.6B to hundreds of billions
with a consistent interface, genuinely good multilingual performance, and
matching embedding and reranker models so the whole RAG stack comes from one
family.

## Qwen3 — the text models

**Dense models**, in increasing size: 0.6B, 1.7B, 4B, 8B, 14B, 32B.

**Mixture-of-Experts (MoE)**: Qwen3-30B-A3B (30B total parameters, ~3B active
per token) and Qwen3-235B-A22B. MoE gives you the memory footprint of a large
model with the *speed* of a small one — a good fit for a workstation with plenty
of RAM but modest compute.

### Which one?

| You have | Use | Realistic expectation |
|---|---|---|
| CPU only, 8 GB RAM | **Qwen3-0.6B** | Learns the mechanics. Weak at facts and multi-step tool use. |
| CPU only, 16 GB RAM | **Qwen3-1.7B / 4B** | Noticeably better instruction following. |
| 8 GB VRAM | **Qwen3-8B** (int4) | Good tool calling and RAG. The sweet spot for the capstone. |
| 16 GB VRAM | **Qwen3-14B** (int4) | Reliable multi-step agents. |
| 24 GB+ VRAM or 32 GB RAM | **Qwen3-30B-A3B** (Q4) | Strong, and surprisingly fast for its size. |
| Free Colab T4 | **Qwen3-4B / 8B** (int4) | Everything in this workshop. |

{: .tip }
> Start with 0.6B for *every* exercise. Get it working, understand the
> mechanism, then scale up. Debugging a 0.6B model takes seconds per iteration;
> debugging a 14B model takes minutes.

### Thinking mode

Qwen3 hybrid checkpoints can emit a reasoning block before answering:

```
<think>The user is asking... let me work through it...</think>
The answer is 391.
```

Toggle it with `enable_thinking=True/False` in the chat template, or inline with
`/think` and `/no_think` in a user message.

Use the **right sampling preset for the mode** — this is the single most common
cause of repetitive or rambling Qwen3 output:

| Mode | temperature | top_p | top_k |
|---|---|---|---|
| Thinking | 0.6 | 0.95 | 20 |
| Non-thinking | 0.7 | 0.8 | 20 |

Greedy decoding is **not** recommended in thinking mode — it tends to loop.

{: .note }
> Later `-Instruct-2507` and `-Thinking-2507` releases split the two modes into
> separate checkpoints and dropped the switch. Qwen2.5 never had it. The
> workshop's `build_prompt()` checks the chat template and degrades gracefully,
> so your code works across all of them.

## Qwen3-Embedding and Qwen3-Reranker

Sizes 0.6B, 4B and 8B.

- **Embedding** models turn text into vectors for search. Instruction-aware:
  encode queries with a task instruction, documents without one. Multilingual,
  so you can search English notes with a Kazakh query.
- **Reranker** models are cross-encoders that re-score retrieved candidates,
  reading the query and document *together*. Slower per item, much more
  accurate. The standard pattern is retrieve 20 with embeddings, rerank to 3.

Used in notebooks 08 and 09.

## Specialist models

| Family | For | Note |
|---|---|---|
| **Qwen3-Coder** | Code generation, repository-scale editing | Strong; large sizes available |
| **Qwen2.5-Coder** | Code, in small sizes (0.5B–7B) | The practical choice for a laptop code assistant |
| **Qwen3-VL** | Images, documents, screenshots, video | The natural next step after this workshop |
| **Qwen3-Omni** | Text, image, audio, speech output | Multimodal in and out |
| **Qwen2.5-Math** | Mathematical reasoning | Narrow but strong |

## Naming conventions

```
Qwen3-8B                     base + instruct, hybrid thinking
Qwen3-8B-Base                pretrained only, NOT chat-tuned
Qwen3-8B-FP8                 FP8 quantized
Qwen3-8B-GGUF                GGUF builds for llama.cpp / Ollama
Qwen3-30B-A3B                MoE: 30B total, 3B active
Qwen3-Embedding-0.6B         embeddings
Qwen3-Reranker-0.6B          reranking
```

{: .warning }
> `-Base` models are **not** chat models. They continue text rather than
> following instructions, and have no chat template. If your "chat" model
> rambles and ignores your questions, check whether you downloaded a base
> checkpoint.

## Licensing

The Qwen3 dense and MoE models are released under **Apache 2.0**: commercial use,
modification and redistribution are permitted, with a patent grant. Some older
and some very large checkpoints in the wider family use different terms
(including research-only licences). **Check the `LICENSE` file on the specific
repo you download** — do not assume family-wide consistency.

## Ollama tags

```bash
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull qwen3:30b-a3b
```

Ollama picks a quantization for you (usually Q4_K_M). For a specific quant,
append it: `qwen3:8b-q8_0`. Confirm availability with `ollama list` or at
[ollama.com/library/qwen3](https://ollama.com/library/qwen3).

## Reading further

- [huggingface.co/Qwen](https://huggingface.co/Qwen) — all model cards
- [Qwen documentation](https://qwen.readthedocs.io) — official usage guides
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) — the official agent framework
- The Qwen technical reports on arXiv — worth an hour of anyone's time
