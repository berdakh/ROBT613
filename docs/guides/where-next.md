---
title: Where next
parent: Guides
nav_order: 11
---

# Where next
{: .no_toc }

You can now run an open-weight model locally, give it tools and documents, wrap
it in an agent, and measure whether it works. Here is where each thread leads.

1. TOC
{:toc}

---

## Multimodal

The most immediately useful extension. **Qwen3-VL** reads images, screenshots,
documents and video frames.

Your receipt parser could take photographs. Your kitchen assistant could look in
the fridge. Your study buddy could read lecture slides directly instead of
needing them converted.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
# The API is close to what you already know; the processor handles images.
```

Start with the Qwen3-VL model card and cookbook. **Qwen3-Omni** adds audio in
and speech out.

---

## MCP — package your tools properly

You wrote tools as Python functions. The [Model Context
Protocol](https://modelcontextprotocol.io) lets you publish them as a server
that *any* client can use — Claude Desktop, IDEs, Qwen-Agent, your own code.

Write the tool once, use it everywhere. This is where the ecosystem has settled,
and it is an afternoon's work to convert what you already have.

---

## Better retrieval

Notebook 09 built a working RAG pipeline. Production systems add:

- **Rerankers** at scale — Qwen3-Reranker on 50 candidates, not 8.
- **Hybrid search** with real BM25 (`rank_bm25`, or Elasticsearch).
- **Query decomposition** — split "compare X and Y" into two retrievals.
- **Contextual chunking** — prepend a document-level summary to each chunk so it
  makes sense in isolation.
- **GraphRAG** — build an entity graph for questions that span many documents.
- **Vector databases** — Qdrant, LanceDB, pgvector when NumPy stops being enough
  (roughly 10⁵ vectors, or when you need filtered search).

---

## Evaluation at scale

The most under-rated skill in the field, and the one that makes you employable.

- **LLM-as-judge** with a rubric, calibrated against human labels.
- **Regression suites** — every bug you fix becomes a permanent test case.
- **Adversarial sets** — deliberately hard and deliberately malicious inputs.
- **Tracing** — [Langfuse](https://langfuse.com), OpenTelemetry, or your own
  structured logs. You cannot debug what you did not record.
- **A/B testing** with real users, if you have them.

---

## Efficiency

- **Speculative decoding** — a small model drafts, a large one verifies. 2–3×
  faster with identical output.
- **KV cache quantization** — the cache is often larger than the weights at long
  context.
- **Prefix caching** — reuse the KV cache for a shared system prompt.
- **Continuous batching** — what vLLM does; worth understanding, not
  reimplementing.
- **torch.compile** and Flash Attention for the transformers path.

---

## Training

Notebook 11 covered LoRA SFT. Beyond it:

- **DPO / ORPO** — preference tuning from pairs of good and bad responses.
- **QLoRA** — LoRA on a 4-bit base, so you can fine-tune a 14B model on a 16 GB GPU.
- **Continued pretraining** — for a genuinely new domain or language, not a new style.
- **Distillation** — train a small model on a large model's outputs for one
  narrow task. Often the right answer when you need 0.6B-level latency.

---

## Read the primary sources

The habit that keeps you current when tutorials go stale.

**Foundational**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the transformer
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — read alongside it
- [Karpathy, *Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) — build one from nothing

**Techniques used in this workshop**
- [LoRA](https://arxiv.org/abs/2106.09685)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [RAG](https://arxiv.org/abs/2005.11401)
- [GPTQ](https://arxiv.org/abs/2210.17323) and [AWQ](https://arxiv.org/abs/2306.00978)

**Practical**
- [Anthropic, *Building effective agents*](https://www.anthropic.com/research/building-effective-agents)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- The Qwen, DeepSeek and OLMo technical reports — more useful per hour than
  almost any tutorial

---

## Practise

Reading will not make you good at this. These will:

1. **Use your capstone for a month.** Fix what annoys you. Nothing teaches like
   being your own user.
2. **Rebuild it with a different model.** Qwen → Llama → Mistral. You will learn
   what was general and what was Qwen-specific.
3. **Break someone else's agent.** Try prompt injection on a classmate's
   capstone, with permission. Then defend your own.
4. **Contribute.** llama.cpp, vLLM and transformers all have approachable issues
   labelled `good first issue`.
5. **Write up what you learned.** A blog post that explains one thing clearly is
   worth more to your future self — and to a hiring manager — than a repository
   of unfinished experiments.

---

## Finally

The field moves fast, and most of what is written about it is either marketing
or hype. The parts that will still be true in five years are the ones this
workshop spent its time on:

- A model predicts the next token. Everything else is built on that.
- Retrieval beats memorisation for facts.
- Tools beat prompting for anything computable.
- Measure, or you are guessing.
- The engineering is in the guardrails, not the demo.

Go build something.
