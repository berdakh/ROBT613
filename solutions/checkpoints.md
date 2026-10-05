# Checkpoint answers

The 49 questions that close notebooks 01–11. Use them as an oral check, a quick
written quiz, or a revision sheet.

Markers as in the [README](README.md): ✅ verified · 🔬 expected · ✍️ yours.
Unmarked answers are definitional and safe to assert.

---

## 01 · LLM foundations

**1. What does a language model output at each step?**
A probability distribution over **every token in the vocabulary** — ~150,000
numbers summing to 1. Not a word; a distribution. *(Watch for "it outputs the
next word" — the distinction is the whole chapter.)*

**2. Why does the same prompt give different answers?**
Because a token is **sampled** from that distribution rather than chosen. Set
`do_sample=False` and answers become identical. Non-determinism is a setting,
not a defect.

**3. Why is counting letters hard, and what is the correct fix?**
The model sees tokens, not characters — "strawberry" is three opaque chunks. The
fix is a **tool** (`word.count(letter)`), not a bigger model or a cleverer
prompt. This generalises: do not ask the model for what a function computes
exactly.

**4. What does the chat template do, and why print it when debugging?**
It turns a list of messages into the single string the model actually receives,
inserting role markers like `<|im_start|>`. Print it because most "the model is
behaving strangely" bugs are **visible in that string** — a missing generation
prompt, a system message that never made it, a mangled tool block.

**5. Name two things open weights give you that an API cannot.**
Any two of: **privacy** (nothing leaves your machine), **permanence** (the model
cannot be deprecated or silently updated under you), **cost** (no per-token
billing), **control** (fine-tune, quantize, inspect, pin a version), **offline
operation**.

---

## 02 · Environment and hardware

**1. How much memory does Qwen3-8B need at int4, with overhead?**
✅ About **4.4 GB of weights** (8e9 × ~0.55 bytes), plus KV cache and ~1 GB
overhead → **~6 GB** at a short context. Credit any answer in 5–7 GB *that
mentions the overhead is not optional*.

**2. Which model did the recommender pick for you, and why?**
✍️ Personal. The reasoning should reference their own RAM/VRAM and the fact that
the CPU budget is deliberately stricter — a model that technically fits but runs
at 2 tok/s ruins a session.

**3. Where is your cache, and how would you move it?**
`~/.cache/huggingface/hub` by default; move it with `export HF_HOME=/path`.
Bonus for knowing `huggingface-cli scan-cache` and `delete-cache`.

**4. Why prefer `.safetensors` over `.bin`?**
`.bin` is a Python **pickle** — loading it executes arbitrary code from whoever
uploaded it. `safetensors` is pure data and cannot execute anything. Faster
memory-mapped loading is a secondary benefit, not the reason.

---

## 03 · First generation

**1. What are the four steps from a message list to displayed text?**
① messages → chat template → one string ② string → token ids ③ `generate()`
④ **slice off the prompt**, then decode.

**2. Why must you slice the output of `generate()`?**
It returns **prompt + completion** concatenated. Without the slice your app
echoes the user's own question back at them. `out[0][inputs["input_ids"].shape[-1]:]`

**3. Why does a chatbot's prompt grow every turn?**
The model has **no memory between calls**. "Memory" is the entire transcript
being re-sent each turn. Hence rising latency and cost, an eventual context
limit, and the classic bug of forgetting to append the assistant's reply.

**4. What is your machine's tokens/second for Qwen3-0.6B?**
✍️ Personal. 🔬 Expect 60–150 on a GPU, 30–60 on Apple Silicon, 8–20 CPU-only.

**5. Why does batching need left padding?**
Decoder-only generation continues from the **last token**. With right padding
the last token is padding, so the model continues from nothing and produces
confident nonsense. Left padding keeps every sequence's real final token at the
end.

---

## 04 · Decoding and prompting

**1. What does temperature do to the logits, mathematically?**
Divides them before softmax: `P(t_i) = exp(z_i / T) / Σ exp(z_j / T)`. `T < 1`
sharpens; `T > 1` flattens. ✅ It adds no information — it only changes how
boldly you sample from what the model already believes.

**2. Why is top-p usually preferable to top-k?**
top-p **adapts to the model's confidence**. On a confident prompt it keeps one
or two tokens; on an uncertain one it keeps many. top-k keeps a fixed number
regardless, so it is either too permissive or too restrictive depending on the
context.

**3. Two tasks where thinking mode pays, two where it does not.**
Pays: maths, multi-step logic, planning, code that must be right first time.
Does not: greetings, translation, classification into fixed labels, anything
latency-sensitive or already constrained by a schema.

**4. Name the four prompt patterns, with an example each.**
① **Be specific about output shape** ("3 bullets, under 15 words each")
② **Give an example** (few-shot classification)
③ **Give an escape hatch** ("reply exactly 'I don't know' if unsure")
④ **Put instructions after the data** (long document, then the question).

---

## 05 · Quantization and runtimes

**1. What does 4-bit quantization store per group of weights?**
A **scale** (and often a zero-point) at higher precision, plus each weight as a
4-bit integer. Reconstruct with `value ≈ scale × integer`.

**2. What do `K` and `M` mean in `Q4_K_M`?**
`K` = **k-quants**, a scheme that spends more bits on the layers that matter
most. `M` = the **medium** variant of that mix (`S` smaller, `L` larger).

**3. Bigger model at Q4 or smaller model at Q8?**
**Bigger at Q4**, at the same file size. Qwen3-8B-Q4_K_M generally beats
Qwen3-4B-Q8_0. Capacity buys more than precision in this range.

**4. Why does 4-bit save memory but not necessarily compute?**
Weights are **stored** in 4 bits but **dequantized to bf16 for each matrix
multiply**. The arithmetic is unchanged. In practice it is often still faster,
because these workloads are memory-bandwidth bound and you are moving a quarter
of the bytes.

---

## 06 · Serving

**1. Why does the OpenAI SDK work against a local server?**
Because the OpenAI chat-completions API became a **de facto standard** and every
runtime implements it. The SDK is just an HTTP client; `base_url` points it
wherever you like. The `api_key` is required by the SDK and ignored locally.

**2. What does `finish_reason == "length"` mean, and why check it?**
Generation was **cut off by `max_tokens`**, not finished. Your JSON is truncated,
your sentence is half-written — and it often still *parses* or *reads* as
complete. Unchecked, this is a silent data-corruption bug.

**3. What does vLLM give you that Ollama does not?**
**Throughput under concurrency** — PagedAttention manages the KV cache in pages,
and continuous batching admits new requests mid-generation. Order-of-magnitude
gains when serving many users. Ollama queues.

**4. First two things to lock down before exposing a server?**
Any two of: **do not bind `0.0.0.0`** without meaning to; **add authentication**
(an open LLM endpoint is free compute for whoever scans your network);
**rate-limit**; **cap `max_tokens` and `--max-model-len` server-side**.

---

## 07 · Structured output and tools

**1. Rank the three JSON techniques.**
① **Prompting** — weakest, works anywhere, always pair with a tolerant parser.
② **Pydantic validation + retry** — essential; feed the validation error back and
the model usually fixes it. ③ **Constrained decoding** — strongest; makes invalid
output *impossible* rather than unlikely. Use ③ where supported, ② always.

**2. In the tool-calling flow, which step executes the function?**
**Step 3 — your code.** The model only ever *requests* a call. Every security
property of an agent lives in your decision whether to honour it.

**3. Why is a good docstring a good prompt?**
Because the docstring **becomes the tool description in the schema**, and that
is what the model reads when choosing. A vague docstring produces vague tool
selection. Saying *when* to use a tool matters more than saying what it does.

**4. Name three rules for writing a safe tool.**
Any three of: never `eval`/`exec` model output; never build SQL or shell strings
from it; allow-list rather than deny-list; confine file access (check
`path.resolve().is_relative_to(root)`); read-only by default; require
confirmation for irreversible actions; cap output size.

---

## 08 · Embeddings and search

**1. What does an embedding model output, and how does it differ from a chat model?**
**One fixed-length vector for the whole text** (1024 dims for
Qwen3-Embedding-0.6B). A chat model outputs a distribution over the next token.
One is for comparing, the other for generating.

**2. Why encode queries and documents differently?**
Qwen3-Embedding is **instruction-aware**: queries carry a task instruction,
documents do not. It mainly improves the **margin** between the right answer and
the rest, which is what makes retrieval robust as the corpus grows.

**3. Three query types where keyword search wins.**
Exact identifiers ("invoice 88213", a function name), **negation** ("notes that
do *not* mention the exam"), numbers and date ranges. Also rare proper nouns the
model never saw.

**4. Why is "the index always returns k results" dangerous?**
Because **nearest ≠ relevant**. With nothing matching, you still get your *k*
least-irrelevant chunks, at respectable-looking scores. Feed those to a model
without a **score threshold** and you have built a hallucination machine.

---

## 09 · RAG

**1. What are the four stages of RAG?**
Chunk → embed/index → retrieve → generate from the retrieved text.

**2. Why measure retrieval separately from answer quality?**
They fail for **different reasons and have different fixes**. If retrieval fails,
no prompt engineering can help — the model never received the text. Measuring
only the final answer tells you something is wrong but not which half.

**3. What does recall@k measure, and what do you do when it is low?**
The fraction of queries where **a relevant chunk appears in the top k**. When
low: smaller or larger chunks, more overlap, bigger `k`, hybrid search, query
rewriting, a reranker. ✅ Fix retrieval *before* touching the prompt.

**4. Why is a score threshold necessary?**
See 08-4. Without a floor, an unanswerable question still returns *k* chunks and
invites the model to invent. The threshold is what lets the system **refuse**.

**5. When would you fine-tune instead of using RAG?**
When you need **behaviour**, not knowledge: a consistent tone, a house format, a
domain register that prompting cannot hold. Never for facts — those change, and
fine-tuned facts cannot be cited, updated or removed.

---

## 10 · Agents

**1. Write the agent loop from memory, in four steps.**
① **Think** — model reads the transcript. ② If it requested a tool, **act** —
*your code* runs it. ③ **Observe** — append the result. ④ Repeat until it answers
without calling a tool, **or `max_steps` is hit**. *(Missing the step limit is
the answer to mark down.)*

**2. Difference between a chain and an agent?**
In a **chain you decide the steps**; in an **agent the model decides** how many
and which. You trade predictability for flexibility. Most production "agents"
should have been chains — if you can draw the flowchart, write the flowchart.

**3. Name the five failure modes.**
① infinite loop ② wrong tool, confidently ③ ignoring the tool result
④ cascading errors from an early mistake ⑤ prompt injection.

**4. Why does `stop=["Observation:"]` matter in ReAct?**
Without it the model **writes its own Observation** — inventing the tool result
and reasoning from fiction. It has no concept of its turn ending; it just
continues the document. Native tool calling removes this failure structurally.

**5. Two defences against prompt injection that do not rely on the model.**
Any two of: **least privilege** (no send tool → nothing to exfiltrate with),
**confirmation gates** on irreversible actions, **trust separation** (retrieved
text in a user message, never the system prompt), **output filtering**,
**logging**. ✅ "Telling the model to ignore instructions in documents" is *not*
one — mark that down explicitly.

---

## 11 · Fine-tuning

**1. Give the five-step decision list before fine-tuning.**
① Can a better **prompt** fix it? ② Is it missing **knowledge** → RAG.
③ Need **structured output** → constrained decoding. ④ Need it to **do things**
→ tools. ⑤ Need a consistent **style/format** prompting cannot hold → *now*
fine-tune. Stop at the first yes.

**2. What does LoRA learn, and why is it so much cheaper?**
Two thin matrices `B` and `A` whose product is added to the frozen weights:
`W' = W + (α/r)·B@A`. At rank 16 that is **well under 1%** of the parameters, so
gradients and optimiser state shrink proportionally. Full fine-tuning of an 8B
model needs ~96 GB; LoRA fits on a consumer GPU, and the adapter is a few MB.

**3. How would you detect overfitting on 12 examples?**
Loss collapsing to near zero within a few steps; the model reproducing training
responses verbatim; good behaviour on the 12 and poor behaviour on anything
else. ✅ With 12 examples you *will* memorise — the question is whether you
noticed.

**4. Why measure the baseline before training?**
Otherwise you cannot attribute any change to the training. The base model may
already do the task, or the improvement may be smaller than run-to-run variance.
This generalises well past fine-tuning.

**5. When do you merge an adapter, and when keep it separate?**
**Merge** for deployment as a standalone model — no PEFT dependency, no
inference overhead, and it can be converted to GGUF. **Keep separate** when you
want several behaviours from one base model, hot-swapping adapters against a
single loaded backbone.
