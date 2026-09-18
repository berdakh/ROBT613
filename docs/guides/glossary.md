---
title: Glossary
parent: Guides
nav_order: 10
---

# Glossary

Every term used in this workshop, defined plainly. Terms you will meet on day 1
are marked ⭐.

---

**Adapter** — A small set of trained weights added to a frozen base model. See
*LoRA*. A few megabytes rather than a few gigabytes.

**Agent** ⭐ — A model in a loop with tools, deciding for itself what to do next.
Distinguished from a *chain*, where you decide the steps. See
[Agentic AI](agentic-ai.html).

**Alignment** — Training that makes a model helpful and safe, typically after
pretraining. Usually *SFT* followed by *RLHF* or *DPO*.

**Attention** — The mechanism letting each token's representation depend on
other tokens. The core operation of a transformer.

**AWQ** — Activation-aware Weight Quantization. A 4-bit scheme for GPU serving.

**Base model** ⭐ — A pretrained model that has *not* been instruction-tuned. It
continues text rather than following instructions and has no chat template. If
your "chat" model rambles, check you did not download a `-Base` checkpoint.

**bf16 / brain float 16** — A 16-bit float with the same exponent range as fp32.
The standard precision for modern LLM inference and training.

**bitsandbytes** — Library that loads transformers models in 8-bit or 4-bit on
NVIDIA GPUs.

**BM25** — A classical keyword-ranking algorithm. Still beats embeddings for
exact terms, names and identifiers; the keyword half of *hybrid search*.

**Catastrophic forgetting** — When fine-tuning on a narrow task makes a model
worse at everything else.

**Chain** ⭐ — A fixed sequence of steps you define. Predictable and testable.
Most production "agents" should have been chains.

**Chat template** ⭐ — The Jinja template that turns a list of messages into the
single string the model actually receives. Stored in `tokenizer_config.json`.
**Print it when debugging.**

**Constrained decoding** — Masking out tokens that cannot continue a valid
output under a grammar or JSON schema. Makes invalid output *impossible* rather
than merely unlikely.

**Context window** ⭐ — The maximum number of tokens the model can attend to at
once. Prompt + generation must fit.

**Cosine similarity** ⭐ — The angle between two vectors, used to compare
embeddings. For normalised vectors it is just the dot product.

**Cross-encoder** — A model that reads a query and document *together* and
scores their relevance. Slower and more accurate than comparing embeddings. See
*reranker*.

**Decoding** ⭐ — Turning the model's probability distribution into actual tokens.
Governed by temperature, top-p, top-k. See notebook 04.

**Dense model** — A model where every parameter is used for every token, as
opposed to *MoE*.

**Embedding** ⭐ — A fixed-length vector representing a piece of text, where
distance corresponds to meaning.

**Few-shot prompting** ⭐ — Including worked examples in the prompt. Usually the
cheapest large improvement available.

**Fine-tuning** — Further training on your own data to change behaviour. Good
for style and format; bad for injecting facts (use *RAG*).

**finish_reason** ⭐ — Why generation stopped. `stop` is natural; **`length`
means truncated** and your output may be incomplete.

**Flash Attention** — A memory-efficient attention implementation. Faster, same
mathematics.

**GGUF** ⭐ — llama.cpp's single-file model format. The standard for CPU and
Apple Silicon inference.

**Greedy decoding** — Always take the highest-probability token. Deterministic,
and prone to repetition. Not recommended for Qwen3 thinking mode.

**Guardrail** — A constraint limiting what a model or agent can do: a step
limit, an allow-list, a confirmation gate.

**Hallucination** ⭐ — Confidently stating something false. Not a bug to be fixed
but a property to be managed — with retrieval, citations and refusals.

**Hybrid search** — Combining semantic (embedding) and keyword (BM25) retrieval.
Almost always better than either alone.

**Inference** — Running a trained model. As opposed to training.

**Instruction tuning** — Fine-tuning a base model to follow instructions,
producing an "instruct" or "chat" model.

**KV cache** ⭐ — Cached keys and values from previous tokens, so each new token
does not recompute the whole sequence. Grows with conversation length and can
exceed the size of the weights at long context.

**Latency vs. throughput** — Latency is how fast one answer arrives; throughput
is how many answers per second across all users. Batching improves throughput,
not latency.

**llama.cpp** ⭐ — The C++ inference engine underneath Ollama and LM Studio.

**LoRA** — Low-Rank Adaptation. Freezes the base weights and learns two thin
matrices per layer. Trains <1% of the parameters. See notebook 11.

**MCP (Model Context Protocol)** — A standard for exposing tools to any model.
Write a tool once, use it from any client.

**min-p** — A sampling filter keeping tokens above a fraction of the top token's
probability. Robust at high temperature.

**MoE (Mixture of Experts)** — Only a subset of parameters ("experts") is used
per token. Qwen3-30B-A3B has 30B total but ~3B active: memory of a large model,
speed closer to a small one.

**Open weights** ⭐ — The trained parameters are published; the training data
usually is not. Distinct from *open source*. See
[Open-weight models](open-weight-models.html).

**PagedAttention** — vLLM's technique for managing KV cache memory in pages,
enabling high-concurrency serving.

**PEFT** — Parameter-Efficient Fine-Tuning. The Hugging Face library containing
LoRA and friends.

**Perplexity** — A measure of how surprised a model is by text. Lower is better.
Useful for comparing quantization levels; a poor proxy for usefulness.

**Prompt injection** ⭐ — Hostile instructions hidden in content the model reads.
The main security threat to agents. Defend with least privilege, not with
politely asking the model to ignore it.

**Quantization** ⭐ — Storing weights in fewer bits. Smaller and faster to load,
slightly worse quality. See notebook 05.

**RAG (Retrieval-Augmented Generation)** ⭐ — Retrieve relevant documents, then
generate an answer from them. The highest-value technique in this workshop.

**ReAct** — *Reason + Act*. An agent pattern using a text protocol
(Thought/Action/Observation). Predates native tool calling and still works with
any model.

**Recall@k** ⭐ — The fraction of queries for which a relevant document appears in
the top *k* results. The number to measure when debugging retrieval.

**Reranker** — A cross-encoder that re-scores retrieved candidates. Retrieve 20
cheaply, rerank to 3 accurately.

**Safetensors** ⭐ — A weight file format that is pure data and cannot execute
code. Always prefer it over the legacy `.bin` pickle format.

**Sampling** ⭐ — Choosing the next token from the probability distribution. See
*decoding*.

**SFT (Supervised Fine-Tuning)** — Training on input/output pairs. What
notebook 11 does.

**System prompt** ⭐ — Instructions given the `system` role, setting the model's
behaviour for the conversation. The cheapest high-leverage control you have.

**Temperature** ⭐ — Rescales the logits before sampling. Low = focused and
repetitive; high = diverse and eventually incoherent. Does not make the model
smarter.

**Thinking mode** ⭐ — Qwen3's reasoning block, emitted between `<think>` and
`</think>` before the answer. Costs tokens, buys accuracy on multi-step problems.

**Token** ⭐ — The unit a model reads and writes. Roughly 4 characters in English,
fewer in most other scripts. Not a word and not a letter — which is why counting
letters is hard.

**Tokenizer** ⭐ — Converts text to token ids and back. Also holds the chat
template.

**Tool calling** ⭐ — The model requests a function call; **your code** executes
it and returns the result. The model never runs anything itself.

**top-k** — Keep only the *k* most likely tokens. Fixed width, ignores how
confident the model is.

**top-p (nucleus sampling)** ⭐ — Keep the smallest set of tokens whose
probabilities sum to *p*. Adapts to the model's confidence, which is why it is
usually preferred to top-k.

**Transformer** — The neural architecture behind essentially all current LLMs.

**vLLM** ⭐ — A high-throughput inference server for GPUs. Continuous batching and
PagedAttention.

**Vector database** — Storage and search for embeddings. For under ~10k chunks,
a NumPy array is genuinely the right tool.
