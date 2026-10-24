---
title: Glossary
parent: Guides
nav_order: 7
---

# Glossary

Every term used in this workshop, defined plainly. Terms you will meet on day 1
are marked ⭐.

**Lost?** Start with the plain-language table below, then look the word up in
the A–Z. Every analogy here is a crutch, not a definition — where one is
actively misleading, the entry says so.

1. TOC
{:toc}

---

## In plain terms

### Models and generation

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Token** ⭐ | Syllables, not letters. The model reads and writes in chunks like `straw` + `berry`. | Chunks are learned from data, so they are not real syllables — and they are much coarser in Kazakh than in English. |
| **Context window** ⭐ | A desk. Only so much paper fits; put a new page on and an old one falls off the back. | Nothing "falls off" by itself — *your code* decides what to drop (notebook 17). |
| **Embedding** | A shelf position in a library. Books on similar topics sit near each other, so "near" means "about the same thing". | Nearness is about usage, not truth. "Always" and "never" sit close together. |
| **Temperature** ⭐ | How closely a cook follows the recipe. 0 = exactly; 1 = improvising. | It does not make the model more *creative*, only less predictable. |
| **top-p / top-k** | A shortlist before choosing. Only the plausible next words stay on the list. | — |
| **Perplexity** | How torn the model is. Perplexity 5 ≈ "it was choosing between about 5 options". | Comparable only within one tokenizer — never across models with different vocabularies. |
| **Hallucination** ⭐ | A confident student who did not do the reading. Fluent, plausible, wrong. | It is not lying — there is no inner knowledge of being wrong to conceal. |
| **Thinking mode** | Working on scratch paper before writing the final answer. | The scratch paper is also generated text and can be wrong too. |
| **Quantization** | Rounding every price to the nearest tenge to save space on the receipt. | Rounding the *weights*, not the prices — and some weights matter far more than others. |
| **Base vs instruct model** ⭐ | Someone extremely well-read who was never taught to answer questions, versus someone who was. | — |

### Giving the model knowledge and skills

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **RAG** ⭐ | An open-book exam. Look it up, then answer, and cite the page. | Only as good as whether the right page was found (see *recall@k*). |
| **Chunking** | Cutting a book into index cards so you can pull just the relevant ones. | Cut badly and a fact ends up split across two cards, so neither answers the question. |
| **Recall@k** | Of the `k` index cards you pulled out, was the right one among them? | Says nothing about whether the model then *used* it. |
| **Reranker** | A second, pickier reader who re-sorts the shortlist you pulled. | — |
| **Fine-tuning** ⭐ | Teaching an experienced new hire your house style. | It teaches *form*, not *facts*. For facts, use RAG. |
| **LoRA** | Sticky notes on a manual instead of reprinting the manual. | The base manual is unchanged, so you can peel the notes off — or swap in a different set. |
| **System prompt** ⭐ | The job description, handed over before the first customer. | Not a security boundary. Retrieved text can argue with it (*prompt injection*). |

### How models are trained

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Pre-training** | School. Years of reading everything, with no particular job in mind. | — |
| **Post-training** | The first month on the job: how we answer here, what we don't say. | — |
| **Loss** | Your score on a practice test, where **lower is better**. | — |
| **Gradient** | Which way to turn each dial, and how hard, to score better next time. | — |
| **Learning rate** | Stride length. Too short and you never arrive; too long and you stride past the door. | — |
| **Epoch** | One full pass through the textbook. | — |
| **Overfitting** ⭐ | Memorising past exam papers. Perfect on those, lost on the real exam. | — |
| **Early stopping** | Handing in the exam when you stop improving, instead of fiddling until the bell. | — |
| **Loss masking** | Marking only the answers, not the questions. The question is context, not something to imitate. | — |
| **Preference pair** | "This answer was better than that one" — easier to judge than writing a good answer yourself. | — |
| **RLHF / DPO** | Training on *comparisons* instead of *examples*. DPO is the cheap way to do it. | Optimises what annotators **picked**, which is a proxy for good — and the gap is where sycophancy comes from. |

### Agents

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Chain** ⭐ | A recipe. Same steps, same order, every time. | — |
| **Agent** ⭐ | A cook told "make dinner", with a fridge and free rein. | The cook has no memory of yesterday unless you leave a note. |
| **Tool calling** ⭐ | The model fills in a request form. **Your code** decides whether to act on it. | The model never runs anything itself — which is why agents can be made safe at all. |
| **ReAct** | Thinking aloud: "I need the date → I'll check the calendar → it says Tuesday → so…". | — |
| **Plan-then-execute** | A trip itinerary written before you leave, versus asking directions at each corner. | A plan written before you set off was written in ignorance — hence replanning. |
| **Router** | The receptionist at a clinic. Listens for ten seconds, sends you to the right specialist. | Decides **once**, then steps out of the way. That is why it is not an agent. |
| **Orchestrator** | A wedding planner. Asks the florist, the caterer and the photographer each for their piece, then assembles the day. | You pay all three whether or not you use what they say. |
| **Pipeline** | A passport office. Desk 1 → desk 2 → desk 3, and nobody sees your whole case. | Detail is lost at every desk, which is why nobody can answer "why was this rejected?". |
| **Critic loop** | A writer and an editor. Draft, red pen, revise, until the editor signs it off. | Only works if the editor can actually say **no** — and if you cap the rounds. |
| **Handoff** | Repeating a story to the next person. Each retelling loses detail. | — |
| **Scratchpad** | The back of an envelope while cooking. Binned afterwards. | — |
| **Fact store** | The sticky note on the fridge. Deliberate, few, and still there next month. | Somebody has to decide what goes on the fridge — and a wrong note stays wrong. |
| **Sliding window** | A desk that holds ten pages; the oldest falls off the back. | The oldest page is usually the one where the job was explained. |
| **Context compaction** | Meeting minutes replacing the recording. | Whoever wrote the minutes decided what mattered. |
| **Trajectory** ⭐ | A student's working, not just the final number. | — |
| **Outcome** ⭐ | The final number on the exam. | Right for the wrong reason looks identical to right. |
| **Lucky pass** | Arriving on time having run three red lights. | — |
| **Guardrail** ⭐ | The handrail on the stairs, not the sign asking you to hold it. | A rule in the prompt is the sign. A tool the agent does not have is the handrail. |
| **Prompt injection** ⭐ | A note slipped into a document that the reader mistakes for instructions from their boss. | The model has no way to tell your instructions from text it read. |
| **max_steps** | The taxi meter limit you agree before setting off. | — |
| **pass@k vs pass^k** | "Got it right at least once in 5 tries" vs "got it right all 5 times". | For anything that sends email or moves files, only the second one counts. |
| **Harness** / scaffolding | The cart, reins and traces. The horse supplies the power; the harness is what makes it pull something. | Swap the harness and the same horse pulls a different load — which is why benchmark scores are a property of both, never of the model alone. |
| **MCP** | USB-C for tools. Write the adapter once; every client can use it. | — |

---

## A to Z

**Active parameters** — How many parameters actually run for each token, as
opposed to how many are stored. Identical to the total in a dense model; far
smaller in an *MoE* one, where `Qwen3-30B-A3B` means 30B total and 3B active.
**Total parameters predict memory; active parameters predict speed and serving
cost; neither predicts quality.** A comparison that quotes one without saying
which is not a comparison. See
[handbook chapter 8](../handbook/who-builds-them.html#total-parameters-and-active-parameters).

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

**Blocked (task)** — A planned task that **never ran** because something it depended on failed. Distinct from *failed*, which ran and broke. Conflating the two sends you debugging a task that did nothing wrong.

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

**Context compaction** — Replacing older turns with a summary to fit the *context window*. Meeting minutes replacing the recording — and whoever wrote the minutes decided what mattered. A summariser must fold the previous summary in, or the summary grows until it costs more than it saves.

**Context window** ⭐ — The maximum number of tokens the model can attend to at
once. Prompt + generation must fit.

**Cosine similarity** ⭐ — The angle between two vectors, used to compare
embeddings. For normalised vectors it is just the dot product.

**Critic loop** — A writer agent and a critic agent, alternating until the critic approves or a round cap is hit. A writer and an editor. The one multi-agent pattern where losing context is the *point*: a critic who has not seen the reasoning judges the output as a reader will. Needs an explicit accept token — "looks good, although…" is not approval.

**Cross-encoder** — A model that reads a query and document *together* and
scores their relevance. Slower and more accurate than comparing embeddings. See
*reranker*.

**Decoding** ⭐ — Turning the model's probability distribution into actual tokens.
Governed by temperature, top-p, top-k. See notebook 04.

**Dense model** — A model where every parameter is used for every token, as
opposed to *MoE*.

**DPO (Direct Preference Optimization)** — Training on *(prompt, better answer, worse answer)* triples, with no reward model and no RL loop. The cheap, stable way to do what *RLHF* does. See notebook 15.

**Early stopping** — Keeping the checkpoint where validation loss bottomed, rather than the last one. Handing in the exam when you stop improving instead of fiddling until the bell. Requires `load_best_model_at_end` — detection without retention is worthless.

**Embedding** ⭐ — A fixed-length vector representing a piece of text, where
distance corresponds to meaning.

**Epoch** — One full pass over the training data. One pass through the textbook. For LoRA on a small set, 2–3; for *DPO*, 1.

**Fact store** — Durable, explicitly chosen facts an assistant keeps between sessions. The sticky note on the fridge. The only memory strategy that survives a restart and the only one you can read and correct.

**Failed (task)** — A planned task that ran and raised. See *blocked*.

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

**Gradient** — Which way to turn each parameter, and how hard, to reduce the *loss*. A *gradient norm* spike usually means a batch unlike the others — bad data, not a bad model.

**Gradient clipping** — Capping the size of a parameter update so one pathological batch cannot wreck a run. A seatbelt for rare crashes — **not** a fix for a learning rate that is too high.

**Greedy decoding** — Always take the highest-probability token. Deterministic,
and prone to repetition. Not recommended for Qwen3 thinking mode.

**Guardrail** — A constraint limiting what a model or agent can do: a step
limit, an allow-list, a confirmation gate.

**Hallucination** ⭐ — Confidently stating something false. Not a bug to be fixed
but a property to be managed — with retrieval, citations and refusals.

**Handoff** — One agent passing work to another. Repeating a story to the next person: each retelling loses detail. The cost nobody counts in a multi-agent design.

**Harness** — Everything around the model that turns "predict the next token"
into useful work. From *test harness* in software, and before that a horse's
harness: the model is the engine, the harness is the gear that connects it to
the load. Two kinds, often confused:

* An **evaluation harness** turns a model into a *number*. It owns the prompt
  format, the few-shot examples, how an answer is extracted from free text, the
  scoring rule and the aggregation. `lm-evaluation-harness` is the standard one.
* An **agent harness** — also called **scaffolding** — turns a model into a
  *system that acts*: the loop, the tool definitions, the parser, the retries,
  the memory, the context assembly, the stopping condition. Notebooks 17–20
  build one from nothing.

**The same weights score differently under different harnesses, often by more
than two models differ from each other.** Change the answer-extraction rule and
a benchmark moves points; add a retry-on-parse-failure and "capability" rises
with no weight changed. So "model X beats model Y" means nothing until you ask
*under whose harness?* — and your own comparisons are only valid with the
harness held fixed. It also means most agent failures are harness bugs, not
model limitations: notebook 10's hallucinated `Observation:` is a parser that
read too far, not a model that lied.

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

**Learning rate** — How big a step training takes each update. Stride length. The single most consequential hyperparameter; find it with a range test rather than a guess.

**llama.cpp** ⭐ — The C++ inference engine underneath Ollama and LM Studio.

**LoRA** — Low-Rank Adaptation. Freezes the base weights and learns two thin
matrices per layer. Trains <1% of the parameters. See notebook 11.

**Loss** — How wrong the model was, averaged. **Lower is better.** An untrained model scores `ln(vocabulary size)` — about 11.9 for Qwen3 — and a run sitting there is a bug, not slow learning.

**Loss masking** — Choosing which tokens the loss is computed on. Marking only the answers, not the questions. The usual default is completion-only: the prompt is context to condition on, not behaviour to imitate.

**Lucky pass** — A test case where the answer is right and the *trajectory* is wrong. Arriving on time having run three red lights. The count of these is the gap between what you measured and what you have.

**MCP (Model Context Protocol)** — A standard for exposing tools to any model.
Write a tool once, use it from any client.

**min-p** — A sampling filter keeping tokens above a fraction of the top token's
probability. Robust at high temperature.

**MoE (Mixture of Experts)** — Only a subset of parameters ("experts") is used
per token. Qwen3-30B-A3B has 30B total but ~3B active: memory of a large model,
speed closer to a small one. See *active parameters*, and
[handbook chapter 8](../handbook/who-builds-them.html#total-parameters-and-active-parameters).

**Open weights** ⭐ — The trained parameters are published; the training data
usually is not. Distinct from *open source*. See
[Open-weight models](open-weight-models.html).

**Orchestrator** — An agent that fans subtasks out to specialists and synthesises their results. A wedding planner briefing the florist, caterer and photographer. Costs roughly three times a single agent, and you pay every specialist whether or not their output is used. Only worth it when the subtasks are genuinely independent.

**Outcome evaluation** — Checking whether the final answer is right. Necessary and not sufficient for an agent — see *trajectory evaluation*.

**Overfitting** — When a model fits its training data at the expense of everything else. Memorising past exam papers. Visible as training loss falling while validation loss rises; the fix is more data, then fewer steps.

**PagedAttention** — vLLM's technique for managing KV cache memory in pages,
enabling high-concurrency serving.

**pass@k / pass^k** — Over `k` runs: did it succeed **at least once** (`pass@k`), or **every time** (`pass^k`)? For an agent that can send a message or move a file, only `pass^k` decides whether you ship.

**PEFT** — Parameter-Efficient Fine-Tuning. The Hugging Face library containing
LoRA and friends.

**Perplexity** — A measure of how surprised a model is by text. Lower is better.
Useful for comparing quantization levels; a poor proxy for usefulness.

**Pipeline (multi-agent)** — Agents chained in a fixed order, each seeing only the previous one's output. A passport office: desk 1, desk 2, desk 3, and nobody sees the whole case.

**Plan-then-execute** — Writing all the steps before running any of them, as a dependency graph. A trip itinerary versus asking directions at each corner. Gains parallelism and a review point; loses the ability to adapt without replanning.

**Post-training** — Everything after *pre-training*: *SFT*, then preference tuning (*RLHF* / *DPO*). The first month on the job. This is where behaviour comes from; pre-training is where knowledge comes from.

**Pre-training** — The first and longest training phase: next-token prediction over a very large corpus. School. Produces a *base model*.

**Preference pair** — A prompt with a better and a worse answer. People are far more consistent at **picking** than at **writing**, which is the whole argument for collecting them.

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

**Replanning** — Replacing the failed and blocked parts of a plan while keeping the completed work. Re-running what already succeeded is slower and can loop forever.

**Reranker** — A cross-encoder that re-scores retrieved candidates. Retrieve 20
cheaply, rerank to 3 accurately.

**Router** — An agent that picks which specialist handles a task, then steps out of the way. The receptionist at a clinic. Two model calls, the cheapest useful multi-agent pattern — and because it decides **once**, it is not itself an agent.

**Safetensors** ⭐ — A weight file format that is pure data and cannot execute
code. Always prefer it over the legacy `.bin` pickle format.

**Sampling** ⭐ — Choosing the next token from the probability distribution. See
*decoding*.

**Scratchpad** — Working notes kept during a single task and discarded afterwards. The back of an envelope while cooking. Writing failures down here is the cheapest way to stop an agent repeating a call it was already told would fail.

**Scripted model** — A stand-in that replays written-down decisions instead of calling a real model, so agent logic can be tested without a server. `qwen_workshop.scripted`. It verifies **your code**, never whether a real model would decide the same way.

**SFT (Supervised Fine-Tuning)** — Training on input/output pairs. What
notebook 11 does.

**Sliding window** — Keeping only the newest turns that fit a token budget. A desk that holds ten pages, with the oldest falling off the back — and the oldest is usually where the task was explained.

**Specialist** — A named agent with a description a *router* can read. The description is the routing signal, not documentation: "handles data" routes nothing.

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

**Trajectory evaluation** — Checking **how** an agent reached its answer: which tools, in what order, and whether it did anything forbidden. A student's working, not just the final number. Catches the sent email that *outcome evaluation* reports as a pass.

**Transformer** — The neural architecture behind essentially all current LLMs.

**Vector database** — Storage and search for embeddings. For under ~10k chunks,
a NumPy array is genuinely the right tool.

**vLLM** ⭐ — A high-throughput inference server for GPUs. Continuous batching and
PagedAttention.

**Wave (in a plan)** — A set of tasks whose dependencies are all satisfied, so they can run together. The number of waves is the shortest a plan can possibly take.
