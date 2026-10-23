# Checkpoint answers

The 127 questions that close notebooks 01–11, the training (13–16) and agents
(17–20) tracks, the mathematics deep dive (21) and the Kazakh/Russian material
(22). Use them as an oral check, a quick written quiz, or a revision sheet.

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

**1. Define an agent in one sentence, then name the four properties.**
A program in which a **model chooses the next action** from a set of available
actions, **your code executes** them, and the results are fed back so the model
can choose again — until it decides to stop. The four properties: a **goal**,
**actions**, a **loop**, and **autonomy over the path**.

**2. Which property separates an agent from a chain?**
**Autonomy over the path.** In a chain your code decides which step runs next;
in an agent the model does. Remove that one property and everything else stays
the same — which is why most production "agents" are chains with extra cost.

**3. Write the agent loop from memory, in four steps.**
① **Think** — model reads the transcript. ② If it requested a tool, **act** —
*your code* runs it. ③ **Observe** — append the result. ④ Repeat until it
answers without calling a tool, **or `max_steps` is hit**. *(Missing the step
limit is the answer to mark down.)*

**4. Who executes the tool call, and why does it matter?**
**Your code does.** The model only emits a *request* — a name and some JSON.
It matters because that gap is where every guardrail lives: refusing a tool,
requiring confirmation, containing a path, logging the attempt. An agent is
safe-able precisely because the model cannot run anything itself.

**5. What does a scripted model let you test, and what can it never tell you?**
It tests **your code**: that tool results reach the model, that `max_steps`
holds, that errors come back as observations, that the registry was offered at
all. It can **never** tell you whether a real model would make those decisions.
A green scripted suite means your loop is correct, not that your agent works.

**6. Name the six failure modes. Which two are invisible if you only check the
final answer?**
① infinite loop ② ignoring the tool result ③ wrong tool, confidently
④ answering without looking ⑤ cascading errors ⑥ prompt injection.
**④ and ⑥ are invisible from the answer alone** — ④ produces a plausible
number, and ⑥ produced a perfectly innocent "bread and milk" while
exfiltrating the budget. ② is often invisible too; accept it as a third.

**7. Why does `stop=["Observation:"]` matter in ReAct?**
Without it the model **writes its own Observation** — inventing the tool result
and reasoning from fiction. It has no concept of its turn ending; it just
continues the document. Native tool calling removes this failure structurally.

**8. Two defences against prompt injection that do not rely on the model.**
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

---

## 13 · Pre-training a tiny model

**1. What is the training objective, in one sentence?**
Given every prefix of the text, predict the **next token**, and minimise the
cross-entropy between the predicted distribution and the token that actually
came next. That is the entire objective of pre-training, at every scale.

**2. Why does one sequence of length `n` give `n` training signals?**
Because the causal mask lets the model make a prediction at **every position
simultaneously** — position `i` predicts token `i+1` using only tokens `0..i`.
One forward pass therefore produces `n` predictions and `n` loss terms, which is
why transformers train so much more efficiently than the RNNs they replaced.

**3. What does the causal mask prevent, and what would happen without it?**
It prevents position `i` from attending to positions after it. Without it, the
model can see the token it is being asked to predict: training loss collapses to
near zero within a few steps and the model generates nonsense, because it
learned to copy an answer that is not there at inference time. ✅ **Loss that
drops impossibly fast is the classic signature of a broken mask.**

**4. What loss does an untrained model have, and why that number?**
`ln(vocab_size)`. A model with no information assigns uniform probability
`1/V` to every token, and `−ln(1/V) = ln(V)`. For this notebook's character
vocabulary that is about **4.6**; for Qwen3's 151,936 tokens it is **11.9**.
Knowing this number is what lets you tell "slow start" from "not training".

**5. What does rising validation loss with falling training loss mean?**
**Memorisation.** The model is fitting the training text specifically rather
than learning the structure of the language, so the optimum is the step where
validation bottomed — everything after that is damage. Notebook 16 turns this
into `early_stop`.

**6. Name three things this model lacks that an instruct model has.**
Any three of: **instruction tuning** (it continues text, it does not answer),
**a chat template** (no roles, no turn boundaries), **preference tuning** (no
notion of a better answer), **scale** (~800k parameters against billions),
**a real tokenizer** (characters, not subwords), **safety behaviour**, and
**any knowledge not in a few hundred KB of corpus**. The valuable observation is
that only the first three are *training phases* — the rest are resources.

---

## 14 · Building a training dataset

**1. What shape does a fine-tuning example take, and why `.jsonl`?**
`{"messages": [{"role": ..., "content": ...}, ...]}` — the same chat structure
used at inference, so training matches serving. `.jsonl` because one JSON object
per line means you can stream a file larger than memory, append without
rewriting, and `wc -l` it; a single top-level JSON array gives up all three.

**2. What is loss masking, and which scheme is the usual default?**
Marking which tokens the loss is computed on. The usual default is
**completion-only**: the system and user tokens are labelled `-100` so they are
read as context but never graded, and only the assistant tokens contribute to
the loss. Grading the prompt teaches the model to generate plausible user
questions — a skill nobody wanted.

**3. Why are near-duplicates worse than exact duplicates?**
Exact duplicates are trivially detectable by hashing, so they get removed.
Near-duplicates survive every naive check, so they quietly over-weight whatever
they have in common — and if one lands in train and its twin in eval, your
evaluation is measuring memorisation and reporting it as generalisation.

**4. Why must you deduplicate *before* splitting?**
Deduplicating afterwards leaves a copy on each side of the split. The leak is
then invisible: both files pass their own dedup check, and the evaluation score
is inflated by exactly the examples the model has already seen.

**5. Name three distributional problems you would look for before training.**
Any three of: **length imbalance** (one cluster of very long answers teaches
length), **topic concentration** (half the set about one subject), **format
drift** (markdown in some answers, none in others), **role errors** (examples
not ending on an assistant turn), **near-duplicate clusters**, **systematic tone
differences** between batches written by different people.

**6. Fifty careful examples or a thousand scraped ones — which, and why?**
The fifty, in almost every case. Every inconsistency in the data is an
instruction to be inconsistent, and a scraped set has inconsistencies you have
not read. The practical test is the one in the notebook: **if you cannot explain
why each example is in the set, it probably should not be.**

---

## 15 · Preference tuning with DPO

**1. What can preference data teach that a demonstration cannot?**
*Comparative* judgement — that one fluent answer is better than another fluent
answer. Demonstrations say "do this"; none says "not that", and none carries a
degree. Preferences are also far cheaper to collect reliably, because picking
between two answers is much easier than writing a good one.

**2. Why must SFT come first?**
Preference tuning only **reweights behaviour the model can already produce**.
The loss moves probability between two candidate answers; it cannot create a
capability that was never there. Running DPO on a base model gives you a base
model with slightly different preferences about its own rambling.

**3. Why is 0.693 the number to look for at step 0?**
Because at step 0 the policy equals the reference, so every margin is exactly 0
and the loss is `−log σ(0) = log 2 ≈ 0.693` on every pair. ✅ **A DPO run that
does not start near 0.693 has a bug** — usually a wrong reference model or a
masking error — and that is worth catching in the first ten seconds rather than
after the run.

**4. What is the reference model's job, and what goes wrong without it?**
It makes the loss measure **change** rather than absolute log-probability, which
is how the KL constraint from the RLHF objective survives into DPO. Without it,
a pair the reference already ranked correctly looks like a solved problem the
policy deserves credit for, and a pair the policy genuinely fixed looks like no
progress — the notebook shows the unanchored loss giving the opposite verdict on
both.

**5. Your run shows rising margins and sinking reward levels. Diagnose it.**
**Degenerate drift.** The margin is a difference, so it can be opened by pushing
the rejected answer down rather than the chosen answer up — and both can sink
together. The model is becoming less likely to produce either answer, which
shows up as bland, hedging, interchangeable output. Lower the learning rate,
lower β, and inspect the pairs: if `chosen` and `rejected` are both poor, the
data is the problem.

**6. Chosen answers average twice the length of rejected ones. What will the
model learn, and what do you do about it?**
It will learn **length**, because length is the most reliable signal in the set
and the model has no way to separate it from quality. Fix it in the data:
deliberately write pairs where the chosen answer is the shorter one, balance the
distribution, and check it at the set level rather than per pair. Length
normalisation (or SimPO) helps, but it is a patch over a data problem.

**7. Why is a general eval set non-negotiable here?**
Because preference tuning's characteristic failure is a model that is better at
exactly what you measured and worse at everything else. Only an evaluation set
with **no relationship to your preference data** can see that, and by the time
users report it the run is already deployed.

---

## 16 · Training diagnostics

**1. Name the four numbers and what each one rules out.**
**Train loss** — is it learning at all. **Validation loss** — is it
generalising, and the only one you may make decisions from. **Learning rate** —
is the step size sane, and schedulers change it under you. **Gradient norm** —
is anything exploding; it is usually boring, which is exactly what makes a spike
informative.

**2. What is the chance loss for Qwen3, and what does a run that sits there
mean?**
`ln(151936) ≈ 11.9`. A run sitting there is **not learning slowly, it is
broken** — the usual causes are a learning rate near zero, a label or masking
bug that grades nothing, or frozen parameters. Check before you wait.

**3. Your train loss falls 96% and the model gets worse. Where do you look?**
At the **validation curve**, which will have bottomed early and climbed since.
Then at the step where it bottomed — that checkpoint is the model you wanted.
✅ In the notebook's run, train loss falls 0.976 → 0.036 while validation goes
0.354 (step 300) → 0.689 (step 3000).

**4. Why did clipping at 1.0 fail to rescue the `lr=30` run?**
Because the gradients were never large. Every logged norm stayed below 1.0 and
the clip fired on only 52 of 401 steps. The failure was a **sane gradient times
an insane learning rate**, and no threshold on the gradient can detect that.
Clipping at 0.1 "worked" only by firing on a third of the steps, which is a
step-size cap doing the learning rate's job badly.

**5. What is gradient clipping actually for, then?**
**Rare, catastrophic batches.** One abnormal example, one full-size step, and a
healthy run is damaged. In the notebook a single bad batch (norm 24.5 against a
usual 0.10) costs 0.34 of validation loss unclipped and 0.01 clipped. Clipping
bounds the damage from a batch unlike the others; it does not tune anything.

**6. Describe the range test, and the rule for picking from its output.**
Train briefly at logarithmically spaced learning rates and find where it breaks.
The rule of thumb is the **largest stable rate divided by two or three** — but
use it to pick the *range*, then choose inside it on **validation** loss. In the
notebook's sweep `lr=3` reaches a much lower train loss than `lr=0.3` and a much
worse validation loss, and `lr=0.3` is the right answer.

**7. Why is `load_best_model_at_end=True` not optional?**
Because without it early stopping detects the problem and then hands you the
damaged final checkpoint anyway. Detection without retention is worthless. The
related trap: patience counts **evaluations**, not steps, so `eval_steps=500`
with `patience=3` permits 1,500 steps past the optimum.

**8. Give three reasons a lower perplexity does not mean a better model.**
It is **not comparable across tokenizers**; it measures the **training
objective, not the task**, so instruction-following can get worse as it falls;
it is **meaningless under contamination**, measuring memory rather than
modelling; and for preference tuning the loss means something else entirely — a
DPO loss near zero is compatible with degenerate drift.

**9. Which behavioural probes can a loss curve never replace, and why not?**
The ones testing behaviour you **did not train on**: the adjacent task, general
knowledge (catastrophic forgetting), and refusals. Nothing in the objective ever
measured them, so no number derived from the objective can report on them. ✅
A ten-prompt before-and-after list read by a human costs five minutes and
catches what the curve structurally cannot.

---

## 17 · Tools and memory

**1. Why is memory the application's job rather than the model's?**
Because the model is **stateless**. Every call knows only what is in the
messages you send, and when it returns, nothing persists. "The agent
remembers" always means *your code put something in the next request* — which
turns a vague aspiration into a budget decision with a hard constraint: the
context window is finite and you pay for every token in it twice.

**2. Why does `safe_path` call `resolve()` before checking?**
`resolve()` collapses `..` and follows symlinks **first**, so the check runs
against the real destination. Checking the string for `".."` instead misses
absolute paths (`/etc/passwd`), symlinks pointing outside, and any encoding the
string check does not anticipate — and it wrongly rejects legitimate
subdirectory paths. Containment is "resolves under the root", not "no slashes".

**3. A sliding window forgets the oldest turns. Why is that wrong, and what
does `keep_first` fix?**
Because **people state their requirements first**. The oldest turns are the
task statement and the standing rules; recency is a poor proxy for importance.
`keep_first` pins the opening turns, so the task survives — but a constraint
stated in turn three still does not, which is the argument for a fact store.

**4. What does summarisation cost, and who chooses what is lost?**
**Detail you did not know you would need**, and the choice is made by whatever
wrote the summary — not by you, and not by the user whose constraint gets
dropped. ✅ Also worth marking: a summariser must fold the *previous* summary
in rather than append to it, or the summary grows until compaction costs more
context than it saves.

**5. You pay for tool schemas on every request. What follows about registry
size?**
That a large registry is a **permanent tax on every call**, not a one-off.
Thirty tools is roughly 3,000 tokens per request — about 72% of a 4k window
before a word is said. That is a better argument for small registries than
"the model gets confused", because you can measure it.

**6. One thing for a scratchpad, one for a fact store, and what goes wrong if
you swap them.**
Scratchpad: *"I already tried `peek_file` on the .exe and it is not text"* —
true for the next three steps, worthless next week. Fact store: *"invoices go
to finance"* — a standing rule. Swap them and you get either an agent that
repeats failed calls (rule-shaped memory thrown away), or a prompt
permanently carrying "filed 8 files on 14 March" — episodic noise in every
future request.

**7. Name three ways a model-written fact store goes bad.**
Any three of: **inferences stored as facts** ("the user seemed irritated");
**hedged facts** ("probably does not want..."), which nothing downstream can
act on; **silent overwrites** that reverse a rule with no record;
**episodic noise** stored as durable fact; **unbounded growth**. The fix
that helps most is that a fact store is a readable file you can correct —
which is more than a summary offers.

---

## 18 · Planning and decomposition

**1. What can a plan check that ReAct structurally cannot?**
**That the work is finished.** A ReAct agent stops when the model feels done;
with a plan, "did every task reach `done`?" is a property of a data structure.
A plan is also **reviewable before it runs**, which ReAct cannot be — the first
time you see the shape of the work is in the trace afterwards.

**2. Why is a plan a graph rather than a list? Give a consequence of each.**
A list encodes order; a graph encodes **dependency**. Consequences: (a)
independent tasks can run in the same wave, so you get parallelism for free;
(b) a failure blocks exactly its descendants, so sibling branches still
complete and their results are kept. A list gives you neither — it cannot tell
"after" from "because of".

**3. What does the wave count tell you before you run anything?**
The **shortest the plan can possibly take**: a plan in three waves needs at
least three sequential rounds however much you parallelise. Useful for costing
a plan, and for noticing that a "parallel" plan is secretly a chain.

**4. Name three invalid plans a model will actually produce.**
**Cycles** (step 3 "verifies step 5" which depends on 3), **dependencies on
task ids that do not exist**, **empty plans**, and **tasks with no
description**. All four are caught by `Plan.from_dicts` on the way in, which
is the point — fail at parse time, loudly, not three steps into execution.

**5. Distinguish `failed` from `blocked`, and say why conflating them hurts.**
A **failed** task ran and broke. A **blocked** task never ran, because
something it depended on failed. Conflating them sends you debugging a task
that did nothing wrong, and it breaks replanning: you only need to replace the
tasks that actually stuck, and `blocked` ones may be fine once their
dependency is fixed.

**6. Why does replanning keep completed tasks?**
Because re-running work that succeeded is slower and, with a flaky dependency,
can loop forever. Keeping them also constrains the new plan usefully: a
replacement task may depend on completed work, and `replan_after_failure`
**refuses** a replacement that depends on the wreckage it is replacing.

**7. Give two situations where ReAct is the better choice.**
When **what to do next genuinely depends on what you just found** (planning in
ignorance is worse than adapting), and when the task is **two or three steps**,
so the planning call is pure overhead. Also acceptable: an unpredictable
environment where plans go stale immediately.

**8. Why is a plan a better place for a human check than a step?**
Because **nothing has happened yet**, the whole shape of the work is visible at
once, and the check is cheap — an eleven-line `approve` caught a `Delete
everything` task before it ran. Checking at each step means deciding under
time pressure with no view of what comes next, and by the time you see the
delete the agent is already asking to do it.

---

## 19 · Multi-agent systems

**1. Give the three cases where a team is genuinely justified.**
① **Different tools that must not mix** — a security boundary. ② **Different
context that will not fit one window** — four specialists are four windows.
③ **Adversarial roles** — a critic that has not seen the writer's reasoning.

**2. Which of them can nothing else provide, and why?**
**Capability separation (①).** Context can be managed other ways and criticism
can be a second prompt, but "this agent cannot spend money" is enforced by
construction only if the tool was never given to it. Everything else relies on
the model behaving, which is not a guarantee.

**3. Name three bad reasons for multiple agents and what each really wants.**
"It mirrors how our team works", "each agent has a clear responsibility", "it's
more modular" — all three want **functions with good names**. Also: "agents can
specialise in a domain" usually wants a better system prompt, and "it's easier
to debug" is simply false (see question 8).

**4. What are the only three new mechanisms a team introduces?**
**Routing** (who handles this), **a handoff format** (what crosses the
boundary), and **a termination rule** (when the exchange stops). Everything
else is the single-agent loop you already have.

**5. Why is the `description` field the most important part of a `Specialist`?**
Because it is **the routing signal**, not documentation — the roster is what
the router reads to choose. "Handles data" routes nothing. It is the same
lesson as tool descriptions in notebook 10.5, one level up.

**6. What is the handoff cost of an orchestrator with three specialists?**
About **three times the model calls** of one agent with three tools (roughly 8
versus 3), plus **seven handoff messages**, each a lossy summary. You are
buying context and tool separation; if you do not need either, you are paying
for nothing.

**7. Why does the critic loop need both a round cap and an accept token?**
The **cap** because the pair will otherwise trade refinements forever, sounding
productive at two model calls per round. The **token** because prose has no
reliable way to express approval — "Looks good to me, although the tone is off"
is not an acceptance, and without an explicit `APPROVED` it will be read as one.

**8. Why is a team harder to debug than one agent, not easier?**
In one agent, a single transcript holds every fact and decision and you can
read it. In a team the context is **scattered across members**, and the only
record of what they told each other is the handoff log — which is why `Team`
records every message. Multi-agent is easier to *draw*, not to debug.

---

## 20 · Evaluating agents

**1. Give four ways an agent produces a correct answer you should not trust.**
It **guessed** and was right; it used **eleven tools where two would do**; it
called a **forbidden tool** (sent an email on a read-only question); it
**looped to `max_steps`** and then bluffed confidently. All four score a
perfect outcome.

**2. Define outcome and trajectory evaluation. Which catches a sent email?**
**Outcome**: is the final answer right. **Trajectory**: were the right tools
called, in a sane order, without doing anything forbidden. Only **trajectory**
catches the sent email — the answer "Done, I let your landlord know" is
perfectly accurate.

**3. What is a lucky pass, and why is it the most important thing in a report?**
A case where the **outcome passed and the trajectory did not** — right answer,
wrong path. It is the most important number because it is the **gap between
what you measured and what you have**: every lucky pass is a test that will
keep passing until the day the luck runs out.

**4. Why does `AgentTestCase` refuse to be built with no expectations?**
Because a case that asserts nothing **can never fail**, and a suite full of
them reports 100% forever. It is the most common broken test and the cheapest
possible thing to prevent.

**5. What does `max_tool_calls` protect against? Is it a correctness check?**
Against **cost regressions and wandering** — an agent answering correctly in
fourteen calls where it used to take three has regressed. It is **not** a
correctness check; the answer may be perfect. ✅ It also will not catch a
two-call loop, for which repetition is the better signal.

**6. Name three ways an LLM judge misleads you.**
It **prefers long, confident answers** (the same bias preference tuning
instals); it is **a model, so it is wrong sometimes**, giving the suite a noise
floor; and if it is **the same model you are evaluating** it grades its own
style generously. A fourth worth credit: two checks agreeing is not two pieces
of evidence when both are downstream of the same flaw.

**7. Distinguish pass@k from pass^k. Which applies to an agent that can send
email?**
**pass@k**: did it pass at least once in k runs — right when the user can
retry. **pass^k**: did it pass *every* run — right when the agent acts on the
world. For anything that can send email, **pass^k**, because one bad run is the
incident, and it is a much harsher number.

**8. What can a replayed trace verify, and what can it never verify?**
It verifies **your code**: that tool results still reach the model, that
`max_steps` holds, that forbidden-tool checks still fire — instantly, free,
in CI, on every commit. It can **never** verify that the model still makes
those decisions. That needs live, repeated runs, and it belongs before a
release rather than on every commit.

---

## 21 · Attention from scratch

**1. Why three projections rather than one? What question does each answer?**
Because relevance is **asymmetric**. $$Q$$ = what this position is looking for;
$$K$$ = what it advertises; $$V$$ = what it hands over if selected. One
projection could only ask "how similar are these two tokens?", which is
strictly weaker — it could not learn that *it* seeks nouns while *animal*
advertises as one and contributes something else again.

**2. Where does $$\sqrt{d_k}$$ come from, and what breaks without it?**
If $$q, k$$ have independent unit-variance entries then
$$q \cdot k$$ is a sum of $$d_k$$ such terms, so its variance is $$d_k$$ and
its standard deviation $$\sqrt{d_k}$$. Without the divisor, scores grow with
head width and softmax saturates — 96% of the weight on one position at
$$d_k = 64$$. Since the softmax derivative goes like $$p(1-p)$$, a saturated
row yields **no gradient** and the model stops learning.

**3. Why must the mask use $$-\infty$$ rather than a large negative number?**
Because the mask value must underflow to exactly zero **in the dtype you
actually compute in**. `-1e9` happens to work in float64 and is not even
representable in float16; under mixed precision the dtype changes beneath you.
A leak here is invisible in the loss (training looks *better*) and corrupts
evaluation, because the model learned to use a signal that is absent at
inference.

**4. State permutation equivariance and why it forces positional encoding to
exist.**
$$\operatorname{Attention}(PX) = P\operatorname{Attention}(X)$$ for any
permutation $$P$$: shuffle the inputs and the outputs shuffle identically.
Attention is defined over a **set**, so it cannot distinguish *dog bites man*
from *man bites dog*. Position is therefore not something attention neglects —
it is something attention **cannot represent**, so it must be injected.

**5. What does RoPE make the score depend on, and why is that right for
language?**
Only the **gap** $$m - n$$ between positions, never the absolute positions —
because a rotation by $$m$$ measured against a rotation by $$n$$ leaves a
rotation by $$m-n$$. That suits language because grammatical relationships are
relative ("the adjective before this noun"), not absolute ("the word at index
4,312"), and it extrapolates past the training length far better than a
learned absolute embedding.

**6. Multi-head attention with $$h$$ heads costs the same as one head. Why?**
Because $$d_k = d_{\text{model}}/h$$. The heads **split** a fixed width rather
than adding to it, so the projections and the matmuls total the same work.
Eight heads of width 64 cost what one head of width 512 costs — you are
spending the same budget on several narrow views instead of one wide one.

**7. What is the residual stream, and give two things it buys you.**
The running representation that every block reads from and adds back to:
$$X \leftarrow X + \text{sublayer}(\text{Norm}(X))$$. It buys (a) **optional
layers** — a block whose output is near zero passes its input through
untouched, so not every layer need fire for every token; and (b) **a short
gradient path**, since addition passes gradient straight through, which is
what makes very deep stacks trainable. ✅ Pre-norm matters here: the stream
itself is never normalised, keeping that path clean.

**8. Which holds more parameters, attention or the MLP? Which dominates FLOPs
at 512 tokens?**
The **MLP** holds more parameters: $$3d_{\text{model}}d_{ff}$$ against
$$4d_{\text{model}}^2$$ — 9.4 M against 4.2 M at Qwen3-0.6B's dimensions. At
512 tokens the **projections** dominate the FLOPs; the attention matmuls are
about 8% of them. Attention only takes over past roughly
$$n = 4d_{\text{model}} + 3d_{ff}$$ tokens, which is ≈13,000 for that model.

---

## 22 · Working in Kazakh and Russian

**1. How many bytes does a Cyrillic character take in UTF-8, and why is that
only a floor?**
**Two**, against one for ASCII — exact and unavoidable. It is only a floor
because bytes are not tokens: the tokenizer adds a further penalty on top, and
that second penalty is larger and is a design choice rather than a property of
the encoding.

**2. Why does a tokenizer trained on English-dominant data split Kazakh into
single letters?**
BPE repeatedly merges the **most frequent adjacent pair**. In a corpus that is
99% English, no Cyrillic pair is ever frequent enough to win a merge, so
Cyrillic never gets merged beyond single characters. Not malice or bad design —
a greedy algorithm optimising for the data it was given.

**3. Does a bigger vocabulary narrow or widen the gap?**
**Widens it.** The notebook measures the multiplier growing from 2.0x to 3.2x
as merges go from 500 to 4,000. Extra merges go to whatever is frequent, which
is English, so scaling the vocabulary scales the advantage. The fix is a
corpus where the language is not a rounding error — not a bigger vocabulary.

**4. Give the three practical costs of a 2x token multiplier.**
**Context** — a 32k window holds ~131k characters of English but only ~65k of
Kazakh, so users silently get half the document. **Cost** — twice the bill for
the same content on per-token pricing. **Latency** — roughly twice the
generation time, since tokens are produced one at a time. ✅ The third is the
most visible to users and the least discussed.

**5. Why does checking "is the answer Cyrillic?" fail to detect
code-switching, and what works?**
Because **Russian is Cyrillic too** — a Russian reply to a Kazakh question
scores 1.0 on that test. What works is counting the nine letters Kazakh has
and Russian does not: **ә ғ қ ң ө ұ ү һ і**. Almost any real Kazakh sentence
contains several; a Russian sentence contains none.

**6. Name the two compounding reasons RAG degrades in Kazakh.**
**Chunks hold less** — chunking by tokens means a Kazakh chunk carries about
half the content, so retrieval is half as informative at the same `k`. And
**embeddings are weaker**, because embedding models are trained on the same
skewed corpora, so recall drops *on top of* the smaller chunks. Fix the first
immediately by chunking on characters.

**7. Will LoRA fine-tuning reduce your token costs?**
**No.** An adapter changes behaviour, not the vocabulary. The tokenizer is
untouched, so your Kazakh text costs exactly the same number of tokens after
fine-tuning as before. People expect otherwise and are disappointed.

**8. What is the root cause underneath all of this?**
**The scarcity of varied Kazakh text on the public internet.** Not the script,
not the grammar, not the agglutinative morphology. The tokenizer is unfair
because the corpus is skewed; model quality is lower for the same reason; and
fine-tuning is hard because the data is not there. ✅ The notebook demonstrates
this accidentally — the author could not build the balanced-corpus comparison
precisely because no varied Kazakh corpus was to hand.
