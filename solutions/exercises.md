# Exercise solutions

All 20 in-notebook exercises, in order. See [README](README.md) for the
confidence markers (✅ verified · 🔬 expected · ✍️ yours).

---

# Notebook 01 — LLM foundations

## 01-1 · Add another language to `samples`

> Add Turkish, Arabic or your own language to `samples`. Which is most
> expensive? Write down the chars/token you measured.

🔬 **Expected ordering**, cheapest to most expensive per character:

| Language | Chars/token |
|---|---|
| English | ~4.0 |
| Code | ~3.0 |
| Turkish | ~2.5 |
| Russian | ~2.0 |
| Kazakh (Cyrillic) | ~1.8 |
| Arabic | ~1.8 |
| Chinese | ~1.5 |

**Teaching point.** Tokenizer vocabularies are learned from training data, which
is overwhelmingly English. A Kazakh document can consume **twice** the context
window of the same content in English, and on a paid API costs twice as much.
This is a real fairness issue, not a curiosity.

**Push them further:** ask what this means for a Kazakh-language RAG system.
Answer: half the documents fit in the same context, so chunking and retrieval
quality matter correspondingly more.

**Common mistake.** Comparing *token counts* between languages instead of
*chars/token*. A longer translation naturally has more tokens; the ratio is what
reveals tokenizer bias.

---

## 01-2 · `add_generation_prompt=False`

> Set `add_generation_prompt=False` and print it again. What disappeared? Why
> would the model keep silent (or ramble) without it?

✅ **What disappears:** the trailing assistant header. With it `True`, the
rendered prompt ends with something like:

```
<|im_start|>assistant
```

With it `False`, the string ends after the user's `<|im_end|>`.

✅ **Why it matters.** The model continues whatever text it is given. Ending on
a completed user turn, the most likely continuation is *another user turn* — so
it may invent the next question, write dialogue for both sides, or emit
`<|im_end|>` immediately and produce nothing.

The generation prompt is what says *"you are the assistant and it is your turn
now."* It is not decoration.

**Common mistake.** Assuming the flag adds the assistant's *reply*. It adds only
the header that opens the assistant's turn.

---

## 01-3 · `enable_thinking=True`

> Set `enable_thinking=True`. What extra marker appears?

✅ **Answer:** the template opens a `<think>` block after the assistant header,
so the prompt ends roughly:

```
<|im_start|>assistant
<think>
```

✅ **The subtle consequence**, worth pointing out explicitly: because the
template *opens* the tag, the model only needs to generate the **closing**
`</think>`. That is exactly why `split_thinking()` in
`src/qwen_workshop/chat.py` handles output where `</think>` appears without a
matching `<think>` — the opener never came from the model.

Students who write their own parser assuming both tags will hit this.

---

# Notebook 02 — Environment and hardware

## 02-1 · KV cache at 32k context

> Re-run with `context_tokens=32768`. At what model size does the KV cache start
> to rival the weights?

✅ **Verified** by running the notebook's own `memory_estimate`:

| Model | Precision | Weights | KV @ 32k | KV as % of weights |
|---|---|---|---|---|
| Qwen3-0.6B | bf16 | 1.1 GB | 3.50 GB | **313%** |
| Qwen3-0.6B | int4 | 0.3 GB | 3.50 GB | **1253%** |
| Qwen3-4B | bf16 | 7.5 GB | 4.50 GB | 60% |
| Qwen3-4B | int4 | 1.9 GB | 4.50 GB | **242%** |
| Qwen3-8B | bf16 | 14.9 GB | 4.50 GB | 30% |
| Qwen3-8B | int4 | 3.7 GB | 4.50 GB | **121%** |
| Qwen3-14B | int4 | 6.5 GB | 5.00 GB | 77% |

✅ **The answer is sharper than the question implies.** At 32k context the KV
cache **exceeds the weights for every int4 model in the table**, and for the
0.6B model it is more than twelve times the weights.

**Teaching point.** Quantization shrinks weights and does nothing to the KV
cache. So the harder you quantize, the *more* the cache dominates — which is why
"it fits in VRAM" is a statement about an empty context, and why a long
conversation OOMs on a model that loaded fine.

**Push them further:** what do you do about it? Answers: cap `--max-model-len`,
quantize the KV cache itself, or use a model with fewer KV heads (grouped-query
attention, which is exactly why `n_kv_heads=8` rather than 32 in these numbers).

---

# Notebook 03 — First generation

## 03-1 · Decode without `skip_special_tokens`

> Print `qwen.tokenizer.decode(output_ids[0])` — the whole thing, without
> `skip_special_tokens`. Find `<|im_start|>` and `<|im_end|>`.

✅ **What they see:** the complete conversation as one flat string —

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Name three uses...<|im_end|>
<|im_start|>assistant
1. ...<|im_end|>
```

✅ **The three things to draw out:**

1. **The prompt is included.** `generate()` returns prompt + completion, which is
   why step 4 of the pipeline is the slice. This exercise makes that visible.
2. **Roles are just text with markers.** There is no structured "message object"
   at the model level — only a string.
3. **`<|im_start|>` is a single token**, not 12 characters. That is what makes
   it hard for a user to forge a system message by typing it.

**Best use:** run this live. It converts "the chat template does something
abstract" into "oh, it's a string" in about twenty seconds.

---

## 03-2 · A system prompt the model must hold

> Write a system prompt that makes the model always answer in Kazakh, or always
> reply as valid JSON. Which instruction does the 0.6B model follow reliably,
> and which does it drift away from after a few turns?

🔬 **Expected result:** **language holds better than format.**

- *Always answer in Kazakh* — usually survives several turns, though quality is
  poorer than its Russian or English output.
- *Always reply as valid JSON* — drifts quickly. Expect markdown fences, a
  preamble ("Sure, here is the JSON:"), or prose after the closing brace. On a
  0.6B model, drift within 3–5 turns is normal.

**Teaching point, and the one to land:** a prompt is a *request*, not a
guarantee. Reliability that depends on the model continuing to cooperate is not
reliability. Notebook 07 replaces the request with constrained decoding, which
makes invalid JSON impossible rather than unlikely.

**Marking note.** A student who reports "it worked fine" has probably tested one
turn. Ask for five.

---

## 03-3 · Record your tokens/second

> Record your tok/s. Compare with a classmate on different hardware. If you have
> a GPU, re-run with `device="cpu"` and measure the slowdown factor.

🔬 **Expected ranges for Qwen3-0.6B:**

| Hardware | tok/s |
|---|---|
| Modern GPU | 60–150 |
| Apple Silicon (MPS) | 30–60 |
| CPU, 4–8 cores | 8–20 |

🔬 **GPU → CPU slowdown:** typically **5–15×**. Less than people expect, because
a 0.6B model is small enough that CPU inference stays tolerable. Repeat at 4B
and the gap widens sharply.

**Teaching point.** Reading speed is roughly 7 tok/s. Below that, streaming is
doing the heavy lifting on perceived latency. Above ~30 tok/s, faster stops
being noticeable for a chat interface — which matters when students are choosing
hardware.

✍️ **Collect the class numbers on a whiteboard.** The spread across the room is
the lesson, and it makes the day-1 hardware discussion concrete.

---

# Notebook 04 — Decoding and prompting

## 04-1 · top_p=1.0 vs top_p=0.9 at high temperature

> Generate the same prompt 5 times with `temperature=1.5, top_p=1.0` and 5 times
> with `temperature=1.5, top_p=0.9`. Count how many outputs in each group are
> coherent.

🔬 **Expected:** roughly **1–3 of 5 coherent** at `top_p=1.0`, and **4–5 of 5**
at `top_p=0.9`.

**Why.** At `T=1.5` the distribution is flattened enough that the long tail of
nonsense tokens gains real probability mass. `top_p=1.0` keeps all of them.
`top_p=0.9` removes the tail *before* sampling, so high temperature produces
variety among plausible options rather than variety including garbage.

**The clean framing:** temperature controls *how adventurous* the sampling is.
top-p controls *what is allowed in the pool at all*. They are not two dials for
the same thing.

**Common mistake.** Concluding "high temperature is bad". High temperature with
proper truncation is exactly what you want for brainstorming. It is the
*combination* that misbehaves.

**Marking note.** "Coherent" is a judgement call — that is fine, and worth
saying. Ask two students to score the same 10 outputs and compare. Disagreement
is a free introduction to why evaluation is hard.

---

## 04-2 · Stop the model adding "Sure, here is…"

> The model will sometimes add "Sure, here is the rewritten message:" despite
> being told not to. Fix it with prompting.

🔬 **Prompting fixes that usually help**, roughly in order of effectiveness:

1. **Give one worked example.** Few-shot beats instruction for format compliance,
   reliably.
2. **Put the constraint last.** Instructions at the end of the prompt are
   followed more consistently than ones at the top.
3. **Be concrete about the first character:** "Your reply must begin with the
   first word of the rewritten message."
4. **Prefill the assistant turn** if your runtime allows it — the model cannot
   write a preamble it has already been given past.
5. **Strip it in post-processing.** Unglamorous, completely reliable, and often
   the right answer.

✅ **The point of the exercise is that none of these are guarantees.** Even the
best prompt fails some fraction of the time, and that fraction is invisible
until you measure it. Notebook 07's constrained decoding changes the category of
the problem rather than improving the odds.

**Marking note.** Full credit for a student who says "I got it to 9/10 and then
wrote a `.removeprefix()`". That is the engineering answer.

---

# Notebook 05 — Quantization and runtimes

## 05-1 · `group_size` 128 and 16

> Change `group_size` to 128 and to 16. Smaller groups mean better accuracy but
> more scales to store.

✅ **Verified** with the notebook's own code (4-bit, 4096 normal weights,
seed 613):

| group_size | Mean abs error | Relative error | Bits/weight |
|---|---|---|---|
| 16 | 0.007230 | 9.1% | 5.00 |
| 32 | 0.008480 | 10.6% | 4.50 |
| 64 | 0.009549 | 12.0% | 4.25 |
| **128** | 0.010601 | **13.3%** | **4.12** |
| 256 | 0.011251 | 14.1% | 4.06 |

✅ **Read the trade-off:** going from group 128 → 16 cuts relative error from
13.3% to 9.1% (a 32% improvement) but costs 0.88 extra bits per weight — over
20% more storage for a 4-bit payload.

**Teaching point.** This *is* the `_K_S` / `_K_M` / `_K_L` distinction. Those
suffixes encode how generously the scheme spends bits on scales and on
sensitive layers. The exercise turns a mysterious filename suffix into an
arithmetic trade-off.

**Push them further:** why is the error not simply proportional to group size?
Because each group's scale is set by its **largest magnitude** weight. A big
group is more likely to contain an outlier, which stretches the scale and
coarsens the quantization of every other weight in that group. Outliers are why
this matters at all.

---

## 05-2 · A 20-question quiz, 0.6B vs 4B-Q4

> Extend `QUIZ` to 20 questions about something you know well. Score Qwen3-0.6B
> and Qwen3-4B at Q4_K_M. Is the bigger quantized model better?

🔬 **Expected:** yes, and usually by a wide margin — something like 0.6B scoring
40–60% and 4B-Q4 scoring 75–90% on general-knowledge questions.

**The headline:** *a bigger model quantized harder beats a smaller model
quantized lightly, at the same file size.* Students should leave able to state
this as a rule.

✍️ **Make them write the quiz about something they know well.** Questions about
their own degree subject, home town or hobby expose the small model's
limitations far more honestly than trivia, where it does deceptively well.

**Marking note.** 20 questions is far too few to conclude anything, and the
notebook says so. A student who notes that unprompted has understood the deeper
lesson. The method — fix the task, vary one thing, measure — is the deliverable.

---

# Notebook 07 — Structured output and tools

## 07-1 · A `CalendarEvent` model

> Build a `CalendarEvent` model (title, date, start time, duration minutes,
> location, attendees list) and extract events from three free-text sentences.
> Which field does the small model get wrong most often?

🔬 **Expected worst field: `date`**, specifically relative dates. "Next Tuesday",
"tomorrow" and "the 3rd" require knowing today's date, which the model does not
have unless you give it. Expect invented dates, or dates from the training
distribution.

Ranked, worst first: `date` → `duration_minutes` (inferring "a quick chat" = 15)
→ `attendees` (missing implied people) → `location` → `title` (nearly always
fine).

✅ **The fix is a tool, not a better prompt.** Put today's date in the system
prompt or expose `current_date()`. This is the same lesson as letter-counting:
do not ask the model for something a function supplies exactly.

**A reference model to compare against:**

```python
class CalendarEvent(BaseModel):
    title: str = Field(min_length=1)
    date: str = Field(description="ISO date, YYYY-MM-DD")
    start_time: str = Field(description="24-hour HH:MM")
    duration_minutes: int = Field(gt=0, le=24 * 60)
    location: str | None = None
    attendees: list[str] = Field(default_factory=list)
```

**Marking note.** Reward anyone who adds a `field_validator` to reject a date in
the past, or who notices that `duration_minutes` needs an upper bound.

---

## 07-2 · Chaining `search_notes` and `read_note`

> Add a `search_notes(query)` tool. Ask a question that needs both `read_note`
> and `search_notes`. Does the 0.6B model chain them correctly?

🔬 **Expected: often not.** Realistically around 1 in 3 on a 0.6B model. Typical
failures:

- calls `search_notes`, gets a match, then **answers from the snippet** without
  opening the file;
- calls `read_note` first with a guessed filename;
- calls one tool and stops.

✅ **This failure is the point**, and the notebook should not be read as
promising success. Multi-step tool chaining is precisely where small models fall
down. Have them re-run with `qwen3:4b` — the difference is dramatic and is the
single most convincing argument in the workshop for matching model size to task.

**What improves it without changing model:** sharper descriptions that say
*when* to use each tool ("use `search_notes` to find which note, then
`read_note` to get the detail"), and a system prompt that insists on reading the
file before answering.

**Note.** `search_notes` already exists in `src/qwen_workshop/demo_tools.py`, so
a student can compare their version with a worked one.

---

# Notebook 08 — Embeddings and search

## 08-1 · Negation

> Add a sentence that is lexically similar but semantically opposite, e.g.
> "The cat did not sit on the mat." How well does the model separate them?

🔬 **Expected: badly.** Similarity between "The cat sat on the mat" and "The cat
did not sit on the mat" typically lands around **0.85–0.95** — about as high as
a genuine paraphrase.

✅ **Why.** Embeddings are trained so that texts about the same *topic* land
near each other. "Not" is one short token in an otherwise identical sentence, and
topical similarity dominates. The models are not broken; they are doing what
they were trained to do.

**Why it matters in practice.** A RAG system asked *"which notes do **not**
mention the exam?"* will retrieve exactly the notes that mention the exam. Any
query whose meaning turns on a negation, an exclusion or a date range is
unreliable through embeddings alone.

**The fixes:** metadata filters for anything structured, keyword/BM25 for exact
terms, and query rewriting to turn a negation into a positive retrieval plus a
post-filter.

**Push them further:** try "I love this" vs "I hate this". Same effect, and it
explains why embedding similarity is a poor sentiment classifier.

---

## 08-2 · Point it at your own documents

> Replace `data/notes/` with a folder of your own notes, lecture slides
> converted to text, or documentation. Search it.

✍️ **No single answer — this is the exercise that changes how students feel
about the workshop.**

**Practical notes to have ready:**

- **PDFs need converting.** `pip install pymupdf4llm`, then
  `pymupdf4llm.to_markdown("slides.pdf")`. Scanned pages need OCR and usually
  are not worth the trouble in a session.
- **Chunk size matters more than they expect.** Slides convert to short,
  fragmented text; a `max_chars` of 400–500 often beats the 700 default.
- **Remind them nothing leaves the machine.** This is the moment the privacy
  argument for open weights stops being abstract, and it is worth saying aloud.
- **`data/private/` is already gitignored** for exactly this.

**Marking note.** The valuable report is "it failed on X". Students searching
their own material find real retrieval failures far faster than any prepared
corpus produces.

---

# Notebook 09 — RAG

## 09-1 · Write 10 evaluation questions and measure

> Write 10 evaluation questions about your own documents, with expected sources.
> Measure recall@3. Then change one thing and measure again.

✍️ **The most important exercise in the workshop.** Budget 45 minutes and do not
cut it.

🔬 **Expected recall@3 on a reasonable corpus: 60–85%** on a first attempt.
Anything near 100% means the questions are too easy or the corpus too small —
worth saying before they start, so a high score prompts suspicion rather than
satisfaction.

**Changes that usually help, in rough order:**

1. **Hybrid search** (adding keyword matching) — biggest single win, especially
   for identifiers and proper nouns.
2. **Chunk size** — 400–500 for dense reference text, 700–900 for prose.
3. **Larger k** — raises recall but adds distracting context; the trade-off is
   the point.
4. **Query rewriting** — strong on vague conversational questions.
5. **A reranker** — retrieve 20, rerank to 3.

✅ **Insist on good question design.** The common failure is questions that quote
the source almost verbatim, which tests string matching rather than retrieval.
Push them to write questions **a real user would ask**, in their own words.

**Marking note.** A student whose change made recall *worse*, who reports that
honestly, has done the exercise correctly. Say so out loud — it sets the tone
for the capstone's honesty section.

---

# Notebook 10 — Agents

## 10-1 · Remove the `stop` parameter

> Remove the `stop` parameter and re-run. Watch the model write its own
> `Observation:` line.

✅ **What happens:** the model generates `Thought:`, `Action:`, `Action Input:`
and then — because it is simply continuing text and `Observation:` is what comes
next in the pattern — **invents the observation**, then reasons from the fiction
and produces a confident, entirely fabricated answer.

✅ **Why.** The model has no concept of "my turn ended". It continues the
document. The ReAct protocol only works because something external stops
generation at the boundary and inserts the *real* result.

**This is the best live demo in the workshop.** It takes 30 seconds and makes
two things visceral at once: that the model is always just continuing text, and
that the scaffolding around it is load-bearing.

**Connect it forward:** native tool calling exists precisely so this failure is
structurally impossible. The server returns a `tool_calls` field and generation
stops; there is no text for the model to run past. That is why notebook 10
recommends `Agent` over `ReActAgent` for anything real.

---

## 10-2 · Rewrite the agent as a chain

> Take the budget question from 10.4 and rewrite it as a chain: three explicit
> function calls, one prompt, no loop. Compare latency, token cost and
> reliability across 5 runs each.

🔬 **Expected: the chain wins on every axis measured.**

| | Agent | Chain |
|---|---|---|
| Model calls | 4–7 | 1 |
| Latency | 4–7× the chain | baseline |
| Tokens | 5–10× | baseline |
| Reliability over 5 runs | variable | ~identical every time |

**A reference chain:**

```python
target  = search_notes("budget target")          # no model needed
spend   = spending_by_category("2026-01")        # no model needed
diff    = calculate(f"{spend['total_kzt']} - 180000")
answer  = model(f"Target {target}, spent {spend}, difference {diff}. "
                "Summarise in two lines.")       # one call, at the end
```

✅ **The lesson students should state back to you:** *the agent's flexibility
bought nothing here, because the steps never change.* If you can draw the
flowchart, write the flowchart.

**Push them further:** when does the agent win? When the steps depend on what is
found — "why is this test flaky?" might take one step or twelve. That is the
honest case for agents, and it is narrower than the marketing suggests.

**Marking note.** Full credit requires a *recommendation with a reason*, not
just a table. "I would ship the chain because the steps are fixed and I can test
it" is the answer.

---

# Notebook 11 — Fine-tuning

## 11-1 · 40 more examples, then measure

> Write 40 more examples in the same style and retrain. Measure: average reply
> length, fraction using "₸", fraction with a "Tip:" line. Report the three
> numbers before and after.

🔬 **Expected, going from 12 → ~52 examples:**

| Metric | Base | 12 examples | ~52 examples |
|---|---|---|---|
| Avg reply length | 60–120 words | 30–60 | 15–30 |
| Uses ₸ | ~20% | 50–70% | 85–100% |
| Has a "Tip:" line | ~0% | 40–70% | 85–100% |

**The shape matters more than the numbers:** format compliance improves sharply
and then saturates. Most of the gain from 12 → 52 examples; much less from
52 → 200 for a task this narrow.

✅ **Insist on consistency.** Every inconsistency in the training data is an
instruction to be inconsistent. Forty carefully uniform examples beat two
hundred sloppy ones, and students usually need telling twice.

**A measurement snippet to hand out:**

```python
import re
def score(replies):
    return {
        "avg_words": sum(len(r.split()) for r in replies) / len(replies),
        "uses_tenge": sum("₸" in r for r in replies) / len(replies),
        "has_tip":    sum(bool(re.search(r"\bTip:", r)) for r in replies) / len(replies),
    }
```

**Also have them check off-task behaviour** (notebook 11 §11.8). If the tuned
model now answers "what is the capital of France?" with a budget tip, that is
catastrophic forgetting, and it is a better finding than a clean win.

---

# Notebook 12 — Capstone

## 12-1 · Improve `suggest_shopping_list`

> The prices are hardcoded, it ignores `meals` entirely, and it has no idea what
> recipes are possible. Improve one of those three.

✍️ **Deliberately open. All three are reasonable; they differ in difficulty:**

| Fix | Difficulty | What it teaches |
|---|---|---|
| **Honour `meals`** | easiest | scale quantities; an argument that is accepted and ignored is a bug |
| **Prices from data** | medium | move hardcoded values into `data/`; cite the source in the return value |
| **Recipe awareness** | hardest | cross-reference pantry against `data/notes/cooking.md` — the genuinely interesting one |

✅ **What to reward regardless of which they pick:**

- **The return value stays structured.** A tool that returns prose is harder for
  the model to use than one returning a dict.
- **New arguments are validated.** `ToolError` with a message that tells the
  model how to fix the call.
- **They tested the tool without the model.** Most agent bugs are tool bugs, and
  the test is five lines.

**Marking note.** A student who says "I made it honour `meals` and then realised
my test did not check the quantities actually changed" has learned the lesson
this exercise exists for.
