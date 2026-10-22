# Exercise solutions

All 30 in-notebook exercises, in order. See [README](README.md) for the
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

---

# Notebook 13 — Pre-training a tiny model

## 13-1 · Train for 6000 steps

> Train for 6000 steps instead of 2000. Does validation loss turn upward? At
> which step? That step is where this model stops learning and starts
> memorising.

🔬 **Expected: yes, it turns.** The corpus is a few hundred KB and the model has
~800k parameters, so there is far more capacity than data. Validation loss
typically bottoms somewhere in the **2,500–4,500** range and then climbs while
train loss keeps falling.

**The exact step does not matter and students should be told so.** What matters
is the shape:

| Observation | What it proves |
|---|---|
| Train loss still falling at step 6000 | Train loss is not evidence of anything |
| Val loss bottoms, then rises | The optimum is a point, not an endpoint |
| The gap widens monotonically after the bottom | Every step past it is damage |

**Push them further:** ask what you would change to move the bottom later.
Answers in rough order of effect: **more data** (the real fix), a smaller model,
dropout, weight decay. Getting "more data" first is the point.

**Common mistake.** Reporting "it overfit" without the step number. The whole
exercise is locating the turn; `min(history, key=lambda h: h[2])` does it in one
line and notebook 16 generalises it into `early_stop`.

---

## 13-2 · Double `n_embed`, halve the steps

> Double `n_embed` to 256 and halve `MAX_STEPS`. You now have roughly 4× the
> parameters and half the training. Does validation loss end up better or worse?
> This is the scaling-law trade-off, at a scale you can actually run.

🔬 **Expected: worse, or no better.** The run is already data-limited, so paying
for capacity with steps trades away the thing that was actually binding.

Note the arithmetic first — `n_embed` enters the parameter count roughly
quadratically through the attention and MLP matrices, so doubling it is about
4× the parameters *and* about 4× the time per step. Halving `MAX_STEPS` does not
make the run cheaper; it makes it about twice as expensive for less training.

✅ **The transferable result** is Chinchilla's: compute split between model size
and tokens seen has an optimum, and both sides of it are worse. Students who
have only heard "bigger is better" should leave this exercise unable to believe
that again.

**Marking note.** A student who reports "worse, and it took longer" has the
whole lesson. A student who reports "better" has probably not held the seed
fixed — run-to-run variance at this scale is large, which is itself exercise
16-3's point.

---

# Notebook 14 — Building a training dataset

## 14-1 · Ten much-longer answers

> Add ten examples whose answers are deliberately much longer than the rest.
> Re-run `describe`. Then predict what the fine-tuned model's answer length
> distribution would look like, and why that is a problem.

✅ **What `describe` shows:** the mean answer length jumps, and — more
informative — the distribution goes **bimodal**. Students should be pushed to
look at the spread, not the mean: a mean that moves from 80 to 140 characters
tells you much less than "there are now two clusters".

🔬 **The prediction:** the model does *not* learn "answer at two different
lengths depending on the question". It learns something closer to the average,
and it applies it indiscriminately — short questions get padded answers and long
ones get truncated. Length is a style feature with no conditioning signal
attached, so the model has nothing to condition on.

**Why that is a problem, in one sentence:** you have taught a behaviour you
never chose, and it will show up on every output.

**Push them further:** how would you *deliberately* teach two lengths? The
answer is to put the signal in the prompt — "answer in one line" versus "explain
in detail" — so there is something to condition on. That is the difference
between a dataset with a confound and a dataset with a feature.

---

## 14-2 · Corrupt 25% of the set

> Take the twenty examples here and deliberately make five of them inconsistent
> — different answer length, different tone, markdown where the others have none.
> You now have a 25% corruption rate. Predict what the fine-tuned model does,
> then test the prediction by feeding the corrupted file to notebook 11's LoRA
> run.

🔬 **Prediction:** the model becomes *inconsistent*, not *wrong*. It will
sometimes use markdown and sometimes not, with no apparent pattern. This is
worse than uniformly bad output, because it cannot be fixed downstream — you
cannot write a parser for "sometimes".

✅ **The principle, which is the actual deliverable:** every inconsistency in the
data is an instruction to be inconsistent. The model has no way to know that
five of twenty examples were a mistake; from inside the loss, they are evidence.

**Marking note.** 25% is deliberately large. Ask what they think the threshold
is. There is no clean answer, and the honest one is instructive: **a few percent
of contradictory examples is enough to show up in the outputs**, which is why
"read every example" is realistic advice at 50 examples and why curation
pipelines exist at 50,000.

**If they actually run it:** the training loss will be *higher* and flatter than
on the clean set, because contradictory targets cannot all be fit. That is a
detectable signature — and a good reason to compare loss curves between data
versions rather than looking at one in isolation.

---

# Notebook 15 — Preference tuning with DPO

## 15-1 · A rule for a different checkable property

> Write a `score` function for a different checkable property — "the answer
> cites at least one source in `[n]` form", or "the code compiles". Feed it
> three candidates and confirm `pair_by_rule` picks the pair you expect. Then
> say in one sentence why this approach cannot teach the model to be *tactful*.

✅ **A correct answer for the compile case:**

```python
def compiles_score(text):
    try:
        compile(text, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return 0.0
    return 1.0
```

and for citations, `1.0 if re.search(r"\[\d+\]", text) else 0.0`.

**What to check in their code:**

- **It returns a number, not a bool**, so `sorted` ranks rather than groups.
- **It catches the right exceptions.** `compile` raises `ValueError` on embedded
  null bytes as well as `SyntaxError`; a bare `except Exception` is acceptable
  here and worth discussing rather than marking down.
- **They checked the `None` case.** Three candidates that all score the same
  must produce no pair. A student who did not test that has not understood why
  the guard is in `pair_by_rule`.

✅ **Why it cannot teach tact:** tact is not a computable property of the text —
there is no function from the string to a score, which is precisely the
condition under which you need a human or a judge model.

**Push them further:** ask which of the four failure modes in 15.8 a
rules-generated dataset is *immune* to. Answer: sycophancy, because no annotator
preference entered the data. It remains fully exposed to length bias if the
rule correlates with length.

---

## 15-2 · Margins of −20 and +20

> The sign of the margin matters more than its size. Compute `dpo_loss` for
> margins of −20 and +20 at β=0.1 and explain the asymmetry in the gradient:
> which of the two pairs would dominate a batch containing both, and why is
> that the behaviour you want?

✅ **The numbers**, at β=0.1 (so βm = ∓2):

| margin | loss | dL/dmargin |
|---|---|---|
| −20 | 2.1269 | −0.0881 |
| +20 | 0.1269 | −0.0119 |

The loss is **17× larger** on the wrong pair, and the gradient is about **7×
larger**.

✅ **Which dominates:** the −20 pair, decisively. The logistic loss is
asymptotically linear on the wrong side and asymptotically flat on the right
side, so a batch's gradient is dominated by the pairs the model currently gets
wrong.

**Why that is what you want:** training capacity goes to the examples that still
carry information. A pair the model already ranks correctly with a wide margin
has nothing left to teach, and continuing to push it would only increase drift
from the reference for no behavioural gain.

**Push them further:** what does this imply about adding more easy pairs to a
preference set? Answer: almost nothing happens — you pay for the forward passes
and get gradients near zero. Preference-set *difficulty* matters more than
preference-set size, which is the opposite of the usual intuition about data.

---

## 15-3 · Negative `reward_chosen` with a falling loss

> Using `dpo_loss` (which returns `loss, reward_chosen, reward_rejected, grad`),
> find policy log-probs where `reward_chosen` is negative and the loss is
> nonetheless below the starting 0.693. Then say what you would check in a real
> run that showed this, and what the `grad` value tells you about whether the
> run will keep moving.

✅ **One worked answer**, against the notebook's reference of `(−42, −38)`:

```python
dpo_loss(-50.0, -60.0, -42.0, -38.0)
# loss 0.2204, reward_chosen -8.0, reward_rejected -22.0, grad -0.01978
```

The policy made the chosen answer **8 nats less likely** than the reference did,
and the loss went *down*, because it made the rejected answer 22 nats less
likely. Margin +14. Any pair where both rewards are negative and
`reward_chosen > reward_rejected` works.

✅ **What to check in a real run:** the **reward levels**, not the margin —
`rewards/chosen` and `rewards/rejected` in the trainer's logs. Both sinking well
below zero is the degenerate drift of 15.8. Then: generate from the model.
Suppressing everything shows up as bland, hedging, interchangeable answers long
before any metric catches it.

**The `grad` reading:** −0.0198 against −0.0500 at step 0, so the pair is still
contributing about **40%** of its initial pull. It has not saturated, and the
run **will keep moving in this direction**. That is the part students miss: a
falling loss with sinking reward levels is not a run that is about to
self-correct. It is one that is 60% of the way to not caring any more, with the
reward levels still heading down.

**Fixes, in order:** lower the learning rate, lower β, then look at the pairs —
if `chosen` and `rejected` are both poor answers, the data is the problem and no
hyperparameter will rescue it.

---

# Notebook 16 — Training diagnostics

## 16-1 · Break one of `diagnose`'s thresholds

> `diagnose` has four hard-coded thresholds: `10 * chance_loss`, `0.95`, `1.10`
> and `0.05`. Pick one, find a run where it gives the wrong answer, and say what
> you would use instead. Hint: the `0.05` tail-slope test is the most fragile —
> what does `every=1000` do to it?

✅ **The `0.05` tail slope is the intended answer, and the hint gives it away on
purpose.** The test compares the first and last entries of the final quarter of
the *history*, and history length depends on `every`. With `every=1000` over
3,000 steps there are 4 entries, the "final quarter" is one entry, the
`max(2, ...)` floor kicks in, and the test is now comparing two points a third
of the run apart. The same run logged with `every=100` gives a completely
different verdict.

**The fix:** measure the slope **per step**, not per logged entry —
`(loss[-2] - loss[-1]) / (step[-1] - step[-2])`, compared against a threshold
scaled by the loss level. Logging frequency should not change a diagnosis.

✅ **The others, for students who pick them:**

| Threshold | How to break it | Better |
|---|---|---|
| `1.10` overfit | The `enough data` run *is* past its minimum by step 4000; a 1.02 threshold flags it, 1.10 does not | Compare against the noise level of the val curve, e.g. 2× its step-to-step standard deviation |
| `0.95` chance | A genuinely hard task can sit just under chance for thousands of steps and then learn | Require it to *stay* there for N evaluations |
| `10 × chance` | Catches `nan` late; a run can be unrecoverable at 3× | Flag the first *increase* over a window, not an absolute level |

**Marking note.** The real lesson is that every one of these is a judgement call
dressed as a constant, and a student who says "all four are arbitrary and should
be derived from the variance of the curve" has got further than the notebook
did.

---

## 16-2 · Consistently labelled outliers

> Change `y_outlier` to `[0.0, 0.0]`, so the extreme points are now labelled
> *consistently* with `sign(x1 * x2)`. Does the spike go away? Print the
> per-parameter norms for both labellings and for an ordinary batch, and explain
> which parameter the spike lives in and why.

✅ **No. The spike barely shrinks, and the damage is almost identical.** Measured
after 50 ordinary steps at `lr=0.3`, `seed=0`:

| Batch | ‖∇W1‖ | ‖∇W2‖ | total | val loss after one step |
|---|---|---|---|---|
| ordinary (`X_small`) | — | — | **0.10** | — |
| outlier, mislabelled `[1, 1]` | 24.42 | 1.58 | **24.48** | 0.5045 → 0.8454 |
| outlier, consistent `[0, 0]` | 15.20 | 1.60 | **15.29** | 0.5045 → 0.8221 |

This is the result almost nobody predicts, and the exercise exists for it.

✅ **Where the spike lives: `W1`, the first layer.** Its gradient is
`X.T @ dh` — the input matrix multiplies in directly. With `|x| = 40` against
training inputs of order 1, the first-layer gradient is inflated by roughly that
factor **whatever the label says**. `W2`'s gradient is `h.T @ dz`, and `h` is a
`tanh` output bounded in [−1, 1], so it stays ordinary in both rows. The spike is
a property of the **input magnitude**, not of the error.

✅ **Why the error does not rescue the consistent case either.** Print `z` for
those two points and you get ≈ −2.42 and +2.17, so `sigmoid(z)` ≈ 0.08 and 0.90.
With labels `[0, 0]` the errors are 0.08 and **0.90** — the model is confidently
wrong on one of them regardless, because nothing in 40 training points in
[−3, 3] ever constrained its behaviour at (−40, 40). **Out-of-distribution
inputs produce arbitrary predictions**, so "correctly labelled" is no protection.

**The transferable point, and it is the opposite of the usual summary:** a
gradient-norm spike means *this batch is unlike the others*, not *this batch is
mislabelled*. In a real run the causes are an abnormally long example, a
tokenizer failure producing a wall of `<unk>`, a corrupt record, or a genuine
distribution shift — mislabelling is only one of them. Go and look at the batch;
the norm tells you which one to look at, not what is wrong with it.

**Push them further:** what does this say about clipping? It caps the damage
without removing the cause, and it works here because the problem is magnitude —
exactly the thing a norm threshold can see. Log the norm *before* clipping, as
the notebook's `train` does, or you destroy the signal while fixing the symptom.

**Marking note.** Credit the measurement over the prediction. A student who
predicted "the spike disappears", ran it, and reported "it does not, and here is
why" has done the thing this whole notebook is about.

---

## 16-3 · Seed-to-seed variance

> Run `train` twice with the same arguments but different `seed`, and record
> both with `run_log`. How large is the seed-to-seed difference in
> `best_val_loss`? Now state the rule it implies for comparing two
> hyperparameter settings.

🔬 **Expected: a difference of a few hundredths to a tenth** on the small-data
run, which is the same order as the gap between neighbouring learning rates in
16.6's sweep. The exact value depends on which seeds they pick, and the variance
itself varies — students should run **more than two**.

✅ **The rule:** a hyperparameter change is only believable when it beats the
seed-to-seed spread. In practice: run each setting at ≥3 seeds and compare
distributions, not single numbers. One run per setting is a coin flip with extra
steps.

**Why it matters more than it sounds.** This is the mechanism behind a large
share of published and internal "improvements" that do not reproduce. Nobody
cheated; they ran one seed per arm and reported the winner.

**Push them further:** ask what they would do if three seeds per setting is
unaffordable — which it will be with a real fine-tune. The honest answer is to
**fix the seed and report that you did**, which does not establish that the
change helps but at least does not claim it does. A student who reaches
"then I should not claim the improvement is real" has the point.
