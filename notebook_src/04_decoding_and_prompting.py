# %% [markdown]
# # 04 · Decoding, thinking mode, and prompts that work
#
# **Day 1 evening / Day 2 morning · ~75 minutes**
#
# The model gives you a probability distribution. *You* decide how to turn it
# into words. Most "the model is bad" complaints are really "the sampling
# settings are wrong".
#
# By the end you will be able to:
#
# 1. explain temperature, top-p, top-k and min-p by looking at real numbers;
# 2. pick settings deliberately for factual vs. creative tasks;
# 3. use Qwen3's thinking mode and know when it is worth the tokens;
# 4. apply prompt patterns that measurably improve a small model.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from qwen_workshop.chat import chat
from qwen_workshop.config import SamplingParams
from qwen_workshop.loading import load_model
from qwen_workshop.utils import wrap_print

qwen = load_model("Qwen/Qwen3-0.6B")
print(qwen)

# %% [markdown]
# ## 4.1 · Look at the actual distribution
#
# Let's stop talking about "probabilities over the vocabulary" abstractly and
# print them.

# %%
def next_token_distribution(text: str, top: int = 10) -> None:
    """Show the model's top candidate next tokens and their probabilities."""
    inputs = qwen.tokenizer(text, return_tensors="pt").to(qwen.model.device)
    with torch.inference_mode():
        logits = qwen.model(**inputs).logits[0, -1, :]  # last position only
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, top)

    print(f"After: {text!r}\n")
    print(f"{'token':<18}{'probability':>12}   bar")
    print("-" * 60)
    for prob, idx in zip(values.tolist(), indices.tolist()):
        token = qwen.tokenizer.decode([idx]).replace("\n", "\\n")
        print(f"{token!r:<18}{prob:>11.3%}   {'#' * int(prob * 40)}")
    print(f"\n(vocabulary has {len(probs):,} tokens; the rest share "
          f"{1 - values.sum().item():.1%})")


next_token_distribution("The capital of Kazakhstan is")

# %%
# A genuinely uncertain continuation looks completely different.
next_token_distribution("My favourite colour is")

# %% [markdown]
# **This is the whole game.** The first distribution is sharp (the model is
# confident); the second is flat (many plausible options). Sampling parameters
# decide how much of that tail you allow.

# %% [markdown]
# ## 4.2 · Temperature
#
# Temperature $T$ rescales the logits before softmax:
#
# $$P(t_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
#
# - $T \to 0$: the top token takes all the probability → deterministic, repetitive.
# - $T = 1$: the model's own calibrated distribution.
# - $T > 1$: flattened → adventurous, eventually incoherent.
#
# Temperature does not add knowledge. It only changes how boldly you sample
# from what is already there.

# %%
def temperature_effect(text: str, temperatures=(0.1, 0.7, 1.0, 1.5, 2.0), top: int = 5) -> None:
    inputs = qwen.tokenizer(text, return_tensors="pt").to(qwen.model.device)
    with torch.inference_mode():
        logits = qwen.model(**inputs).logits[0, -1, :].float()

    base_idx = torch.topk(torch.softmax(logits, -1), top).indices
    header = "".join(f"{f'T={t}':>10}" for t in temperatures)
    print(f"{'token':<16}{header}")
    print("-" * (16 + 10 * len(temperatures)))
    for idx in base_idx.tolist():
        token = qwen.tokenizer.decode([idx]).replace("\n", "\\n")
        row = "".join(f"{torch.softmax(logits / t, -1)[idx].item():>9.2%} " for t in temperatures)
        print(f"{token!r:<16}{row}")


temperature_effect("My favourite colour is")

# %% [markdown]
# Read a row left to right: at `T=0.1` the top token dominates; at `T=2.0` the
# probabilities flatten out and unlikely tokens become real possibilities.

# %% [markdown]
# ### The same effect in generated text

# %%
prompt = "Write one sentence describing a winter morning in Astana."

for temp in (0.1, 0.7, 1.3):
    print(f"--- temperature = {temp} ---")
    for run in range(2):
        out = chat(qwen, prompt, sampling=SamplingParams(temperature=temp, top_p=0.95,
                                                         max_new_tokens=60))
        wrap_print(out.text)
    print()

# %% [markdown]
# At `T=0.1` the two runs are nearly identical. At `T=1.3` they diverge — and
# quality becomes a lottery.

# %% [markdown]
# ## 4.3 · top-p, top-k and min-p
#
# These *truncate* the distribution before sampling, cutting off the long tail
# of nonsense.
#
# | Parameter | What it keeps | Character |
# |---|---|---|
# | **top-k** = 20 | the 20 most likely tokens | fixed width, ignores confidence |
# | **top-p** = 0.8 | smallest set whose probability sums to 0.8 | adapts: narrow when confident, wide when not |
# | **min-p** = 0.05 | tokens with ≥ 5% of the top token's probability | adapts, robust at high temperature |
#
# In practice you combine them. Qwen's own recommendations:
#
# | Mode | temperature | top_p | top_k | min_p |
# |---|---|---|---|---|
# | Thinking | 0.6 | 0.95 | 20 | 0 |
# | Non-thinking | 0.7 | 0.8 | 20 | 0 |
#
# **Using the wrong preset for the mode is the most common cause of repetitive
# or rambling Qwen3 output.**

# %%
def show_truncation(text: str, top_p: float = 0.8, top_k: int = 20) -> None:
    """How many tokens survive each filter?"""
    inputs = qwen.tokenizer(text, return_tensors="pt").to(qwen.model.device)
    with torch.inference_mode():
        probs = torch.softmax(qwen.model(**inputs).logits[0, -1, :].float(), -1)

    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    n_top_p = int((cumulative < top_p).sum().item()) + 1
    n_min_p = int((sorted_probs >= 0.05 * sorted_probs[0]).sum().item())

    print(f"{text!r}")
    print(f"  full vocabulary : {len(probs):,} tokens")
    print(f"  top_k={top_k}       : {top_k} tokens")
    print(f"  top_p={top_p}      : {n_top_p} tokens  <- adapts to confidence")
    print(f"  min_p=0.05      : {n_min_p} tokens")
    print()


show_truncation("The capital of Kazakhstan is")   # confident -> few tokens survive
show_truncation("My favourite colour is")          # uncertain -> many survive

# %% [markdown]
# That contrast is exactly why top-p is usually preferred over top-k: it is
# tight when the model knows the answer and generous when it genuinely does not.
#
# > **Exercise 1.** Generate the same prompt 5 times with `temperature=1.5,
# > top_p=1.0` (no truncation) and 5 times with `temperature=1.5, top_p=0.9`.
# > Count how many outputs in each group are coherent.

# %% [markdown]
# ## 4.4 · When to use what
#
# | Task | Settings | Why |
# |---|---|---|
# | Extraction, classification, JSON | `do_sample=False` (greedy) | you want the same answer every time |
# | Factual Q&A, RAG | `T=0.3, top_p=0.8` | mild diversity, low invention |
# | General chat | `T=0.7, top_p=0.8` | Qwen's non-thinking default |
# | Reasoning / maths | `T=0.6, top_p=0.95` | Qwen's thinking default |
# | Brainstorming, fiction | `T=1.0, top_p=0.95` | you *want* surprises |
#
# Reproducibility: seed everything and use greedy decoding.

# %%
from qwen_workshop.config import GREEDY_SAMPLING
from qwen_workshop.utils import seed_everything

question = "List exactly three uses of a thermometer, numbered."
for run in range(2):
    seed_everything(613)
    out = chat(qwen, question, sampling=GREEDY_SAMPLING.replace(max_new_tokens=80))
    print(f"run {run + 1}:")
    wrap_print(out.text)
    print()

# %% [markdown]
# Identical output. Note greedy decoding is **not** recommended for Qwen3
# thinking mode — it tends to fall into repetition loops — but it is ideal for
# structured extraction.

# %% [markdown]
# ## 4.5 · Thinking mode
#
# Qwen3 hybrid models can generate a private reasoning block before answering:
#
# ```
# <think>
# The user asks ... let me work through it step by step ...
# </think>
# The answer is 42.
# ```
#
# You show the user only the part after `</think>`. The reasoning costs tokens
# and time, and buys accuracy on multi-step problems.
#
# **Availability matters:** the switch exists on hybrid checkpoints
# (Qwen3-0.6B/1.7B/4B/8B/14B/32B). The later `-Instruct-2507` and
# `-Thinking-2507` releases split the two modes into separate models, and
# Qwen2.5 has no switch at all. Our helper checks before using it.

# %%
from qwen_workshop.chat import supports_thinking_switch

print("this checkpoint supports enable_thinking:", supports_thinking_switch(qwen.tokenizer))

# %%
puzzle = (
    "A shop sells pens in packs of 5 and notebooks in packs of 3. "
    "Aisulu bought 4 packs of pens and 6 packs of notebooks. "
    "She gave 7 pens and 4 notebooks to friends. How many of each does she have left?"
)

print("=== WITHOUT thinking ===")
fast = chat(qwen, puzzle, enable_thinking=False)
wrap_print(fast.text)
print(f"\n{fast.completion_tokens} tokens, {fast.seconds:.1f}s\n")

print("=== WITH thinking ===")
slow = chat(qwen, puzzle, enable_thinking=True,
            sampling=SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=1024))
print("[reasoning, normally hidden from the user]")
wrap_print(slow.thinking or "(none produced)")
print("\n[the answer the user sees]")
wrap_print(slow.text)
print(f"\n{slow.completion_tokens} tokens, {slow.seconds:.1f}s")

# %% [markdown]
# Correct answer: 4×5 − 7 = **13 pens**, 6×3 − 4 = **14 notebooks**.
#
# Check both. A 0.6B model often gets this wrong without thinking and right
# with it. Also note the cost: thinking can use 5–20× more tokens.
#
# ### When thinking is worth it
#
# | Use thinking | Skip thinking |
# |---|---|
# | Maths, logic, multi-step deduction | Greetings, rewriting, translation |
# | Code that must be correct first time | Classification into fixed labels |
# | Planning an agent's next steps | Anything already constrained by a schema |
# | Ambiguous questions needing analysis | Latency-sensitive UI |
#
# You can also steer it inline: append `/think` or `/no_think` to a user message
# on hybrid Qwen3 models.

# %%
for tag in ("/no_think", "/think"):
    out = chat(qwen, f"What is 17 * 23? {tag}", enable_thinking=True,
               sampling=SamplingParams(temperature=0.6, top_p=0.95, max_new_tokens=600))
    thought = f"{len(out.thinking.split())} words of reasoning" if out.thinking else "no reasoning"
    print(f"{tag:<10} -> {thought}; answer: {out.text[:90]!r}")

# %% [markdown]
# (17 × 23 = 391, in case you want to check.)

# %% [markdown]
# ## 4.6 · Prompt patterns that actually help small models
#
# A 0.6B model is not a mind reader. These four patterns give the biggest
# improvement per minute of effort.

# %% [markdown]
# ### Pattern 1 — Be specific about the output shape
#
# Vague in, vague out.

# %%
vague = "Tell me about lentils."
specific = (
    "In exactly 3 bullet points, each under 15 words, describe lentils for "
    "someone planning a cheap student dinner. Cover: cost, cooking time, protein."
)

for label, p in [("vague", vague), ("specific", specific)]:
    print(f"--- {label} ---")
    wrap_print(chat(qwen, p, sampling=SamplingParams(temperature=0.3, max_new_tokens=150)).text)
    print()

# %% [markdown]
# ### Pattern 2 — Give an example (few-shot)
#
# One worked example beats three paragraphs of instructions, especially for
# format compliance.

# %%
few_shot = """Classify each expense into: groceries, transport, eating out, utilities.

Input: "Small Magnum, 4200 KZT"
Output: groceries

Input: "Yandex Go, 1800 KZT"
Output: transport

Input: "Coffee Boom, 1500 KZT"
Output:"""

print(chat(qwen, few_shot, sampling=GREEDY_SAMPLING.replace(max_new_tokens=10)).text)

# %% [markdown]
# ### Pattern 3 — Give the model an escape hatch
#
# Models invent answers partly because nothing in the prompt permits them not
# to. Say explicitly that "I don't know" is an acceptable answer.

# %%
unanswerable = "What was the exact attendance at the ROBT613 lecture on 3 March 2019?"

print("--- without escape hatch ---")
wrap_print(chat(qwen, unanswerable, sampling=SamplingParams(temperature=0.3)).text)

print("\n--- with escape hatch ---")
wrap_print(chat(
    qwen, unanswerable,
    system="If you do not know something, reply exactly: 'I don't know.' "
           "Never guess at specific numbers, names or dates.",
    sampling=SamplingParams(temperature=0.3),
).text)

# %% [markdown]
# ### Pattern 4 — Put instructions *after* the data
#
# For long inputs, models attend most reliably to the beginning and the end.
# Data in the middle, instruction at the end, is a reliable win.

# %%
document = (REPO_ROOT / "data" / "notes" / "cooking.md").read_text(encoding="utf-8")

print(chat(
    qwen,
    f"<document>\n{document}\n</document>\n\n"
    "Using ONLY the document above, list every mistake the author says they "
    "keep making. Use a numbered list. If there are none, say so.",
    sampling=SamplingParams(temperature=0.3, max_new_tokens=250),
).text)

# %% [markdown]
# ## 4.7 · Mini-lab
#
# Build a "rewrite this message" tool and tune it.
#
# 1. Write a system prompt that rewrites a blunt message politely.
# 2. Choose sampling settings and justify them in one sentence.
# 3. Test on the three messages below.
# 4. Find one input where it fails, and fix the prompt.

# %%
TONE_SYSTEM = """You rewrite short messages to be polite and professional.

Rules:
- Keep the original meaning and all facts exactly.
- Keep it under 40 words.
- Output ONLY the rewritten message, with no preamble or explanation.
"""

drafts = [
    "Your report is late again. Send it now.",
    "I can't come to the meeting, it's pointless anyway.",
    "This code is a mess, who wrote this?",
]

for draft in drafts:
    out = chat(qwen, draft, system=TONE_SYSTEM,
               sampling=SamplingParams(temperature=0.5, top_p=0.9, max_new_tokens=80))
    print(f"before: {draft}")
    print("after : ", end="")
    wrap_print(out.text)
    print()

# %% [markdown]
# > **Exercise 2.** The model will sometimes add "Sure, here is the rewritten
# > message:" despite being told not to. Fix it with prompting. Then note: in
# > notebook 07 we make this structurally impossible with constrained decoding.
#
# ## Checkpoint
#
# 1. What does temperature do to the logits, mathematically?
# 2. Why is top-p usually preferable to top-k?
# 3. Give two tasks where thinking mode pays for itself, and two where it does not.
# 4. Name the four prompt patterns and give one example each.
#
# ➡️ **Next:** [`05_quantization_and_local_runtimes.ipynb`](05_quantization_and_local_runtimes.ipynb) —
# shrink the model so bigger ones fit on your laptop.
