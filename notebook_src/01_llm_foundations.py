# %% [markdown]
# # 01 · What an LLM actually is
#
# **Day 1 · ~60 minutes**
#
# By the end of this notebook you will be able to:
#
# 1. explain, without hand-waving, what a language model computes;
# 2. tokenise text with the real Qwen tokenizer and read the output;
# 3. explain why "how many r's in strawberry" is hard and why that is *not*
#    a sign the model is stupid;
# 4. explain what "open weights" means and how it differs from "open source".
#
# **Nothing here downloads a multi-gigabyte model.** We only fetch a tokenizer
# (a few megabytes), so this notebook runs on any laptop.

# %% [markdown]
# ## 1.1 · The whole idea in one sentence
#
# > A language model takes a sequence of tokens and outputs a probability
# > distribution over which token comes next.
#
# That is it. That is the entire mechanism. Everything you have heard about —
# chat, reasoning, tool use, agents — is built on repeated application of that
# one operation.
#
# Text generation is therefore a loop:
#
# ```
# tokens = tokenize("The capital of Kazakhstan is")
# repeat:
#     probabilities = model(tokens)      # a score for every token in the vocabulary
#     next_token    = pick(probabilities) # "sampling" - notebook 04 is all about this
#     tokens.append(next_token)
# until next_token == end_of_text or we hit a length limit
# ```
#
# Three consequences that explain most LLM behaviour:
#
# | Because... | ...this follows |
# |---|---|
# | The model only ever predicts the *next* token | It cannot plan ahead or revise. "Thinking out loud" (notebook 04) is how it buys itself room to plan. |
# | The unit is a token, not a letter | Counting letters is genuinely hard for it. |
# | Output is a *distribution*, and we sample from it | The same prompt gives different answers. That is a setting, not a bug. |

# %%
# Setup: make the workshop package importable from anywhere in the repo.
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import qwen_workshop

print("workshop package version:", qwen_workshop.__version__)
print("repo root:", REPO_ROOT)

# %% [markdown]
# ## 1.2 · Tokens: the units a model really sees
#
# Install the one dependency we need if it is missing. `transformers` brings
# the tokenizer; we are not loading any weights yet.

# %%
# If this fails, run in a terminal:  pip install -r requirements.txt
try:
    from transformers import AutoTokenizer

    print("transformers is installed")
except ImportError:
    print("Installing transformers (one-off, ~1 minute)...")
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)
    from transformers import AutoTokenizer

# %%
from qwen_workshop.loading import load_tokenizer

# Downloads ~10 MB of tokenizer files on first run, then caches them.
tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")

print(f"vocabulary size : {tokenizer.vocab_size:,} tokens")
print(f"model max length: {tokenizer.model_max_length:,} tokens")

# %% [markdown]
# ### Look at a real tokenisation
#
# The rule of thumb for English is **1 token ≈ 4 characters ≈ 0.75 words**.
# Watch where the boundaries actually fall — and note the leading spaces,
# which are part of the token.

# %%
def show_tokens(text: str, tokenizer=tokenizer) -> None:
    """Print each token with its id, so the boundaries are visible."""
    ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"{text!r}")
    print(f"  {len(text)} characters -> {len(ids)} tokens "
          f"({len(text) / max(len(ids), 1):.1f} chars/token)")
    print("  " + " | ".join(f"{p}" for p in pieces))
    print("  ids: " + " ".join(str(i) for i in ids))
    print()


show_tokens("The capital of Kazakhstan is Astana.")
show_tokens("Tokenization")
show_tokens("antidisestablishmentarianism")

# %% [markdown]
# ### Not all languages cost the same
#
# Tokenizers are trained on text, and that text is mostly English. Languages
# written in other scripts get chopped into more, smaller pieces — which means
# **the same sentence costs more tokens, more money and more context window**.
# This is a real fairness issue, not a curiosity.

# %%
samples = {
    "English": "Good morning, how are you today?",
    "Russian": "Доброе утро, как ваши дела сегодня?",
    "Kazakh": "Қайырлы таң, бүгін қалыңыз қалай?",
    "Chinese": "早上好，你今天怎么样？",
    "Code": "for i in range(10): print(i ** 2)",
    "Numbers": "The total was 1234567.89 KZT",
}

print(f"{'language':<10}{'chars':>7}{'tokens':>8}{'chars/token':>13}")
print("-" * 38)
for name, text in samples.items():
    n_chars, n_tokens = len(text), len(tokenizer.encode(text))
    print(f"{name:<10}{n_chars:>7}{n_tokens:>8}{n_chars / n_tokens:>13.2f}")

# %% [markdown]
# **Read the table.** English gets ~4 characters per token. Kazakh and Russian
# usually get 2 or fewer. A Kazakh document can consume twice the context
# window of the same document in English.
#
# > **Exercise 1.** Add Turkish, Arabic or your own language to `samples`.
# > Which is most expensive? Write down the chars/token you measured.

# %% [markdown]
# ## 1.3 · Why letter-counting is hard
#
# The famous "how many r's in strawberry" failure is a *tokenisation* problem.
# The model never sees the letters — it sees two or three opaque chunks.

# %%
for word in ["strawberry", "raspberry", "Mississippi"]:
    ids = tokenizer.encode(word)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"{word:<14} -> {pieces}")
    print(f"{'':<14}    the model sees {len(ids)} opaque chunk(s), not "
          f"{len(word)} letters\n")

# %% [markdown]
# Asking a model to count letters is like asking you to count the brush strokes
# in a word you are reading — the information is technically there, but it is
# not what the representation is built for.
#
# **The fix is not a better model. The fix is a tool.** Notebook 07 gives the
# model a `count_letters` function, and the problem disappears permanently.
# That is the core lesson of the agentic half of this workshop: *do not ask
# the model to do what a three-line Python function does perfectly.*

# %%
# The three-line function in question.
def count_letters(word: str, letter: str) -> int:
    """Count occurrences of `letter` in `word`, case-insensitively."""
    return word.lower().count(letter.lower())


print("strawberry has", count_letters("strawberry", "r"), "r's - always, exactly, for free")

# %% [markdown]
# ## 1.4 · Special tokens and the chat format
#
# A base model continues text. A **chat** model has been fine-tuned to follow a
# strict format with special marker tokens. Qwen uses the ChatML style:
#
# ```
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Hello!<|im_end|>
# <|im_start|>assistant
# ```
#
# The model generates what comes after that final line. Those `<|im_start|>`
# markers are **single tokens**, not literal text — that is what makes it hard
# for a user to impersonate the system role.

# %%
special = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"]
for token in special:
    ids = tokenizer.encode(token, add_special_tokens=False)
    status = "single token" if len(ids) == 1 else f"{len(ids)} tokens (not special here)"
    print(f"{token:<16} id={str(ids):<12} {status}")

# %% [markdown]
# ### The chat template turns messages into that string
#
# **This is the most important cell in the notebook.** Every "why is my model
# behaving strangely" bug should start by printing this.

# %%
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the capital of Kazakhstan?"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # append the "assistant" header so the model replies
    enable_thinking=False,       # Qwen3-specific; see notebook 04
)
print(prompt)
print("---")
print(f"that string is {len(tokenizer.encode(prompt))} tokens")

# %% [markdown]
# > **Exercise 2.** Set `add_generation_prompt=False` and print it again.
# > What disappeared? Why would the model keep silent (or ramble) without it?
#
# > **Exercise 3.** Set `enable_thinking=True`. What extra marker appears?

# %% [markdown]
# ## 1.5 · "Open weights" is not "open source"
#
# You will hear both terms used loosely. The distinction matters legally and
# practically.
#
# | | Open weights | Open source (as software people mean it) |
# |---|---|---|
# | You get the trained parameters | ✅ | ✅ |
# | You get the training code | sometimes | ✅ |
# | You get the **training data** | almost never | ✅ |
# | You can run it offline, unmodified | ✅ | ✅ |
# | You can fine-tune and redistribute | licence-dependent | ✅ |
#
# Nearly every "open" LLM — Qwen, Llama, Mistral, Gemma, DeepSeek — is
# **open weights**. The data and the exact training recipe stay private. You
# cannot reproduce the model from scratch; you can only use and adapt it.
#
# Why this is still a big deal:
#
# - **Privacy.** Your notes, your medical questions, your company's contracts
#   never leave your machine. This is the reason we use your own notes in
#   notebook 09.
# - **Cost.** No per-token billing. The electricity is the cost.
# - **Availability.** No rate limits, no outages, no deprecation of the model
#   you built on.
# - **Control.** You can fine-tune it (notebook 11), quantise it (05), inspect
#   its internals, and pin a version forever.
#
# The trade-off is honest: open-weight models you can run on a laptop are
# **less capable** than the largest hosted models. Much of this workshop is
# about closing that gap with tools, retrieval and structure rather than raw
# model size.
#
# 📖 Full treatment: [`docs/guides/open-weight-models.md`](../docs/guides/open-weight-models.md)

# %% [markdown]
# ## 1.6 · Where Qwen fits
#
# Qwen (通义千问, *Tongyi Qianwen*) is Alibaba's model family. We use it here
# because it is a rare combination of: genuinely strong, released under
# **Apache 2.0** for the main Qwen3 line, available in sizes from 0.6B to
# hundreds of billions, and multilingual in a way that matters in Central Asia.
#
# The 0.6B model we use all workshop is *small*. It will get facts wrong. Treat
# it as a reasoning and formatting engine that you feed with tools and
# retrieval — never as an encyclopaedia.

# %%
from qwen_workshop.config import MODELS

print(f"{'key':<24}{'params':<26}{'kind':<11}{'int4 RAM':>9}")
print("-" * 72)
for key, info in MODELS.items():
    print(f"{key:<24}{info.params:<26}{info.kind:<11}{info.approx_ram_gb_q4:>7.1f} GB")

# %% [markdown]
# ## Checkpoint
#
# You should now be able to answer these without scrolling up:
#
# 1. What does a language model output at each step?
# 2. Why does the same prompt give different answers?
# 3. Why is counting letters hard, and what is the correct fix?
# 4. What does the chat template do, and why print it when debugging?
# 5. Name two things you get from open weights that an API cannot give you.
#
# ➡️ **Next:** [`02_environment_and_hardware.ipynb`](02_environment_and_hardware.ipynb) —
# work out what your specific machine can run, before downloading anything.
