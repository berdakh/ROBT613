# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407LLM/blob/master/notebooks/13_pretrain_a_tiny_model.ipynb)
#
# *Click the badge to run this notebook in Google Colab - no install required. The first code cell sets everything up.*

# %% [markdown]
# # 13 · Pre-train a model from scratch
#
# **Training track · ~75 minutes · runs on a CPU**
#
# Handbook chapter 3 says pre-training installs knowledge, and that knowledge is
# a *side effect* of compressing text. That is a strange claim to accept on
# faith. In this notebook you watch it happen.
#
# You will build a small transformer, train it on a few hundred kilobytes of
# text, and sample from it every few hundred steps. It goes from random
# characters, to things that look like words, to actual words and punctuation —
# in about five minutes on a laptop CPU.
#
# By the end you will be able to:
#
# 1. explain next-token prediction as a loss you have personally watched fall;
# 2. name every component of a transformer block because you wrote it;
# 3. read a training curve and say what it means;
# 4. explain why this exact procedure, scaled up 10 million times, produces
#    something that knows facts.
#
# **This is the only notebook where you train a model from random weights.**
# Everything else in the workshop adapts somebody else's.

# %%
# --- Setup: works on your laptop AND in Google Colab ------------------------
import os
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    if not Path("/content/ROBT407LLM").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/ROBT407LLM.git",
                        "/content/ROBT407LLM"], check=True)
    os.chdir("/content/ROBT407LLM")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], check=True)

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "src" / "qwen_workshop").is_dir())
sys.path.insert(0, str(REPO_ROOT / "src"))
print("repo root:", REPO_ROOT)

# %%
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(613)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} on {DEVICE}")
if DEVICE == "cpu":
    print(f"{torch.get_num_threads()} CPU threads - this notebook is sized for that")

# %% [markdown]
# ## 13.1 · The corpus
#
# We train on the workshop's own documentation. It is a few hundred kilobytes
# of English prose with a consistent vocabulary — small enough to train on in
# minutes, structured enough that progress is obvious when you sample from it.
#
# Real pre-training uses 10–15 *trillion* tokens. We use about 300,000
# characters. The procedure is identical; only the scale differs, and that
# difference is the entire story of why large models know things and this one
# will not.

# %%
def load_corpus() -> str:
    """All the workshop's markdown, concatenated."""
    import re

    parts = []
    for path in sorted((REPO_ROOT / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)  # front matter
        text = re.sub(r"<img[^>]*>", "", text)                          # image tags
        parts.append(text)
    return "\n\n".join(parts)


corpus = load_corpus()
print(f"{len(corpus):,} characters")
print(f"{len(corpus.split()):,} words")
print("\n--- a sample ---")
print(corpus[2000:2400])

# %% [markdown]
# ## 13.2 · A tokenizer, built by hand
#
# Notebook 01 used Qwen's tokenizer. Here we build our own, and the simplest
# one that works: **one token per character**.
#
# This is a real design choice with real consequences. A character tokenizer
# has a tiny vocabulary (~100 entries instead of 150,000), so the model's
# embedding and output layers are small — which is exactly what we want on a
# CPU. The cost is that sequences are ~4× longer in tokens, so the model has to
# look further back to see the same amount of text.
#
# Production models use subword tokenizers (BPE) precisely to buy back that
# sequence length.

# %%
vocabulary = sorted(set(corpus))
VOCAB_SIZE = len(vocabulary)
stoi = {ch: i for i, ch in enumerate(vocabulary)}
itos = dict(enumerate(vocabulary))


def encode(text: str) -> list[int]:
    return [stoi[c] for c in text if c in stoi]


def decode(ids) -> str:
    return "".join(itos[int(i)] for i in ids)


print(f"vocabulary: {VOCAB_SIZE} characters")
print(f"sample    : {''.join(vocabulary[:70])!r}")
print()
print("encode('hello') ->", encode("hello"))
print("decode(...)     ->", decode(encode("hello")))

# %%
# The whole corpus as one long tensor of token ids.
data = torch.tensor(encode(corpus), dtype=torch.long)

# Hold out the last 5% to measure generalisation. Without this you cannot tell
# learning from memorisation - which is the entire point of section 13.7.
split = int(0.95 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"train: {len(train_data):,} tokens")
print(f"val  : {len(val_data):,} tokens")

# %% [markdown]
# ## 13.3 · The training objective, made concrete
#
# Pre-training predicts the next token. Here is exactly what one training
# example looks like — and notice that a single sequence of length `n` gives
# the model `n` separate predictions to learn from, which is why this scales so
# well.

# %%
BLOCK_SIZE = 128  # how many tokens of context the model sees

example = train_data[:9]
print("Given a sequence of 9 tokens, the model gets 8 training signals:\n")
for i in range(1, 9):
    context, target = example[:i], example[i]
    print(f"  context {decode(context)!r:<22} -> predict {decode([target])!r}")

# %%
def get_batch(split: str, batch_size: int = 32):
    """Sample random windows from the corpus.

    x is the context, y is x shifted one position left: for every position in
    x, y holds the token that actually came next. No labels, no annotation -
    the text supplies its own answer. That is why pre-training scales.
    """
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


xb, yb = get_batch("train", batch_size=4)
print(f"x: {tuple(xb.shape)}   y: {tuple(yb.shape)}")
print(f"\nx[0] starts: {decode(xb[0][:40])!r}")
print(f"y[0] starts: {decode(yb[0][:40])!r}   <- the same text, shifted by one")

# %% [markdown]
# ## 13.4 · The model
#
# Now build the thing from handbook chapter 2's diagram. Every component is
# here and nothing is hidden.

# %%
class SelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Each position computes a query, and compares it against the keys of every
    position *at or before* it. The mask is what makes it causal: without it
    the model could see the answer it is being asked to predict, and the loss
    would drop to zero while learning nothing.
    """

    def __init__(self, n_embed: int, n_heads: int):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim): each head attends independently
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention, with the causal mask applied for us
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    """One transformer block: attention, then a feed-forward network.

    Both are wrapped in a residual connection (`x = x + ...`) and preceded by
    normalisation. The residual is what lets gradients reach the early layers
    of a deep stack; without it, depth stops helping.
    """

    def __init__(self, n_embed: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = SelfAttention(n_embed, n_heads)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """A complete language model. About 90 lines of the notebook so far."""

    def __init__(self, vocab_size: int, n_embed: int = 128,
                 n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        # Tie the input and output embeddings: the same matrix maps ids to
        # vectors and vectors back to scores. Saves parameters, usually helps.
        self.head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_final(x))

        loss = None
        if targets is not None:
            # Cross-entropy over every position at once: T predictions per
            # sequence, which is what makes the objective so data-efficient.
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.reshape(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 200, temperature: float = 0.8):
        """The generation loop from notebook 01, in eleven lines."""
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -BLOCK_SIZE:]       # never exceed the context
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :] / temperature  # only the last position
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = TinyLM(VOCAB_SIZE).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters  ({n_params / 1e6:.2f}M)")
print(f"\nQwen3-0.6B has {600e6 / n_params:,.0f}x more.")

# %% [markdown]
# ## 13.5 · What an untrained model produces
#
# Before any training, the weights are random. Sampling gives you the
# distribution of nothing in particular.

# %%
start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(repr(decode(model.generate(start, max_new_tokens=200)[0])))

# %% [markdown]
# Note the loss we should expect at this point. With `V` equally likely
# characters, a model that knows nothing has loss `ln(V)`. If your starting
# loss is far from this, something is wrong with the setup.

# %%
_, initial_loss = model(*get_batch("train"))
print(f"loss at initialisation : {initial_loss.item():.3f}")
print(f"ln(vocab_size) = ln({VOCAB_SIZE}) : {math.log(VOCAB_SIZE):.3f}")
print("\nThey match, so the model starts out genuinely uncertain.")

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT407LLM/master/docs/assets/diagrams/tr-training-loop.svg" alt="One training step: batch, forward, loss, backward, weight update, then repeat. Gradient clipping and the learning-rate schedule modify the update. The loop reports train loss, validation loss, learning rate and gradient norm." width="100%">
#
# ### In plain terms
#
# The training loop has four pieces of jargon and they are all everyday ideas:
#
# | | Think of it as |
# |---|---|
# | **Loss** | Your score on a practice test — except **lower is better**. |
# | **Gradient** | Which way to turn each dial, and how hard, to score better. |
# | **Learning rate** | **Stride length.** Too short and you never arrive; too long and you stride past the door. |
# | **Epoch** | One full pass through the textbook. |
#
# And the thing people get wrong: a model that has learned *nothing* does not
# score zero. It scores `ln(vocabulary size)` — the score you get for guessing
# uniformly. Knowing that number is how you tell "learning slowly" from
# "not learning at all".

# ## 13.6 · Train it
#
# Now the loop. Read it once: this is the whole of pre-training, and the only
# difference at GPT scale is the number of zeros.

# %%
@torch.no_grad()
def estimate_loss(eval_iters: int = 50) -> dict:
    """Average loss over several batches - a single batch is too noisy to read."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


MAX_STEPS = 2000 if DEVICE == "cpu" else 5000
EVAL_EVERY = 250

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
# Cosine decay with a short warmup: the standard schedule, and it matters more
# than people expect. A constant high LR plateaus early.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=MAX_STEPS, pct_start=0.1
)

history = []
start_time = time.time()

for step in range(MAX_STEPS + 1):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        history.append((step, losses["train"], losses["val"]))
        elapsed = time.time() - start_time
        print(f"step {step:>5}  train {losses['train']:.4f}  "
              f"val {losses['val']:.4f}  ({elapsed:.0f}s)")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Clip gradients: one bad batch can otherwise blow up the weights and
    # undo an hour of training in a single step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

print(f"\ndone in {time.time() - start_time:.0f}s")

# %% [markdown]
# ## 13.7 · What it learned
#
# Sample again. The same code, the same random seed behaviour — only the
# weights changed.

# %%
print(repr(decode(model.generate(start, max_new_tokens=400)[0])))

# %% [markdown]
# It will not be coherent. It *should* produce real words, plausible
# punctuation, markdown-ish structure, and the vocabulary of the corpus
# ("model", "token", "the"). Compare that with the noise from section 13.5.
#
# **Nobody told it what a word is.** There is no dictionary, no spell-checker,
# no grammar rule anywhere in the code. The only instruction was *predict the
# next character*, and spelling turned out to be the cheapest way to do that.
#
# That is the claim from handbook chapter 3, demonstrated: structure is a side
# effect of compression. Scale this up, and the cheapest way to predict the next
# token starts to require knowing that Astana is the capital of Kazakhstan.

# %% [markdown]
# ## 13.8 · Read the curve
#
# The two numbers you printed every 250 steps are the most important diagnostic
# in all of training.

# %%
print(f"{'step':>6}{'train':>9}{'val':>9}{'gap':>9}")
print("-" * 33)
for step, train_loss, val_loss in history:
    print(f"{step:>6}{train_loss:>9.4f}{val_loss:>9.4f}{val_loss - train_loss:>9.4f}")

try:
    import matplotlib.pyplot as plt

    steps = [h[0] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, [h[1] for h in history], label="train")
    plt.plot(steps, [h[2] for h in history], label="validation")
    plt.axhline(math.log(VOCAB_SIZE), ls="--", lw=1, c="grey", label="random guessing")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.legend()
    plt.title("Pre-training a 0.4M-parameter model")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\n(pip install matplotlib for the plot)")

# %% [markdown]
# **How to read it:**
#
# | Pattern | Meaning | What to do |
# |---|---|---|
# | Both falling together | Learning. Keep going. | nothing |
# | Both flat near `ln(V)` | Not learning at all | check the learning rate, check the mask |
# | Train falls, **val rises** | **Overfitting** — memorising the corpus | stop, or get more data |
# | Both flat but low | Converged for this capacity | bigger model, or more data |
# | Sudden spike | A bad batch, or the LR is too high | lower the LR, check gradient clipping |
#
# With 300k characters and 0.4M parameters you are close to the overfitting
# boundary. If your validation loss turned upward, you watched memorisation
# begin — which is exactly what deduplication (handbook 3, phase 0) exists to
# prevent at scale.
#
# > **Exercise 1.** Train for 6000 steps instead of 2000. Does validation loss
# > turn upward? At which step? That step is where this model stops learning
# > and starts memorising.

# %% [markdown]
# > **In plain terms.** Rising validation loss with falling training loss is
# > **memorising past exam papers**: perfect on the ones you have seen, lost on
# > the real thing. The fix is more papers, not more revision.

# ## 13.9 · Scale is the only difference
#
# You have now done, end to end, what costs labs millions of dollars. The
# procedure was identical. Here is what separates them.

# %%
ours = {"params": n_params, "tokens": len(train_data), "steps": MAX_STEPS}
gpt3 = {"params": 175e9, "tokens": 300e9, "steps": 300_000}
llama3_8b = {"params": 8e9, "tokens": 15e12, "steps": 1_000_000}

print(f"{'':<16}{'parameters':>14}{'train tokens':>16}{'tokens/param':>14}")
print("-" * 60)
for name, m in [("this notebook", ours), ("GPT-3 (2020)", gpt3), ("Llama 3 8B", llama3_8b)]:
    print(f"{name:<16}{m['params']:>14,.0f}{m['tokens']:>16,.0f}"
          f"{m['tokens'] / m['params']:>14,.1f}")

print(f"\nLlama 3 8B saw {llama3_8b['tokens'] / ours['tokens']:,.0f}x more text "
      f"than we did,\nwith {llama3_8b['params'] / ours['params']:,.0f}x more parameters to store it in.")

# %% [markdown]
# Look at the last column against handbook chapter 3's scaling-laws section.
# GPT-3 used ~2 tokens per parameter; Chinchilla found ~20 to be
# compute-optimal; Llama 3 8B used nearly 2,000 because **inference cost
# dominates** once a model is trained. That is why good small models exist.
#
# > **Exercise 2.** Double `n_embed` to 256 and halve `MAX_STEPS`. You now have
# > roughly 4× the parameters and half the training. Does validation loss end
# > up better or worse? This is the scaling-law trade-off, at a scale you can
# > actually run.

# %% [markdown]
# ## 13.10 · What this model is not
#
# Be clear about what you built, because the gap is the rest of the workshop:
#
# | This model | A model you would use |
# |---|---|
# | Completes text | Follows instructions (phase 2 installs that) |
# | 0.4M parameters | 0.6B–400B |
# | 300k characters of one website | 10–15 trillion tokens |
# | Character tokens | Subword (BPE) tokens |
# | No chat template, no roles | ChatML, system prompts, tools |
# | Knows the shape of English | Knows facts |
#
# **This is a base model.** Exactly the artefact handbook chapter 3 describes
# coming out of phase 1 — and exactly why a `-Base` checkpoint rambles when you
# ask it a question.
#
# Notebook 14 builds the dataset that turns one into an assistant.
#
# ## Checkpoint
#
# 1. What is the training objective, in one sentence?
# 2. Why does one sequence of length `n` give `n` training signals?
# 3. What does the causal mask prevent, and what would happen without it?
# 4. What loss does an untrained model have, and why that number?
# 5. What does rising validation loss with falling training loss mean?
# 6. Name three things this model lacks that an instruct model has.
#
# ➡️ **Next:** [`14_building_a_training_dataset.ipynb`](14_building_a_training_dataset.ipynb)
