# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407LLM/blob/master/notebooks/21_attention_from_scratch.ipynb)
#
# *Click the badge to run this notebook in Google Colab - no install required. The first code cell sets everything up.*

# %% [markdown]
# # 21 · Attention from scratch
#
# **Mathematics track · ~2 hours · NumPy only, no GPU, no model**
#
# Notebook 13 trained a transformer by calling
# `F.scaled_dot_product_attention`. This notebook opens that function.
#
# Everything here is built from the equations in NumPy and then **proved by
# measurement**: not "attention is a weighted average" but a program that
# demonstrates the causal mask really is causal, that position is invisible
# without RoPE, and that $$\sqrt{d_k}$$ is not a magic constant.
#
# By the end you will be able to:
#
# 1. derive the attention operation from the question it answers;
# 2. say what $$Q$$, $$K$$ and $$V$$ are and why there are three of them;
# 3. **show** why the scores are divided by $$\sqrt{d_k}$$;
# 4. prove a causal mask is causal, and that attention alone is blind to order;
# 5. implement RoPE and demonstrate it encodes *relative* position;
# 6. assemble a full Qwen-style block and count its parameters, FLOPs and KV cache.
#
# 📖 The reading companion, with the derivations set out properly:
# [`docs/handbook/the-mathematics.md`](../docs/handbook/the-mathematics.md)
#
# **Prerequisites:** matrix multiplication, and what a dot product means
# geometrically. Nothing else. No calculus — we never differentiate anything.

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
import numpy as np

rng = np.random.default_rng(613)
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# %% [markdown]
# ## 21.1 · Notation, once
#
# Every symbol used in this notebook. Keep it open; the rest of the maths is
# easier than the notation makes it look.
#
# | Symbol | Shape | What it is |
# |---|---|---|
# | $$n$$ | — | number of tokens in the sequence |
# | $$d_{\text{model}}$$ | — | width of the residual stream (Qwen3-0.6B: 1024) |
# | $$h$$ | — | number of attention heads |
# | $$d_k$$ | — | width of one head, usually $$d_{\text{model}}/h$$ |
# | $$X$$ | $$n \times d_{\text{model}}$$ | the token representations coming in |
# | $$W_Q, W_K, W_V$$ | $$d_{\text{model}} \times d_k$$ | learned projections |
# | $$Q, K, V$$ | $$n \times d_k$$ | queries, keys, values: $$Q = XW_Q$$ etc. |
# | $$A$$ | $$n \times n$$ | attention weights, each row summing to 1 |
#
# **One sentence to hold on to:** a transformer moves information *between
# positions* with attention, and *within a position* with the MLP. Everything
# below is detail on those two moves.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT407LLM/master/docs/assets/diagrams/mt-attention-pipeline.svg" alt="The attention pipeline: X projects to Q, K and V; Q times K transpose gives scores; scores are scaled, masked and softmaxed into weights; weights times V gives the output" width="100%">
#
# ## 21.2 · Deriving attention
#
# ### The question it answers
#
# A token at position $$i$$ needs information held at other positions. "The
# animal didn't cross the street because **it** was too tired" — to represent
# *it*, the model must look back at *animal*.
#
# So we need, for each position $$i$$, a set of weights over all positions
# $$j$$ saying **how much position $$i$$ should take from position $$j$$**.
# Those weights must depend on the *content* of both, and must be a
# probability distribution so the result is an average rather than an
# arbitrary sum.
#
# That is the whole specification. Attention is what you get by writing it
# down.
#
# ### Three projections, not one
#
# The question "how relevant is $$j$$ to $$i$$?" is asymmetric: what a token
# *advertises* about itself is not the same as what it *looks for* in others,
# and neither is the same as what it *contributes* once chosen.
#
# | Projection | Role | Analogy |
# |---|---|---|
# | $$Q = XW_Q$$ | what this position is looking for | the search box |
# | $$K = XW_K$$ | what this position advertises | the index entry |
# | $$V = XW_V$$ | what it hands over if chosen | the page itself |
#
# With one projection you could only ask "how similar are these two tokens?",
# which is a much weaker question. Three lets the model learn that *it* should
# look for nouns, that *animal* should advertise as a noun, and that what
# *animal* passes along is something else again.

# %%
def project(X, W_q, W_k, W_v):
    """The three views of the same input."""
    return X @ W_q, X @ W_k, X @ W_v


n, d_model, d_k = 5, 8, 8
X = rng.normal(size=(n, d_model))
W_q, W_k, W_v = (rng.normal(size=(d_model, d_k)) * 0.3 for _ in range(3))
Q, K, V = project(X, W_q, W_k, W_v)
print(f"X {X.shape}  ->  Q {Q.shape}  K {K.shape}  V {V.shape}")

# %% [markdown]
# ### Scores: the dot product as similarity
#
# $$S = QK^\top, \qquad S_{ij} = q_i \cdot k_j$$
#
# The dot product $$q_i \cdot k_j = \|q_i\|\,\|k_j\|\cos\theta$$ is large when
# the two vectors point the same way. "Does what $$i$$ is looking for match
# what $$j$$ advertises?" — one matrix multiply answers it for every pair at
# once, which is the reason transformers are fast on a GPU and RNNs are not.

# %%
S = Q @ K.T
print("S (scores), shape", S.shape)
print(S)

# %% [markdown]
# ### The $$\sqrt{d_k}$$ that everyone quotes and nobody checks
#
# $$S = \frac{QK^\top}{\sqrt{d_k}}$$
#
# The usual justification: if the entries of $$q$$ and $$k$$ are independent
# with mean 0 and variance 1, then
#
# $$q \cdot k = \sum_{t=1}^{d_k} q_t k_t$$
#
# is a sum of $$d_k$$ independent terms each of variance 1, so
# $$\operatorname{Var}(q \cdot k) = d_k$$ and the standard deviation is
# $$\sqrt{d_k}$$.
#
# That is a claim about numbers. Let's check it.

# %%
print(f"{'d_k':>6}{'measured std(q·k)':>20}{'sqrt(d_k)':>12}")
for d in (4, 16, 64, 256, 1024):
    q = rng.normal(size=(20_000, d))
    k = rng.normal(size=(20_000, d))
    print(f"{d:>6}{(q * k).sum(1).std():>20.2f}{np.sqrt(d):>12.2f}")

# %% [markdown]
# The prediction holds. Now **why it matters**: softmax of large numbers is
# nearly one-hot, and a nearly one-hot attention row means the position reads
# from exactly one other position and ignores everything else.

# %%
def softmax(x, axis=-1):
    """Numerically stable softmax. Subtracting the max changes nothing
    mathematically (exp(a-m)/sum exp(a-m) = exp(a)/sum exp(a)) and everything
    computationally: exp(800) overflows, exp(0) does not."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


d = 64
q1 = rng.normal(size=(1, d))
K8 = rng.normal(size=(8, d))

print(f"{'':>10}{'max weight':>12}{'entropy (nats)':>17}  attention row")
for label, scores in (("unscaled", q1 @ K8.T), ("÷ sqrt(d_k)", q1 @ K8.T / np.sqrt(d))):
    row = softmax(scores)[0]
    entropy = -(row * np.log(row + 1e-12)).sum()
    print(f"{label:>10}{row.max():>12.4f}{entropy:>17.3f}  {np.round(row, 3)}")
print(f"\nfor reference, a uniform row over 8 positions has entropy "
      f"{np.log(8):.3f} nats")

# %% [markdown]
# Unscaled, one position takes **96%** of the weight and the row carries
# 0.18 nats — it has collapsed to very nearly a hard lookup. Scaled, the
# weight is spread and the row carries 1.86 nats out of a possible 2.08.
#
# The effect worsens as models get wider: $$d_k$$ is only 64 here, and the gap
# is already stark.
#
# Why that is fatal rather than merely untidy: the gradient of softmax is
# proportional to $$p_i(1 - p_i)$$, which is $$\approx 0$$ when $$p \approx 1$$.
# A saturated attention row **stops learning**. The scaling is not cosmetic;
# without it, deep models with wide heads do not train.
#
# > **Exercise 1.** The argument assumes $$q$$ and $$k$$ have unit variance.
# > Re-run the measurement with `rng.normal(scale=0.1, ...)` and with
# > `scale=3`. Does $$\sqrt{d_k}$$ still produce a well-spread row? Then say
# > what that implies about why the projections are initialised carefully, and
# > why **RMSNorm sits immediately before** the projections in a real block
# > (21.6).

# %% [markdown]
# ### Softmax, then the weighted sum
#
# $$A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right),
# \qquad \operatorname{Attention}(Q,K,V) = AV$$
#
# Row $$i$$ of $$A$$ is a probability distribution over positions; row $$i$$
# of $$AV$$ is the corresponding weighted average of the value vectors. That
# is the entire operation.

# %%
def attention(Q, K, V, mask=None):
    """Scaled dot-product attention. Returns (output, attention weights)."""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        # -inf before softmax => exactly 0 weight after it.
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores)
    return weights @ V, weights


out, A = attention(Q, K, V)
print("attention weights A:")
print(A)
print("\nevery row sums to 1:", np.allclose(A.sum(axis=1), 1.0))
print("output shape:", out.shape)

# %% [markdown]
# ## 21.3 · The causal mask, and proving it works
#
# A language model predicts token $$i+1$$ from tokens $$\le i$$. If position
# $$i$$ could see position $$i+1$$, the training objective would be trivial
# and the model would learn nothing usable (notebook 13 called this out as the
# bug that makes loss collapse impossibly fast).
#
# So we forbid it:
#
# $$S_{ij} \leftarrow -\infty \quad \text{for } j > i$$
#
# and since $$e^{-\infty} = 0$$, those positions get exactly zero weight.
#
# **The claim is that the output at position $$i$$ cannot depend on any token
# after $$i$$.** That is testable, so let's test it rather than believe it.

# %%
causal = np.tril(np.ones((n, n), dtype=bool))
print("mask (True = may attend):")
print(causal.astype(int))

_, A_causal = attention(Q, K, V, mask=causal)
print("\nattention weights - note the exact zeros above the diagonal:")
print(A_causal)

# %%
# The real test: change a LATER token and check earlier outputs are untouched.
CHANGED = 3
X_perturbed = X.copy()
X_perturbed[CHANGED] = rng.normal(size=d_model)

out_before, _ = attention(*project(X, W_q, W_k, W_v), mask=causal)
out_after, _ = attention(*project(X_perturbed, W_q, W_k, W_v), mask=causal)

delta = np.abs(out_before - out_after).max(axis=1)
print(f"changed the token at position {CHANGED}\n")
print(f"{'position':>9}{'max |change in output|':>26}")
for i, value in enumerate(delta):
    marker = "  <- at or after the change" if i >= CHANGED else ""
    print(f"{i:>9}{value:>26.2e}{marker}")

assert delta[:CHANGED].max() == 0.0, "causal mask is leaking!"
print("\npositions before the change are bit-identical. Exactly 0.0, not 'small'.")

# %% [markdown]
# **Exactly zero, not approximately zero.** That distinction matters: a mask
# implemented with a large negative number instead of $$-\infty$$ leaks a tiny
# amount of future information, which is invisible in the loss curve and
# devastating at evaluation time, because the model learned to use a signal
# that does not exist at inference.
#
# > **Exercise 2.** Replace `-np.inf` with `-1e9` and re-run the test. Is the
# > leak exactly zero? Now try `-1e9` with `float16` instead of `float64`
# > (`scores.astype(np.float16)`) and explain what you see. This is a real bug
# > that has shipped in real codebases.

# %% [markdown]
# ## 21.4 · Attention cannot see order
#
# Here is the property almost nobody is shown, and it is the entire reason
# positional encodings exist.
#
# Attention is **permutation equivariant**: shuffle the input tokens and the
# outputs come out shuffled the same way, with no other change. The operation
# has no notion of "before" or "after" — it is a set operation, not a sequence
# operation.
#
# Formally, for a permutation matrix $$P$$:
#
# $$\operatorname{Attention}(PX) = P \operatorname{Attention}(X)$$
#
# Let's demonstrate it (no mask — the mask is itself positional information).

# %%
perm = rng.permutation(n)
print("permutation:", perm)

out_plain, _ = attention(*project(X, W_q, W_k, W_v))
out_shuffled, _ = attention(*project(X[perm], W_q, W_k, W_v))

print("\nmax |attention(PX) - P·attention(X)| =",
      np.abs(out_shuffled - out_plain[perm]).max())
print("\nThat is floating-point zero. The model literally cannot tell")
print("'dog bites man' from 'man bites dog' on attention alone.")

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT407LLM/master/docs/assets/diagrams/mt-rope.svg" alt="RoPE rotates each pair of dimensions by an angle proportional to position, so the inner product between a rotated query and key depends only on their gap" width="100%">
#
# ### RoPE: rotating position into the vectors
#
# The fix used by Qwen (and Llama, and most current models) is **Rotary
# Position Embedding**. Instead of *adding* a position vector, it **rotates**
# $$q$$ and $$k$$ by an angle proportional to their position.
#
# Treat the $$d_k$$ dimensions as $$d_k/2$$ pairs. Rotate pair $$t$$ at
# position $$m$$ by angle $$m\theta_t$$, where
#
# $$\theta_t = \text{base}^{-2t/d_k}, \qquad \text{base} = 10000$$
#
# so early pairs spin fast (local detail) and later pairs spin slowly (long
# range). Each pair is rotated by the standard 2-D rotation matrix:
#
# $$\begin{pmatrix} x' \\\\ y' \end{pmatrix} =
# \begin{pmatrix} \cos m\theta & -\sin m\theta \\\\
# \sin m\theta & \cos m\theta \end{pmatrix}
# \begin{pmatrix} x \\\\ y \end{pmatrix}$$

# %%
def rope(x, positions, base=10000.0):
    """Rotary position embedding. x is (n, d_k) with d_k even."""
    d = x.shape[-1]
    assert d % 2 == 0, "RoPE needs an even head dimension"
    pair_index = np.arange(0, d, 2)
    theta = positions[:, None] / (base ** (pair_index / d))   # (n, d/2)
    cos, sin = np.cos(theta), np.sin(theta)

    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = np.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


# A rotation preserves length - it turns the vector, it does not stretch it.
probe = rng.normal(size=(1, 16))
print("‖x‖ before rotation:", np.linalg.norm(probe))
print("‖x‖ after  rotation:", np.linalg.norm(rope(probe, np.array([7]))))

# %% [markdown]
# ### Why rotation is the right trick
#
# Rotating both vectors by their positions makes their inner product depend
# **only on the gap between them**:
#
# $$\langle R_m q,\; R_n k \rangle = g(q, k, m - n)$$
#
# The model never sees an absolute position in the score — only a relative
# one. That is exactly what language needs, and it is why RoPE extrapolates
# to longer sequences better than a learned absolute embedding.
#
# This is a strong claim. Measure it.

# %%
q_probe = rng.normal(size=(1, 16))
k_probe = rng.normal(size=(1, 16))

print("same gap, wildly different absolute positions:")
print(f"{'m':>6}{'n':>6}{'m-n':>6}{'<R_m q, R_n k>':>20}")
for m, n_pos in [(3, 0), (5, 2), (10, 7), (100, 97), (5000, 4997)]:
    value = (rope(q_probe, np.array([m])) @ rope(k_probe, np.array([n_pos])).T).item()
    print(f"{m:>6}{n_pos:>6}{m - n_pos:>6}{value:>20.9f}")

print("\ndifferent gaps, same starting position:")
for m, n_pos in [(5, 0), (5, 1), (5, 3), (5, 4)]:
    value = (rope(q_probe, np.array([m])) @ rope(k_probe, np.array([n_pos])).T).item()
    print(f"{m:>6}{n_pos:>6}{m - n_pos:>6}{value:>20.9f}")

# %% [markdown]
# Identical to nine decimal places for a gap of 3, whether the tokens are at
# positions 3 and 0 or at 5000 and 4997. Different gaps give different values.
#
# **That is the whole idea of RoPE in one table.** No absolute position
# survives into the score; only the distance does.
#
# > **Exercise 3.** The `base` is 10000 by convention. Print $$\theta_t$$ for
# > the first and last pair at $$d_k = 128$$, and work out the wavelength
# > $$2\pi/\theta_t$$ of each in tokens. Which pair is responsible for
# > long-range position? Then look up what "RoPE scaling" does for long-context
# > models and say, in one sentence, which end of that spectrum it stretches.

# %% [markdown]
# ## 21.5 · Multi-head attention
#
# One attention operation produces one weighted average per position. But a
# token often needs several *different* things at once — the subject of the
# verb, the noun a pronoun refers to, the opening bracket that matches.
#
# So run $$h$$ of them in parallel on slices of the vector, and concatenate:
#
# $$\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,
# \qquad \text{head}_i = \operatorname{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$
#
# Note that $$d_k = d_{\text{model}}/h$$, so **the total compute is unchanged**.
# Eight heads of width 64 cost the same as one head of width 512. You are not
# paying for the heads; you are *spending the same budget differently*.

# %%
def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None, positions=None):
    """Multi-head attention, with optional causal mask and RoPE."""
    n_tokens, d_m = X.shape
    assert d_m % n_heads == 0, "d_model must divide by the head count"
    head_dim = d_m // n_heads

    Q_all, K_all, V_all = project(X, W_q, W_k, W_v)
    outputs = []
    for head in range(n_heads):
        sl = slice(head * head_dim, (head + 1) * head_dim)
        q, k, v = Q_all[:, sl], K_all[:, sl], V_all[:, sl]
        if positions is not None:
            # RoPE is applied to q and k only - never to v. The values carry
            # content, not position, and rotating them would corrupt it.
            q, k = rope(q, positions), rope(k, positions)
        head_out, _ = attention(q, k, v, mask=mask)
        outputs.append(head_out)
    return np.concatenate(outputs, axis=-1) @ W_o


W_o = rng.normal(size=(d_model, d_model)) * 0.3
for heads in (1, 2, 4):
    result = multi_head_attention(X, W_q, W_k, W_v, W_o, heads)
    print(f"h={heads}: output {result.shape}, head width {d_model // heads}")

# %% [markdown]
# ### A sanity check worth doing
#
# With $$h = 1$$, multi-head attention must reduce *exactly* to single-head
# attention followed by $$W_O$$. If your implementation's reshaping is wrong,
# this is where it shows.

# %%
mha_one = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads=1)
single, _ = attention(Q, K, V)
single = single @ W_o

print("max |MHA(h=1) - single-head| =", np.abs(mha_one - single).max())
print("max |MHA(h=2) - MHA(h=1)|    =",
      np.abs(multi_head_attention(X, W_q, W_k, W_v, W_o, 2) - mha_one).max(),
      " (should NOT be zero)")

# %% [markdown]
# > **Exercise 4.** Qwen3 uses **grouped-query attention**: 16 query heads but
# > only 8 key/value heads, with each KV head shared by two query heads.
# > Modify `multi_head_attention` to take `n_kv_heads` and share them. Then
# > compute what that does to the KV-cache size in 21.8 — which is the entire
# > reason GQA exists.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT407LLM/master/docs/assets/diagrams/mt-block-anatomy.svg" alt="A pre-norm block: the residual stream passes straight through while RMSNorm, attention and the SwiGLU MLP compute corrections that are added back" width="100%">
#
# ## 21.6 · The residual stream and RMSNorm
#
# A block does not *replace* its input, it **adds to it**:
#
# $$X \leftarrow X + \operatorname{Attention}(\operatorname{Norm}(X))$$
# $$X \leftarrow X + \operatorname{MLP}(\operatorname{Norm}(X))$$
#
# The running $$X$$ is the **residual stream**. Each block reads it, computes
# a correction, and writes the correction back. Two consequences:
#
# 1. **A block can do nothing.** If its output is near zero, the input passes
#    through untouched. A 36-layer model does not have to use all 36 layers for
#    every token.
# 2. **Gradients have a short path.** The `+` passes gradient straight
#    through, which is what makes very deep stacks trainable at all.
#
# **Pre-norm, not post-norm.** Qwen normalises *before* each sub-layer, so the
# residual path itself is never normalised. That keeps the straight-through
# gradient path clean, and it is why modern models train without the warmup
# gymnastics the original 2017 post-norm transformer needed.
#
# ### RMSNorm
#
# Qwen uses RMSNorm, not LayerNorm. LayerNorm subtracts the mean and divides
# by the standard deviation; RMSNorm skips the mean:
#
# $$\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_t x_t^2 + \epsilon}} \odot g$$
#
# Cheaper, one fewer reduction, and empirically just as good — which is the
# whole justification. Not every detail in a transformer has a deep reason.

# %%
def rms_norm(x, gain=None, eps=1e-6):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    normed = x / rms
    return normed if gain is None else normed * gain


scaled_input = X * 50.0        # pretend an earlier block blew up the scale
normed = rms_norm(scaled_input)
print("input  RMS per row:", np.sqrt((scaled_input ** 2).mean(-1)).round(2))
print("output RMS per row:", np.sqrt((normed ** 2).mean(-1)).round(6))
print("\nRMSNorm fixes the scale, so the sqrt(d_k) argument in 21.2 keeps holding")
print("however large the activations upstream have become.")

# %% [markdown]
# ## 21.7 · The MLP, where most of the parameters live
#
# Attention moves information *between* positions. The MLP transforms it
# *within* a position — the same small network applied to every token
# independently.
#
# Qwen uses **SwiGLU**, a gated variant:
#
# $$\operatorname{SwiGLU}(x) = \big(\operatorname{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\big)W_{\text{down}}$$
#
# where $$\operatorname{SiLU}(z) = z\,\sigma(z)$$. The `gate` branch decides
# *how much* of the `up` branch to let through, per dimension — a learned,
# content-dependent valve. It costs a third projection and is worth it.

# %%
def silu(z):
    return z / (1.0 + np.exp(-z))


def swiglu_mlp(x, W_gate, W_up, W_down):
    return (silu(x @ W_gate) * (x @ W_up)) @ W_down


d_ff = 4 * d_model
W_gate, W_up = (rng.normal(size=(d_model, d_ff)) * 0.1 for _ in range(2))
W_down = rng.normal(size=(d_ff, d_model)) * 0.1
print("MLP output shape:", swiglu_mlp(X, W_gate, W_up, W_down).shape)
print(f"MLP parameters: 3 x {d_model} x {d_ff} = {3 * d_model * d_ff:,}")
print(f"attention parameters: 4 x {d_model}^2 = {4 * d_model ** 2:,}")
print("\nThe MLP is the bigger half of a transformer, which surprises people")
print("who think of attention as 'the' mechanism.")

# %% [markdown]
# ## 21.8 · A whole block, and what it costs
#
# Everything assembled, in the order Qwen actually uses it.

# %%
def transformer_block(X, params, n_heads, mask, positions):
    """One pre-norm transformer block: attention, then MLP, both residual."""
    X = X + multi_head_attention(
        rms_norm(X), params["W_q"], params["W_k"], params["W_v"], params["W_o"],
        n_heads, mask=mask, positions=positions,
    )
    X = X + swiglu_mlp(
        rms_norm(X), params["W_gate"], params["W_up"], params["W_down"],
    )
    return X


params = {"W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
          "W_gate": W_gate, "W_up": W_up, "W_down": W_down}
positions = np.arange(n)

Y = transformer_block(X, params, n_heads=2, mask=causal, positions=positions)
print("block:", X.shape, "->", Y.shape)
print("\nresidual stream is a correction, not a replacement:")
print("  mean |X|      =", np.abs(X).mean().round(4))
print("  mean |Y - X|  =", np.abs(Y - X).mean().round(4), " <- the block's contribution")

# %%
# The causal property must survive the whole block, not just the attention.
X_p = X.copy()
X_p[CHANGED] = rng.normal(size=d_model)
Y_p = transformer_block(X_p, params, 2, causal, positions)
leak = np.abs(Y[:CHANGED] - Y_p[:CHANGED]).max()
print(f"leak into positions before {CHANGED}, through the FULL block: {leak}")
assert leak == 0.0
print("Still exactly zero. The MLP is per-position, so it cannot leak either.")

# %% [markdown]
# ### Counting what it costs
#
# Three numbers decide whether a model fits and how fast it runs.

# %%
def block_costs(d_model_, n_heads_, d_ff_, n_tokens, n_layers, bytes_per_param=2):
    attn_params = 4 * d_model_ ** 2
    mlp_params = 3 * d_model_ * d_ff_
    per_block = attn_params + mlp_params

    # FLOPs for one block's forward pass, counting a multiply-add as 2.
    projections = 2 * n_tokens * (4 * d_model_ ** 2 + 3 * d_model_ * d_ff_)
    attn_matmuls = 2 * 2 * n_heads_ * n_tokens ** 2 * (d_model_ // n_heads_)

    kv_cache = 2 * n_layers * n_tokens * d_model_ * bytes_per_param
    return {
        "params per block": per_block,
        "params total": per_block * n_layers,
        "FLOPs (projections)": projections,
        "FLOPs (attention matmuls)": attn_matmuls,
        "KV cache bytes": kv_cache,
    }


print("Qwen3-0.6B-ish: d_model=1024, 16 heads, d_ff=3072, 28 layers\n")
for n_tok in (512, 4096, 32768):
    costs = block_costs(1024, 16, 3072, n_tok, 28)
    quad = costs["FLOPs (attention matmuls)"]
    lin = costs["FLOPs (projections)"]
    print(f"  {n_tok:>6} tokens:  KV cache {costs['KV cache bytes'] / 1e9:5.2f} GB   "
          f"attention/projection FLOP ratio {quad / lin:6.2f}")

print(f"\n  parameters: {block_costs(1024, 16, 3072, 1, 28)['params total'] / 1e6:.0f}M "
      "in the blocks (plus embeddings)")

# %% [markdown]
# Read the ratio column. At 512 tokens the quadratic term is a rounding error
# and the model is dominated by the projections — the ones that scale
# *linearly* in sequence length. At 32k it dominates completely.
#
# **This is why "quadratic attention" is both true and misleading.** It is
# quadratic, and at ordinary sequence lengths it is not what you are paying
# for — at 512 tokens the attention matmuls are 8% of the projection cost.
# The KV cache, meanwhile, grows linearly and relentlessly: 3.8 GB at 32k
# tokens, roughly three times the weights of a 0.6B model at bf16. Notebook 02
# measured exactly this.
#
# > **Exercise 5.** At what sequence length do the two FLOP terms cross for
# > this configuration? Solve it algebraically, then confirm with the function.
# > Now redo it for $$d_{\text{model}} = 4096$$ and explain why bigger models
# > tolerate longer contexts before attention dominates.
#
# One caveat on that KV figure: the formula caches a full
# $$d_{\text{model}}$$ of keys and values per layer. Qwen3 uses grouped-query
# attention with half as many KV heads as query heads, so its real cache is
# **half** what is printed — which is the point of exercise 4.

# %% [markdown]
# ## 21.9 · Checking against PyTorch
#
# Our `attention` should agree with `F.scaled_dot_product_attention` to
# floating-point precision. If it does not, one of us is wrong, and it is
# worth knowing which.

# %%
try:
    import torch
    import torch.nn.functional as F

    tq, tk, tv = (torch.tensor(a, dtype=torch.float64) for a in (Q, K, V))
    reference = F.scaled_dot_product_attention(tq, tk, tv, is_causal=True).numpy()
    ours, _ = attention(Q, K, V, mask=causal)

    difference = np.abs(ours - reference).max()
    print(f"max |ours - PyTorch| = {difference:.2e}")
    print("agree to double precision" if difference < 1e-12 else "MISMATCH - investigate")
except ImportError:
    print("torch is not installed, so this check did not run.\n"
          "Install it and re-run this cell: the NumPy code above is the whole\n"
          "of scaled_dot_product_attention, and you should confirm that rather\n"
          "than take my word for it.")

# %% [markdown]
# > **Honesty note.** The authors have **not** executed the cell above against
# > a real PyTorch install, so that one comparison is unverified. Everything
# > else in this notebook is checked by the assertions you just watched run —
# > including the two that assert exact zeros.
#
# ## What you now know that you did not
#
# | Claim | How you know |
# |---|---|
# | $$\operatorname{std}(q \cdot k) = \sqrt{d_k}$$ | measured at five widths |
# | Unscaled softmax saturates | 0.96 max weight, 0.18 of 2.08 nats |
# | The causal mask is exactly causal | perturbation test, exactly 0.0 |
# | Attention alone is blind to order | permutation test, 1e-16 |
# | RoPE encodes relative position | identical inner product at gap 3 from m=3 to m=5000 |
# | Multi-head with h=1 is single-head | difference exactly 0.0 |
# | The MLP holds more parameters than attention | 3·d·d_ff vs 4·d² |
# | Attention is quadratic but not always dominant | FLOP ratio across sequence lengths |
#
# ## Checkpoint
#
# 1. Why three projections rather than one? What question does each answer?
# 2. Where does $$\sqrt{d_k}$$ come from, and what breaks without it?
# 3. Why must the mask use $$-\infty$$ and not a large negative number?
# 4. State the permutation-equivariance property and why it forces positional
#    encoding to exist.
# 5. What does RoPE make the attention score depend on, and why is that the
#    right thing for language?
# 6. Multi-head attention with $$h$$ heads costs the same as one head. Why?
# 7. What is the residual stream, and give two things it buys you.
# 8. Which holds more parameters, attention or the MLP? Which dominates the
#    FLOPs at 512 tokens?
#
# ➡️ **Next:** the reading companion,
# [`docs/handbook/the-mathematics.md`](../docs/handbook/the-mathematics.md),
# sets the same material out as derivations with diagrams. Or go back to
# [`13_pretrain_a_tiny_model.ipynb`](13_pretrain_a_tiny_model.ipynb) and read
# its `SelfAttention` class again — it should now contain no surprises.
