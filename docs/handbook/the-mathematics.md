---
title: "7 · The mathematics"
parent: Handbook
nav_order: 7
---

# The mathematics of a transformer
{: .no_toc }

Chapter 2 said attention is a weighted average and moved on. This chapter
does the derivation: where every term comes from, why it has the shape it
has, and what breaks if you change it.

1. TOC
{:toc}

---

{: .note }
> **The companion notebook is where this becomes real.**
> [`21_attention_from_scratch`](../notebooks/21_attention_from_scratch.html)
> implements everything below in NumPy and *proves* the claims by measurement —
> that the causal mask is exactly causal, that attention alone cannot see word
> order, that $$\sqrt{d_k}$$ is not an arbitrary constant. Read here, run
> there.

**What you need:** matrix multiplication, and the geometric meaning of a dot
product. No calculus — nothing here is differentiated.

## Notation

| Symbol | Shape | Meaning |
|---|---|---|
| $$n$$ | — | tokens in the sequence |
| $$d_{\text{model}}$$ | — | width of the residual stream |
| $$h$$ | — | attention heads |
| $$d_k$$ | — | width of one head, $$d_{\text{model}}/h$$ |
| $$X$$ | $$n \times d_{\text{model}}$$ | token representations entering a block |
| $$W_Q, W_K, W_V$$ | $$d_{\text{model}} \times d_k$$ | learned projections |
| $$Q, K, V$$ | $$n \times d_k$$ | queries, keys, values |
| $$A$$ | $$n \times n$$ | attention weights, rows summing to 1 |

One sentence to carry through the chapter: **a transformer moves information
between positions with attention, and within a position with the MLP.**

---

## 1. Deriving attention

### The specification

A token needs information held elsewhere in the sentence. In *"the animal
didn't cross the street because it was too tired"*, representing **it**
requires looking back at **animal**.

So we need, for each position $$i$$, weights over all positions $$j$$ saying
how much $$i$$ should take from $$j$$. Those weights must

1. depend on the **content** of both positions, not just their indices, and
2. form a **probability distribution**, so the result is an average rather
   than an arbitrary sum that could blow up.

Attention is what you get by writing that specification down. There is no
further magic in it.

### Why three projections

"How relevant is $$j$$ to $$i$$?" is an asymmetric question. What a token
*advertises* about itself differs from what it *seeks* in others, and both
differ from what it *contributes* once selected.

| | Definition | Role |
|---|---|---|
| Query | $$Q = XW_Q$$ | what this position is looking for |
| Key | $$K = XW_K$$ | what this position advertises |
| Value | $$V = XW_V$$ | what it hands over if selected |

With a single projection you could only ask "how similar are these two
tokens?", which is strictly weaker. Three lets the model learn that *it*
should seek nouns, that *animal* advertises as a noun, and that what *animal*
passes along is a third thing entirely.

<img src="../assets/diagrams/mt-attention-pipeline.svg" alt="The attention pipeline: X projects to Q, K and V; Q times K transpose gives scores; scores are scaled, masked and softmaxed into weights; weights times V gives the output" width="100%">

### Scores

$$S = QK^\top, \qquad S_{ij} = q_i \cdot k_j$$

Since $$q_i \cdot k_j = \lVert q_i \rVert \lVert k_j \rVert \cos\theta$$, the
score is large when the two point the same way. One matrix multiply answers
the relevance question for every pair simultaneously — which is precisely why
transformers parallelise on a GPU and recurrent networks do not.

### The $$\sqrt{d_k}$$

$$A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

Suppose the entries of $$q$$ and $$k$$ are independent with mean 0 and
variance 1. Then

$$q \cdot k = \sum_{t=1}^{d_k} q_t k_t$$

is a sum of $$d_k$$ independent terms, each with variance 1, so

$$\operatorname{Var}(q \cdot k) = d_k, \qquad \operatorname{sd}(q \cdot k) = \sqrt{d_k}$$

Dividing by $$\sqrt{d_k}$$ returns the scores to unit scale **regardless of
head width**.

**Why it matters.** Softmax of large values is nearly one-hot. The notebook
measures this: at $$d_k = 64$$, unscaled scores give one position 96% of the
weight and an entropy of 0.18 nats, against 1.86 nats scaled.

The consequence is not untidiness, it is that **training stops**. The
derivative of softmax is proportional to $$p_i(1 - p_i)$$, which vanishes as
$$p \to 1$$. A saturated attention row produces no gradient. Without the
scaling, wide-headed deep models do not train at all.

### The operation, complete

$$\operatorname{Attention}(Q, K, V) =
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Row $$i$$ of the softmax is a distribution over positions; row $$i$$ of the
product is the corresponding weighted average of value vectors. That is all
of it.

---

## 2. The causal mask

A language model predicts token $$i+1$$ from tokens $$\le i$$. If position
$$i$$ could attend to $$i+1$$, the objective would be trivial and the model
would learn nothing that survives to inference.

$$S_{ij} \leftarrow -\infty \quad \text{for } j > i$$

and because $$e^{-\infty} = 0$$, those positions receive **exactly** zero
weight.

{: .warning }
> **Use $$-\infty$$, not a large negative number.** A mask built from `-1e9`
> leaks a minuscule amount of future information. It is invisible in the loss
> curve — training looks slightly *better* — and it corrupts evaluation,
> because the model has learned to use a signal that does not exist at
> inference time. In `float16`, `-1e9` is not even representable. This bug has
> shipped in real codebases.

The notebook verifies the property the mask is supposed to have: change a
token at position 3, and the outputs at positions 0–2 are **bit-identical** —
a difference of exactly `0.0`, not merely small.

---

## 3. Why positional encoding has to exist

This is the part most treatments assert and few demonstrate.

**Attention is permutation equivariant.** For any permutation matrix $$P$$:

$$\operatorname{Attention}(PX) = P \operatorname{Attention}(X)$$

Shuffle the input tokens and the outputs come out shuffled identically, with
no other change. The operation is defined over a *set*, not a sequence. On
attention alone, a model cannot distinguish *dog bites man* from *man bites
dog*.

So positional information must be injected deliberately. The notebook
measures the equivariance at $$10^{-16}$$ — floating-point zero — which is
the proof that it is a structural property and not an approximation.

### RoPE

Qwen, Llama and most current models use **Rotary Position Embedding**. Rather
than *adding* a position vector, it **rotates** $$q$$ and $$k$$ by an angle
proportional to position.

Treat the $$d_k$$ dimensions as $$d_k/2$$ pairs. Rotate pair $$t$$ at position
$$m$$ by $$m\theta_t$$, where

$$\theta_t = \text{base}^{-2t/d_k}, \qquad \text{base} = 10000$$

Early pairs spin quickly and carry local detail; later pairs spin slowly and
carry long-range position. Each pair gets the ordinary 2-D rotation:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} =
\begin{pmatrix} \cos m\theta_t & -\sin m\theta_t \\
\sin m\theta_t & \cos m\theta_t \end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix}$$

<img src="../assets/diagrams/mt-rope.svg" alt="RoPE rotates each pair of dimensions by an angle proportional to position; the inner product between a rotated query and a rotated key depends only on the difference of their positions" width="100%">

**The property that makes it work.** Because a rotation by $$m$$ followed by
the inverse rotation by $$n$$ is a rotation by $$m-n$$:

$$\langle R_m q,\; R_n k \rangle = g(q, k, m - n)$$

The score depends **only on the gap**, never on absolute position. The
notebook measures an inner product of `2.155172922` for a gap of 3 at
positions $$(3,0)$$, $$(5,2)$$, $$(10,7)$$, $$(100,97)$$ *and*
$$(5000, 4997)$$ — identical to nine decimal places.

Two consequences worth knowing:

* **Rotation preserves length**, so RoPE cannot change the magnitude of a
  query or key, only its direction. Adding a position vector can.
* It is applied to $$q$$ and $$k$$ **only, never to $$v$$**. Values carry
  content; rotating them would corrupt what is being passed along.

---

## 4. Multi-head attention

One attention operation yields one weighted average per position. A token
usually needs several different things at once — the subject of its verb, the
antecedent of a pronoun, the matching bracket.

$$\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\,W_O$$

$$\text{head}_i = \operatorname{Attention}(XW_Q^i,\; XW_K^i,\; XW_V^i)$$

Since $$d_k = d_{\text{model}}/h$$, **the total compute is unchanged**. Eight
heads of width 64 cost what one head of width 512 costs. You are not buying
heads; you are spending a fixed budget differently — several narrow views
instead of one wide one.

{: .tip }
> A reliable self-test: multi-head attention with $$h = 1$$ must reduce
> *exactly* to single-head attention followed by $$W_O$$. If the reshaping is
> wrong, this is where it shows. The notebook asserts a difference of `0.0`.

### Grouped-query attention

Qwen3 uses fewer key/value heads than query heads — 8 KV heads for 16 query
heads, each KV head shared by two queries. Quality is barely affected and the
**KV cache halves**, which at long context is the binding constraint. See §6.

---

## 5. The rest of a block

<img src="../assets/diagrams/mt-block-anatomy.svg" alt="A pre-norm transformer block: the residual stream passes straight through while RMSNorm, attention and the SwiGLU MLP compute corrections that are added back" width="100%">

### The residual stream

A block does not replace its input. It **adds a correction**:

$$X \leftarrow X + \operatorname{Attention}(\operatorname{RMSNorm}(X))$$

$$X \leftarrow X + \operatorname{MLP}(\operatorname{RMSNorm}(X))$$

The running $$X$$ is the **residual stream**: a shared channel every block
reads from and writes back to. Two consequences:

1. **A block can decline to act.** Output near zero means the input passes
   through untouched, so a 36-layer model need not use 36 layers per token.
2. **Gradients have a short path.** Addition passes gradient straight through,
   which is what makes very deep stacks trainable.

**Pre-norm, not post-norm.** Normalising *before* each sub-layer leaves the
residual path itself un-normalised, keeping that straight-through gradient
path clean. The original 2017 transformer used post-norm and needed careful
learning-rate warmup to train at all; pre-norm is why modern models do not.

### RMSNorm

LayerNorm subtracts the mean and divides by the standard deviation. RMSNorm
skips the mean:

$$\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_t x_t^2 + \epsilon}} \odot g$$

One fewer reduction, cheaper, and empirically no worse. Not every detail in a
transformer has a deep justification; some are simply what survived.

It also matters for §1: normalising immediately before the projections keeps
activations at unit scale, which is the assumption the $$\sqrt{d_k}$$
derivation rests on.

### SwiGLU

Attention moves information *between* positions. The MLP transforms it
*within* one — the same small network applied to every token independently.
Qwen uses a gated variant:

$$\operatorname{SwiGLU}(x) = \big(\operatorname{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\big)W_{\text{down}}$$

with $$\operatorname{SiLU}(z) = z\,\sigma(z)$$. The gate branch decides, per
dimension and per token, how much of the up branch to let through — a learned,
content-dependent valve. It costs a third projection and earns it.

---

## 6. What it costs

Three numbers decide whether a model fits and how fast it runs.

### Parameters

| | Count | At $$d_{\text{model}}=1024$$, $$d_{ff}=3072$$ |
|---|---|---|
| Attention | $$4d_{\text{model}}^2$$ | 4.2 M |
| MLP | $$3 d_{\text{model}} d_{ff}$$ | 9.4 M |

**The MLP is the larger half**, which surprises people who think of attention
as *the* mechanism. Attention is the interesting part; the MLP is where the
parameters are.

### FLOPs

Per block, per forward pass:

$$\underbrace{2n\left(4d_{\text{model}}^2 + 3d_{\text{model}}d_{ff}\right)}_{\text{projections, linear in } n}
+ \underbrace{4 n^2 d_{\text{model}}}_{\text{attention matmuls, quadratic in } n}$$

The notebook evaluates the ratio of the two terms:

| Sequence length | attention ÷ projection FLOPs |
|---|---|
| 512 | 0.08 |
| 4,096 | 0.62 |
| 32,768 | 4.92 |

**"Attention is quadratic" is true and misleading.** It is quadratic, and at
ordinary sequence lengths it is not what you are paying for — at 512 tokens
the attention matmuls are 8% of the projection cost. It only dominates once
you are well past a few thousand tokens.

### The KV cache

Generating token $$n+1$$ needs the keys and values of all previous tokens.
Recomputing them each step would be quadratic in total work, so they are
cached:

$$\text{bytes} = 2 \times n_{\text{layers}} \times n \times d_{\text{model}} \times \text{bytes per element}$$

At 32k tokens on the configuration above that is **3.8 GB** — roughly three
times the weights of a 0.6B model at bf16. Halved again by grouped-query
attention. This is the real limit on context length in deployment, and
notebook 02 measures it for the whole Qwen3 family.

---

## What to take away

1. **Attention is a specification written down**, not an invention: content-
   dependent weights forming a distribution.
2. **Three projections** because relevance is asymmetric.
3. **$$\sqrt{d_k}$$ keeps softmax out of saturation**, where it would stop
   producing gradient.
4. **The mask must be $$-\infty$$.** A large negative number leaks.
5. **Attention cannot see order** — that is a provable property, and the
   reason RoPE exists.
6. **RoPE makes scores depend only on the gap** between positions.
7. **Heads are free**; they split a fixed budget rather than adding to it.
8. **The residual stream is the architecture.** Blocks are corrections to it.
9. **The MLP holds more parameters; attention dominates only at long context.**

---

**Next:** run it. [`21_attention_from_scratch`](../notebooks/21_attention_from_scratch.html)
builds every equation above in NumPy and tests the claims. Then re-read
[notebook 13's](../notebooks/13_pretrain_a_tiny_model.html) `SelfAttention`
class — it should contain no surprises left.
