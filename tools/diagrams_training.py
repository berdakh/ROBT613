#!/usr/bin/env python3
"""Diagrams for the training track (notebooks 13-16).

Imported by ``make_diagrams.py``, which registers these into the shared
``DIAGRAMS`` list and renders them. Each function returns
``(filename, svg, caption)``; the caption doubles as alt text.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import Canvas, diagram  # noqa: E402

# ---------------------------------------------------------------------------
# 13 - the training loop
# ---------------------------------------------------------------------------


@diagram
def training_loop():
    c = Canvas(900, 400)
    c.title("One training step",
            "Every trainer you will ever use is this loop with more bookkeeping.")

    # Clip and Schedule sit ABOVE the row so their connectors cross nothing.
    c.box(588, 86, 136, 50, "Clip", "cap the gradient norm", role="warn")
    c.box(770, 86, 102, 50, "Schedule", "sets the LR", role="warn")

    y = 158
    c.box(28, y, 136, 60, "Batch", "tokens in, targets shifted by one")
    c.arrow(168, y + 30, 206, y + 30, accent=True)
    c.box(210, y, 136, 60, "Forward", "predict every next token", role="accent")
    c.arrow(350, y + 30, 388, y + 30, accent=True)
    c.box(392, y, 150, 60, "Loss", "cross-entropy, in nats", role="accent")
    c.arrow(546, y + 30, 584, y + 30, accent=True)
    c.box(588, y, 136, 60, "Backward", "gradient per parameter", role="accent")
    c.arrow(728, y + 30, 766, y + 30, accent=True)
    c.box(770, y, 102, 60, "Step", "update weights", role="good")

    c.arrow(656, 136, 656, y - 4, dashed=True)
    c.arrow(821, 136, 821, y - 4, dashed=True)

    # The loop back to the next batch, routed clear of everything above it.
    c.path("M 821 218 L 821 256 L 96 256 L 96 218", accent=True)
    c.text(458, 248, "next batch, a few thousand to a few trillion times",
           "small")

    c.rule(28, 300, 872, 300)
    c.text(28, 324, "The only numbers that come out of this loop:",
           "lbl", anchor="start")
    c.text(28, 348,
           "train loss  ·  validation loss  ·  learning rate  ·  gradient norm",
           "mono", anchor="start")
    c.text(28, 374,
           "Notebook 16 is entirely about reading those four. Nothing else is "
           "reported, and nothing else is needed.", "small", anchor="start")

    return "tr-training-loop.svg", c.render(
        "The training loop: a batch goes through a forward pass, a loss, a "
        "backward pass and a weight update, then the loop repeats with the next "
        "batch. Gradient clipping and the learning-rate schedule modify the "
        "update step. The loop reports four numbers: train loss, validation "
        "loss, learning rate and gradient norm."
    ), "One training step, and the four numbers it reports."


# ---------------------------------------------------------------------------
# 14 - loss masking
# ---------------------------------------------------------------------------


@diagram
def loss_masking():
    c = Canvas(900, 400)
    c.title("Loss masking",
            "Which tokens the model is graded on - and which it only reads.")

    segments = [
        ("system", "You are a terse assistant.", "muted", False),
        ("user", "Capital of Kazakhstan?", "muted", False),
        ("assistant", "Astana.", "good", True),
    ]

    x, y = 28, 96
    for role, text, box_role, graded in segments:
        c.box(x, y, 150, 54, role, "role token", role=box_role)
        c.box(x + 162, y, 400, 54, text, role=box_role, mono=True)
        label = "graded  (loss computed here)" if graded else "read only  (label = -100)"
        tone = "t-good" if graded else "t-bad"
        c.text(x + 578, y + 32, label, f"small {tone}", anchor="start")
        y += 72

    c.rule(28, 318, 872, 318)
    c.text(28, 342, "Completion-only masking (the usual default)",
           "lbl", anchor="start")
    c.text(28, 362,
           "The prompt is context the model conditions on, not behaviour you "
           "want it to imitate. Grade it and you", "small", anchor="start")
    c.text(28, 380,
           "teach the model to generate plausible user questions - which is not "
           "the job you hired it for.", "small", anchor="start")

    return "tr-loss-masking.svg", c.render(
        "A chat example split into system, user and assistant segments. The "
        "system and user segments are marked read-only with label -100, so no "
        "loss is computed on them. Only the assistant segment is graded. This "
        "is completion-only masking, the usual default."
    ), "Completion-only loss masking: only the assistant tokens are graded."


# ---------------------------------------------------------------------------
# 15 - RLHF versus DPO
# ---------------------------------------------------------------------------


@diagram
def dpo_vs_rlhf():
    c = Canvas(900, 480)
    c.title("Two ways to learn from preferences",
            "Same data. DPO removes the reward model and the RL loop.")

    c.box(28, 80, 844, 46, "Preference pairs:  (prompt, chosen, rejected)",
          role="accent")

    # --- RLHF, top half ---
    c.text(28, 158, "RLHF / PPO  (2022)", "lbl", anchor="start")
    c.box(28, 172, 150, 56, "Reward model", "trained on the pairs", role="warn")
    c.arrow(182, 200, 218, 200)
    c.box(222, 172, 136, 56, "Sample", "from the policy", role="warn")
    c.arrow(362, 200, 398, 200)
    c.box(402, 172, 136, 56, "Score", "with the reward model", role="warn")
    c.arrow(542, 200, 578, 200)
    c.box(582, 172, 136, 56, "PPO update", "+ KL penalty", role="warn")
    c.path("M 650 228 L 650 248 L 290 248 L 290 228")
    c.text(470, 242, "repeat", "small")
    c.text(740, 196, "4 models in memory", "small t-bad", anchor="start")
    c.text(740, 214, "RL loop, fiddly", "small t-bad", anchor="start")

    c.rule(28, 274, 872, 274)

    # --- DPO, bottom half. Both models score the SAME pair; neither feeds the
    # other, so they sit side by side and converge on the loss. ---
    c.text(28, 306, "DPO  (2023)", "lbl", anchor="start")
    c.box(28, 318, 240, 48, "Policy", "the model you are training",
          role="accent")
    c.box(28, 376, 240, 48, "Frozen reference", "the SFT model, unchanged",
          role="default")
    c.arrow(272, 338, 352, 358)
    c.text(312, 334, "log-probs", "small")
    c.arrow(272, 404, 352, 382)
    c.text(312, 412, "log-probs", "small")
    c.box(356, 340, 250, 60, "One logistic loss",
          "-log sigma(beta x margin)", role="good")
    c.text(628, 358, "2 models, no RL", "small t-good", anchor="start")
    c.text(628, 378, "scored, not sampled", "small t-good", anchor="start")

    c.text(28, 452,
           "The reward model does not get simpler in DPO - it disappears. "
           "Algebra lets the policy's own log-probs", "small", anchor="start")
    c.text(28, 470,
           "stand in for it, which turns the whole problem into supervised "
           "classification over pairs.", "small", anchor="start")

    return "dpo-vs-rlhf.svg", c.render(
        "Both methods start from the same preference pairs. RLHF trains a "
        "separate reward model, then loops: sample from the policy, score with "
        "the reward model, update with PPO plus a KL penalty - four models in "
        "memory. DPO keeps only the policy and a frozen reference model and "
        "optimises a single logistic loss over the pairs directly."
    ), "RLHF trains a reward model and runs PPO; DPO optimises the pairs directly."


# ---------------------------------------------------------------------------
# 16 - loss curve patterns
# ---------------------------------------------------------------------------


def _panel(c, x, y, w, h, name, diagnosis, train_fn, val_fn, tone="bad",
           chance=None):
    """One small-multiples plot. train_fn/val_fn map t in [0,1] to loss in [0,1]."""
    c.box(x, y, w, h, role="ghost", radius=6)
    c.text(x + w / 2, y - 26, name, "lbl")
    c.text(x + w / 2, y - 9, diagnosis, f"small t-{tone}")

    if chance is not None:
        yy = y + h - chance * h
        c.raw(f'<line x1="{x + 6}" y1="{yy:.1f}" x2="{x + w - 6}" y2="{yy:.1f}" '
              f'stroke="#656d76" stroke-width="1" stroke-dasharray="3 3" '
              f'class="d-s-muted_fg"/>')

    def to_path(fn):
        points = []
        for i in range(41):
            t = i / 40
            value = min(max(fn(t), 0.0), 1.0)
            points.append(f"{x + 6 + t * (w - 12):.1f} {y + h - 6 - value * (h - 12):.1f}")
        return "M " + " L ".join(points)

    c.curve(to_path(train_fn), tone="accent")
    c.curve(to_path(val_fn), tone="bad", dashed=True)


@diagram
def loss_curve_patterns():
    c = Canvas(900, 420)
    c.title("Five loss curves, five decisions",
            "Solid blue = train.  Dashed red = validation.  Dotted grey = chance loss.")

    w, h, gap = 152, 118, 24
    x0, y0 = 36, 106

    def decay(rate, floor=0.0):
        return lambda t: floor + (0.92 - floor) * math.exp(-rate * t)

    _panel(c, x0, y0, w, h, "Healthy", "let it run",
           decay(3.2, 0.18), lambda t: 0.04 + 0.2 + (0.92 - 0.2) * math.exp(-3.0 * t),
           tone="good")
    _panel(c, x0 + (w + gap), y0, w, h, "Undertrained", "train longer",
           lambda t: 0.92 - 0.55 * t, lambda t: 0.95 - 0.5 * t, tone="warn")
    _panel(c, x0 + 2 * (w + gap), y0, w, h, "Overfitting", "stop at the dip",
           decay(4.0, 0.04),
           lambda t: 0.25 + 0.67 * math.exp(-9 * t) + 0.75 * max(0.0, t - 0.22) ** 1.4,
           tone="bad")
    _panel(c, x0 + 3 * (w + gap), y0, w, h, "Stuck at chance", "it is a bug",
           lambda t: 0.78 - 0.015 * t, lambda t: 0.80 - 0.01 * t,
           tone="bad", chance=0.87)
    c.text(x0 + 3 * (w + gap) + w / 2, y0 + h - 0.87 * h - 7, "ln(V)", "small")
    _panel(c, x0 + 4 * (w + gap), y0, w, h, "Diverged", "lower the LR",
           lambda t: 0.50 + 0.68 * max(0.0, t - 0.25) ** 1.1,
           lambda t: 0.54 + 0.80 * max(0.0, t - 0.25) ** 1.1, tone="bad")

    c.rule(36, 272, 864, 272)
    c.text(36, 298, "Read validation, decide on validation.",
           "lbl", anchor="start")
    c.text(36, 320,
           "In the middle three panels the train curve is falling or flat - "
           "indistinguishable, at a glance, from the healthy one.",
           "small", anchor="start")
    c.text(36, 340,
           "Only the dashed line separates \"learning\" from \"memorising\" "
           "and \"stuck\". A run logged without a validation set",
           "small", anchor="start")
    c.text(36, 360,
           "cannot be diagnosed at all, which is the single most common reason "
           "a fine-tune ships broken.", "small", anchor="start")
    c.text(36, 392,
           "Notebook 16 produces all five of these from real training runs, "
           "and a diagnose() function that labels them.",
           "small", anchor="start")

    return "loss-curve-patterns.svg", c.render(
        "Five small loss plots. Healthy: train and validation both fall and "
        "flatten together. Undertrained: both still falling at the last step. "
        "Overfitting: train keeps falling while validation dips then climbs. "
        "Stuck at chance: both flat on the chance-loss line. Diverged: both "
        "climb away. In three of the five the train curve alone looks "
        "acceptable, so only the validation curve distinguishes them."
    ), "Five loss-curve shapes and the decision each one implies."
