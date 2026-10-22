#!/usr/bin/env python3
"""Diagrams for the mathematics chapter and notebook 21.

Imported by ``make_diagrams.py``. Each function returns
``(filename, svg, caption)``; the caption doubles as alt text.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import Canvas, diagram  # noqa: E402


@diagram
def attention_pipeline():
    c = Canvas(900, 430)
    c.title("One attention operation, end to end",
            "Every arrow is a matrix multiply or an elementwise step. Nothing else happens.")

    c.box(28, 92, 110, 58, "X", "n x d_model", role="default")
    for dy, label, sub, role in ((0, "Q", "what it seeks", "accent"),
                                 (68, "K", "what it shows", "accent"),
                                 (136, "V", "what it gives", "good")):
        c.arrow(142, 121, 194, 106 + dy)
        c.box(198, 80 + dy, 132, 52, label, sub, role=role)
    c.text(166, 76, "x W", "small")

    c.arrow(334, 106, 386, 130, label="Q K^T", label_dy=-8)
    c.arrow(334, 174, 386, 146)
    c.box(390, 110, 112, 46, "scores", "n x n", role="accent")

    c.arrow(506, 133, 544, 133)
    c.box(548, 110, 120, 46, "÷ sqrt(d_k)", "keeps scale", role="warn")
    c.arrow(672, 133, 710, 133)
    c.box(714, 110, 106, 46, "mask", "-inf above", role="warn")

    c.path("M 767 156 L 767 182 L 446 182 L 446 206", accent=True)
    c.box(390, 210, 112, 46, "softmax", "rows sum to 1", role="accent")

    c.arrow(334, 232, 386, 233)   # V into the product
    c.arrow(506, 233, 560, 233, label="A V", label_dy=-8)
    c.box(564, 210, 126, 46, "output", "n x d_k", role="good")

    c.rule(28, 292, 872, 292)
    c.text(28, 316, "Read it as a sentence:", "lbl", anchor="start")
    c.text(28, 340, "for each position, score every other position by how well "
                    "what I seek matches what it shows;", "small", anchor="start")
    c.text(28, 360, "turn those scores into a distribution; take that weighted "
                    "average of what they give.", "small", anchor="start")
    c.text(28, 392, "The scale and the mask are the only two steps that are not "
                    "in the one-sentence version -", "small", anchor="start")
    c.text(28, 412, "and they are the two that decide whether the model trains "
                    "and whether it cheats.", "small", anchor="start")

    return "mt-attention-pipeline.svg", c.render(
        "X is projected three ways into Q, K and V. Q times K transpose gives "
        "an n by n score matrix, which is divided by the square root of d_k, "
        "masked with minus infinity above the diagonal, and passed through "
        "softmax to give attention weights whose rows sum to one. Those "
        "weights multiply V to give the output."
    ), "The attention operation, step by step."


@diagram
def rope():
    c = Canvas(900, 440)
    c.title("RoPE: position as rotation",
            "The score ends up depending on the gap between tokens, not on where they are.")

    c.text(28, 84, "Each pair of dimensions is rotated by an angle "
                   "proportional to the token's position:", "small", anchor="start")

    # Three clock faces at increasing positions.
    for cx, pos, angle in ((150, 0, 0.0), (330, 1, 0.9), (510, 2, 1.8)):
        cy = 170
        c.circle(cx, cy, 44, role="ghost")
        c.text(cx, cy + 68, f"position {pos}", "small")
        x2 = cx + 36 * math.cos(-angle)
        y2 = cy + 36 * math.sin(-angle)
        c.raw(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" '
              f'stroke="#0969da" stroke-width="2.4" stroke-linecap="round" '
              f'class="d-s-accent"/>')
        c.circle(cx, cy, 3, role="accent")

    c.text(690, 150, "same vector,", "lbl")
    c.text(690, 172, "turned a little", "lbl")
    c.text(690, 194, "further each step", "lbl")
    c.text(690, 222, "length never changes", "small t-good")

    c.rule(28, 258, 872, 258)
    c.text(28, 284, "Why rotation rather than addition", "lbl", anchor="start")
    c.text(28, 308, "Rotating by m and then measuring against something rotated "
                    "by n leaves a rotation by (m - n).", "small", anchor="start")
    c.text(28, 328, "So the inner product carries the GAP and discards the "
                    "absolute positions:", "small", anchor="start")

    c.box(120, 344, 300, 40, "< R_m q , R_n k >  =  g(q, k, m - n)",
          role="accent", mono=True)
    c.text(452, 360, "measured in notebook 21:", "small", anchor="start")
    c.text(452, 378, "gap 3 gives the same value at (3,0) and at (5000,4997)",
           "small t-good", anchor="start")

    c.text(28, 418, "Applied to q and k only - never to v. Values carry content, "
                    "and rotating content corrupts it.", "small", anchor="start")

    return "mt-rope.svg", c.render(
        "A vector drawn as a hand on a clock face, rotated further at each "
        "successive token position, with its length unchanged. Because a "
        "rotation by m measured against a rotation by n leaves a rotation by "
        "m minus n, the inner product between a rotated query and a rotated "
        "key depends only on the gap between the two positions."
    ), "RoPE encodes position as rotation, so scores depend only on the gap."


@diagram
def block_anatomy():
    c = Canvas(900, 500)
    c.title("A pre-norm block",
            "The residual stream runs straight down. Everything else is a correction to it.")

    # The residual stream: one straight line down the left.
    c.raw('<line x1="150" y1="88" x2="150" y2="412" stroke="#0969da" '
          'stroke-width="3" class="d-s-accent"/>')
    c.text(150, 80, "residual stream", "small")

    # Two residual steps. Each: read a normalised copy, compute, add back.
    for norm_y, main_y, label, sub, role, plus_y in (
        (104, 170, "Attention", "move between positions", "accent", 242),
        (266, 332, "SwiGLU MLP", "transform within one", "good", 404),
    ):
        c.box(300, norm_y, 190, 44, "RMSNorm", "fix the scale", role="warn")
        c.arrow(154, norm_y + 22, 296, norm_y + 22, dashed=True)
        c.arrow(395, norm_y + 44, 395, main_y - 4)      # norm -> sub-layer
        c.box(300, main_y, 190, 48, label, sub, role=role)
        # Out of the right edge, down, and back into the stream.
        c.path(f"M 494 {main_y + 24} L 610 {main_y + 24} L 610 {plus_y} "
               f"L 162 {plus_y}", accent=True)
        c.circle(150, plus_y, 11, role="good")
        c.text(150, plus_y + 5, "+", "lbl")

    c.text(700, 126, "read a normalised", "small", anchor="start")
    c.text(700, 144, "copy of the stream", "small", anchor="start")
    c.text(700, 238, "add the result back", "small", anchor="start")
    c.text(700, 256, "(the stream itself is", "small", anchor="start")
    c.text(700, 274, "never normalised)", "small", anchor="start")

    c.rule(28, 436, 872, 436)
    c.text(28, 460, "Because the stream is only ever ADDED to: a block whose "
                    "output is near zero does nothing,", "small", anchor="start")
    c.text(28, 480, "and gradient flows straight down the line. That is what "
                    "makes a 36-layer stack trainable.", "small", anchor="start")

    return "mt-block-anatomy.svg", c.render(
        "A transformer block drawn as a vertical residual stream. Each "
        "sub-layer reads a normalised copy of the stream, computes a result, "
        "and adds it back. RMSNorm then attention form the first residual "
        "step; RMSNorm then the SwiGLU MLP form the second. The stream itself "
        "is never normalised, so gradient passes straight through."
    ), "A pre-norm block: the residual stream, and two corrections added to it."
