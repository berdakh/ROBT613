#!/usr/bin/env python3
"""Diagrams for the agents track (notebooks 17-20).

Imported by ``make_diagrams.py``, which registers these into the shared
``DIAGRAMS`` list and renders them. Each function returns
``(filename, svg, caption)``; the caption doubles as alt text.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import Canvas, diagram  # noqa: E402

# ---------------------------------------------------------------------------
# 17 - memory strategies
# ---------------------------------------------------------------------------


@diagram
def memory_strategies():
    c = Canvas(900, 420)
    c.title("Four ways to remember",
            "The model has no memory. These are four ways your code fakes it.")

    rows = [
        ("Keep everything", "ConversationBuffer", "nothing, until it overflows",
         "good", "the right default"),
        ("Keep the newest", "WindowedMemory", "the beginning - where the task was stated",
         "warn", "bounded, and forgets requirements"),
        ("Compress the old", "SummarisingMemory", "detail, chosen by the summariser",
         "warn", "a longer conversation, lossily"),
        ("Write it down", "FactStore", "everything you did not choose",
         "accent", "the only one that survives a restart"),
    ]

    y = 84
    for label, cls, loses, role, note in rows:
        c.box(28, y, 166, 56, label, cls, role=role, mono=False)
        c.text(214, y + 24, "loses:", "small", anchor="start")
        c.text(262, y + 24, loses, "lbl", anchor="start")
        c.text(214, y + 43, note, "small", anchor="start")
        y += 68

    c.rule(28, 360, 872, 360)
    c.text(28, 384, "Only the last row is memory in the sense people mean. The "
                    "first three are ways to fit", "small", anchor="start")
    c.text(28, 404, "a conversation into a window - and every one of them drops "
                    "the rule stated in turn three.", "small", anchor="start")

    return "ag-memory-strategies.svg", c.render(
        "Four memory strategies as rows. Keep everything (ConversationBuffer) "
        "loses nothing until it overflows. Keep the newest (WindowedMemory) "
        "loses the beginning, where the task was stated. Compress the old "
        "(SummarisingMemory) loses detail chosen by the summariser. Write it "
        "down (FactStore) loses everything you did not choose, and is the only "
        "one that survives a restart."
    ), "Four memory strategies, and what each one loses."


# ---------------------------------------------------------------------------
# 18 - a plan is a graph
# ---------------------------------------------------------------------------


@diagram
def plan_graph():
    c = Canvas(900, 430)
    c.title("A plan is a graph, not a list",
            "Which is why failure has two different consequences.")

    # Wave headers
    for i, x in enumerate((60, 330, 620)):
        c.text(x + 95, 80, f"wave {i}", "small")

    # Wave 0
    c.box(60, 92, 190, 52, "search: RAG", "no dependencies", role="good")
    c.box(60, 164, 190, 52, "search: metrics", "no dependencies", role="bad")
    c.text(155, 236, "✗ failed", "small t-bad")

    # Wave 1
    c.box(330, 92, 190, 52, "read the RAG hit", "needs search: RAG", role="good")
    c.box(330, 164, 190, 52, "read the metrics hit", "needs search: metrics",
          role="muted")
    c.text(425, 236, "∅ blocked - never ran", "small t-warn")

    # Wave 2
    c.box(620, 128, 190, 52, "write the answer", "needs both reads", role="muted")
    c.text(715, 200, "∅ blocked", "small t-warn")

    c.arrow(254, 118, 326, 118, accent=True)
    c.arrow(254, 190, 326, 190, dashed=True)
    c.arrow(524, 118, 616, 142, accent=True)
    c.arrow(524, 190, 616, 166, dashed=True)

    c.rule(28, 268, 872, 268)
    c.text(28, 294, "Two things a list cannot express:", "lbl", anchor="start")
    c.text(28, 318, "1.  The two searches are independent, so they run together. "
                    "Three waves is the fastest this plan can be.",
           "small", anchor="start")
    c.text(28, 340, "2.  One failure blocks exactly its descendants. The other "
                    "branch finished, and its results are kept.",
           "small", anchor="start")
    c.text(28, 372, "failed ≠ blocked", "lbl", anchor="start")
    c.text(142, 372, "A failed task ran and broke. A blocked task never ran. "
                     "Conflating them sends you", "small", anchor="start")
    c.text(28, 392, "debugging a task that did nothing wrong - and replanning "
                    "only needs to replace the ones that stuck.",
           "small", anchor="start")

    return "ag-plan-graph.svg", c.render(
        "A plan drawn as a dependency graph in three waves. Wave 0 has two "
        "independent searches; one succeeds and one fails. Wave 1 has a read "
        "for each; the one depending on the failed search is blocked rather "
        "than failed. Wave 2, which needs both, is also blocked. The "
        "successful branch completed and its results are kept."
    ), "A plan as a dependency graph: failure blocks descendants, not siblings."


# ---------------------------------------------------------------------------
# 19 - team patterns
# ---------------------------------------------------------------------------


@diagram
def team_patterns():
    c = Canvas(900, 470)
    c.title("Four ways to wire several agents",
            "Costs rise left to right. Start at the left.")

    # --- Router ---
    c.text(120, 80, "Router", "lbl")
    c.text(120, 96, "the receptionist", "small")
    c.text(120, 112, "2 model calls", "small t-good")
    c.box(60, 124, 120, 40, "router", role="accent")
    c.arrow(120, 168, 86, 196)
    c.arrow(120, 168, 154, 196)
    c.box(40, 200, 86, 38, "A", role="default")
    c.box(136, 200, 86, 38, "B", role="default")
    c.text(120, 260, "one specialist runs", "small")

    # --- Orchestrator ---
    c.text(360, 80, "Orchestrator", "lbl")
    c.text(360, 96, "the wedding planner", "small")
    c.text(360, 112, "~3x the calls", "small t-warn")
    c.box(300, 124, 120, 40, "orchestrator", role="accent")
    # Three children centred on 360: width 76, gap 8, so 238 / 322 / 406.
    for x in (238, 322, 406):
        c.arrow(360, 168, x + 38, 196)
        c.box(x, 200, 76, 38, {238: "A", 322: "B", 406: "C"}[x], role="default")
    c.path("M 276 238 L 276 262 L 444 262 L 444 238", arrow=False)
    c.arrow(360, 262, 360, 284)
    c.box(300, 288, 120, 38, "synthesise", role="good")
    c.text(360, 344, "all specialists run,", "small")
    c.text(360, 360, "used or not", "small")

    # --- Pipeline ---
    c.text(600, 80, "Pipeline", "lbl")
    c.text(600, 96, "the passport office", "small")
    c.text(600, 112, "lossy at each step", "small t-warn")
    for i, (label, yy) in enumerate((("A", 124), ("B", 182), ("C", 240))):
        c.box(548, yy, 104, 38, label, role="default")
        if i < 2:
            c.arrow(600, yy + 38, 600, yy + 54)
    c.text(600, 304, "each step sees only", "small")
    c.text(600, 320, "the one before it", "small")

    # --- Critic loop ---
    c.text(810, 80, "Critic loop", "lbl")
    c.text(810, 96, "writer and editor", "small")
    c.text(810, 112, "2 calls per round", "small t-warn")
    c.box(752, 124, 116, 40, "writer", role="accent")
    c.box(752, 208, 116, 40, "critic", role="warn")
    c.arrow(790, 168, 790, 204, label="draft", label_dy=-4)
    c.path("M 838 208 L 858 208 L 858 164 L 838 164", accent=True)
    c.text(810, 268, "until APPROVED", "small")
    c.text(810, 284, "or rounds run out", "small")

    c.rule(28, 388, 872, 388)
    c.text(28, 412, "A team is only justified by: separate tool permissions, "
                    "context that will not fit one window,", "small", anchor="start")
    c.text(28, 432, "or an adversarial role. Everything else is functions with "
                    "good names.", "small", anchor="start")
    c.text(28, 458, "Only the critic loop benefits from the information loss; "
                    "for the rest it is the price.", "small", anchor="start")

    return "ag-team-patterns.svg", c.render(
        "Four multi-agent patterns side by side. A router picks one specialist "
        "and costs two model calls. An orchestrator fans out to all "
        "specialists and synthesises, costing about three times as much. A "
        "pipeline passes output from one agent to the next, losing detail at "
        "each step. A critic loop sends a draft between a writer and a critic "
        "until approved or the round cap is hit."
    ), "Router, orchestrator, pipeline and critic loop, with their costs."


# ---------------------------------------------------------------------------
# 20 - the evaluation quadrants
# ---------------------------------------------------------------------------


@diagram
def eval_quadrants():
    c = Canvas(900, 486)
    c.title("Outcome is half the picture",
            "An agent can be right for a reason that will not hold tomorrow.")

    left, top, w, h = 248, 104, 250, 100
    gap = 12

    # Column headers, one per column - a single centred string ran them together.
    c.text(left + w / 2, top - 16, "answer WRONG", "small")
    c.text(left + w + gap + w / 2, top - 16, "answer RIGHT", "small")
    # Row labels to the left of each row.
    c.text(left - 16, top + h / 2 + 4, "path sane", "small", anchor="end")
    c.text(left - 16, top + h + gap + h / 2 + 4, "path wrong", "small", anchor="end")

    c.box(left, top, w, h, "Broken, honestly",
          "wrong answer, sane method", role="default")
    c.box(left + w + gap, top, w, h, "Working",
          "right answer, sound method", role="good")
    c.box(left, top + h + gap, w, h, "Broken, loudly",
          "wrong answer AND wrong path", role="bad")
    c.box(left + w + gap, top + h + gap, w, h, "LUCKY PASS",
          "right answer, wrong path", role="warn")

    c.text(left + w + gap + w / 2, top + 2 * h + gap + 22,
           "an outcome-only suite calls this a pass", "small t-warn")

    c.rule(28, 352, 872, 352)
    c.text(28, 376, "What lands in the lucky-pass box:", "lbl", anchor="start")
    c.text(28, 398, "·  guessed, and was right      ·  used 11 tools where 2 would do"
                    "      ·  looped to max_steps, then bluffed",
           "small", anchor="start")
    c.text(28, 418, "·  sent an email on a read-only question — the answer was "
                    "accurate and the behaviour was unacceptable",
           "small", anchor="start")
    c.text(28, 442, "Every one scores 100% on the answer. Only the trajectory "
                    "tells them apart, and only the trajectory", "small", anchor="start")
    c.text(28, 462, "predicts what happens next week.", "small", anchor="start")

    return "ag-eval-quadrants.svg", c.render(
        "A two-by-two grid. The horizontal axis is whether the answer is "
        "right; the vertical axis is whether the path was sane. Right answer "
        "with a sound method is working. Wrong on both is broken and loud. The "
        "dangerous quadrant is a right answer reached by a wrong path - a "
        "lucky pass, which an outcome-only test suite reports as success."
    ), "Outcome versus trajectory: the lucky-pass quadrant is the dangerous one."
