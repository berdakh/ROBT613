#!/usr/bin/env python3
"""Diagrams for handbook chapter 8 - who builds these models, and what it costs.

Imported by ``make_diagrams.py``. Each function returns
``(filename, svg, caption)``; the caption doubles as alt text.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import Canvas, diagram, wrap  # noqa: E402


@diagram
def params_total_vs_active():
    c = Canvas(900, 496)
    c.title("Total parameters buy memory. Active parameters buy speed.",
            "A Mixture-of-Experts model routes each token to a few experts, "
            "so the two numbers come apart.")

    # --- Dense: every parameter runs for every token ----------------------
    c.text(28, 86, "Dense - one block, all of it runs", "lbl", anchor="start")
    c.box(28, 98, 404, 54, "every parameter, every token", role="accent")
    c.text(28, 174, "load it all, compute it all", "small", anchor="start")
    c.text(28, 194, "memory cost = compute cost", "small t-accent", anchor="start")

    # --- MoE: loaded whole, used in part ----------------------------------
    c.text(468, 86, "Mixture of Experts - loaded whole, used in part",
           "lbl", anchor="start")
    for i in range(8):
        x = 468 + i * 51
        live = i in (2, 5)
        c.box(x, 98, 44, 54, "", role="good" if live else "ghost")
        c.text(x + 22, 131, "on" if live else "off",
               "small t-good" if live else "small")
    c.text(468, 174, "all 8 experts sit in memory; 2 of them do the work",
           "small", anchor="start")
    c.text(468, 194, "memory cost >> compute cost", "small t-good", anchor="start")

    # --- Real numbers -----------------------------------------------------
    c.rule(28, 222, 872, 222)
    c.text(28, 248, "What that looks like on real models", "lbl", anchor="start")

    headers = (("model", 28), ("total", 404), ("active", 534), ("you need", 664))
    for label, x in headers:
        c.text(x, 274, label, "small", anchor="start")

    rows = [
        ("Qwen3-8B (dense)", "8B", "8B", "~16 GB at bf16", "accent"),
        ("gpt-oss-20b (MoE)", "21B", "3.6B", "a 16 GB GPU", "good"),
        ("gpt-oss-120b (MoE)", "117B", "5.1B", "one 80 GB GPU", "good"),
        ("DeepSeek-V3 (MoE)", "671B", "37B", "a multi-GPU node", "warn"),
    ]
    y = 292
    for name, total, active, need, role in rows:
        c.box(28, y, 844, 36, "", role="ghost")
        c.text(40, y + 23, name, "small", anchor="start")
        c.text(404, y + 23, total, "small", anchor="start")
        c.text(534, y + 23, active, "small t-" + ("accent" if role == "accent" else "good"),
               anchor="start")
        c.text(664, y + 23, need, "small", anchor="start")
        y += 44

    c.text(28, 486,
           "So \"671B\" and \"8B\" are not on the same scale. Ask which number "
           "a comparison is quoting before you believe it.",
           "small", anchor="start")

    return "ec-params.svg", c.render(
        "A dense model is drawn as one block in which every parameter runs for "
        "every token, so memory cost equals compute cost. A Mixture-of-Experts "
        "model is drawn as eight expert blocks of which two are active, so all "
        "of them sit in memory while only a couple do the work. A table gives "
        "real figures: Qwen3-8B is 8B total and 8B active; gpt-oss-20b is 21B "
        "total and 3.6B active; gpt-oss-120b is 117B total and 5.1B active; "
        "DeepSeek-V3 is 671B total and 37B active."
    ), "Total parameters buy memory; active parameters buy speed."


@diagram
def cost_stack():
    c = Canvas(900, 520)
    c.title("What a model costs, and which part gets quoted",
            "The quoted figure is one line item. These are the ones it leaves out.")

    # The quoted number.
    c.box(28, 92, 300, 60, "The final training run",
          "the number in the press release", role="good")
    c.text(344, 118, "DeepSeek-V3 disclosed 2.788M H800 GPU-hours",
           "small", anchor="start")
    c.text(344, 138, "and costed it at ~$5.58M at an assumed $2/GPU-hour",
           "small", anchor="start")

    # What it leaves out.
    c.text(28, 194, "What that figure does not include", "lbl", anchor="start")

    omitted = [
        ("Failed and abandoned runs", "the ones that diverged, or trained the wrong thing"),
        ("Research compute", "ablations, scaling experiments, data mixes tried and dropped"),
        ("Salaries", "Epoch puts R&D staff at 29-49% of total development cost"),
        ("Data work", "collection, licensing, cleaning, deduplication, annotation"),
        ("The cluster itself", "bought or reserved long before the run starts"),
    ]
    y = 212
    for label, detail in omitted:
        c.box(28, y, 300, 44, label, role="bad")
        c.text(344, y + 27, detail, "small", anchor="start")
        y += 52

    c.rule(28, 478, 872, 478)
    c.text(28, 500,
           "Epoch AI: the final run is a minority of total R&D spend, and "
           "frontier costs have risen roughly 2.4x a year since 2016.",
           "small", anchor="start")

    return "ec-cost-stack.svg", c.render(
        "The final training run is drawn as a single small box - the figure "
        "quoted in press releases, such as DeepSeek-V3's disclosed 2.788 "
        "million H800 GPU-hours costed at about 5.58 million dollars. Beneath "
        "it sit five larger boxes it excludes: failed and abandoned runs, "
        "research compute, salaries, data work, and the cluster itself. Epoch "
        "AI finds the final run is a minority of total R&D spend."
    ), "The quoted training cost covers the final run only."


@diagram
def disclosure_gradient():
    c = Canvas(900, 488)
    c.title("How open is it, really?",
            "Four rungs, not two. \"Open\" in a headline can mean any of them.")

    rungs = [
        ("Fully open", "weights + data + code + checkpoints",
         "OLMo", "reproducible: you could retrain it", "good"),
        ("Open weights", "weights + a licence; data and code withheld",
         "Qwen, Llama, DeepSeek, gpt-oss",
         "you can run, fine-tune and audit behaviour - not provenance", "accent"),
        ("Open weights, restricted", "weights, but the licence limits who or how",
         "some Llama and Gemma terms",
         "read the licence before you build a product on it", "warn"),
        ("Closed", "an API, a price list and a system card",
         "GPT, Claude, Gemini, Grok",
         "no parameter count is published - quoted ones are rumour", "bad"),
    ]

    y = 86
    for label, what, examples, consequence, role in rungs:
        c.box(28, y, 210, 76, label, role=role)
        c.text(252, y + 24, what, "small", anchor="start")
        c.text(252, y + 44, examples, "small t-" + role, anchor="start")
        for n, line in enumerate(wrap(consequence, 600, 11.5)):
            c.text(252, y + 64 + n * 15, line, "small", anchor="start")
        y += 90

    c.rule(28, 450, 872, 450)
    c.text(28, 474,
           "The useful question is never \"is it open?\" but \"which of these "
           "four, and does that licence cover what I am about to do?\"",
           "small", anchor="start")

    return "ec-disclosure.svg", c.render(
        "Four rungs of openness rather than two. Fully open releases weights, "
        "data, code and checkpoints, like OLMo, and could be retrained. Open "
        "weights releases the weights and a licence but withholds data and "
        "code, like Qwen, Llama, DeepSeek and gpt-oss. Open weights with a "
        "restricted licence limits who may use them or how. Closed models such "
        "as GPT, Claude, Gemini and Grok offer an API, a price list and a "
        "system card, and publish no parameter count at all."
    ), "Openness is four rungs, not two."
