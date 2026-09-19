#!/usr/bin/env python3
"""Diagrams for the handbook (docs/handbook/).

Rendered by ``tools/make_diagrams.py``, which imports this module. Same drawing
kit and the same light/dark colour contract - see ``tools/svgkit.py``.
"""

from __future__ import annotations

from svgkit import Canvas, diagram

# ---------------------------------------------------------------------------
# Chapter 1 - how we got here
# ---------------------------------------------------------------------------


@diagram
def timeline():
    c = Canvas(880, 420)
    c.title("How we got here",
            "Each step solved a problem the previous one created.")

    c.rule(40, 92, 840, 92)
    events = [
        (40, "2017", "Transformer", "attention replaces\nrecurrence", "accent"),
        (175, "2018", "BERT / GPT", "pretrain once,\nreuse everywhere", "accent"),
        (310, "2020", "GPT-3", "scale alone buys\nnew abilities", "good"),
        (445, "2022", "InstructGPT\nChatGPT", "RLHF makes models\nfollow instructions", "good"),
        (580, "2023", "LLaMA + llama.cpp", "open weights run\non a laptop", "warn"),
        (715, "2025", "Reasoning\n+ agents", "models think, then\nuse tools", "bad"),
    ]
    for x, year, name, why, role in events:
        c.circle(x + 62, 92, 7, role)
        c.text(x + 62, 72, year, "lbl")
        c.box(x, 118, 125, 96, name.replace("\n", " "), why.replace("\n", " "), role)

    c.box(40, 246, 390, 74, "The recurring pattern",
          "capability arrives -> it is too expensive -> someone makes it cheap "
          "-> it becomes infrastructure", "default")
    c.box(450, 246, 390, 74, "Why open weights mattered",
          "LLaMA leaked in March 2023; llama.cpp ran it on a MacBook within "
          "days. That week created this whole ecosystem.", "accent")

    c.text(40, 356, "Read the dates loosely. The lesson is the shape, not the "
                    "calendar:", "small", anchor="start")
    c.text(40, 378, "every tool in this handbook exists because something that "
                    "worked was too slow, too big, or too awkward to use.",
           "small", anchor="start")
    c.text(40, 400, "Knowing which problem a tool was born to solve tells you "
                    "when to stop using it.", "small", anchor="start")
    return "hb-timeline.svg", c.render(
        "A timeline from the 2017 transformer through BERT and GPT, GPT-3 scaling, "
        "InstructGPT and ChatGPT, open weights with LLaMA and llama.cpp, to "
        "reasoning models and agents."
    ), "How we got here"


# ---------------------------------------------------------------------------
# Chapter 2 - the mechanism
# ---------------------------------------------------------------------------


@diagram
def transformer_block():
    c = Canvas(880, 440)
    c.title("Inside the model",
            "The same block, stacked 28 to 80 times. That is essentially the whole architecture.")

    c.box(40, 88, 180, 44, "Token ids", "[791, 6864, 315]", role="default", mono=True)
    c.arrow(130, 136, 130, 164, accent=True)
    c.box(40, 168, 180, 44, "Embedding table", "id -> vector", "accent")
    c.arrow(224, 190, 268, 190, accent=True)

    c.box(272, 92, 300, 250, "", role="ghost")
    c.text(422, 118, "One transformer block  (x N)", "lbl")

    c.box(292, 136, 260, 62, "Self-attention",
          "every token looks at the tokens before it", "accent")
    c.text(422, 214, "+ residual, normalise", "small")
    c.box(292, 226, 260, 62, "Feed-forward network",
          "where most of the parameters live", "accent")
    c.text(422, 304, "+ residual, normalise", "small")
    c.path("M 562 167 L 586 167 L 586 320 L 422 320 L 422 300", dashed=True)

    c.arrow(576, 190, 620, 190, accent=True)
    c.box(624, 168, 216, 44, "Final layer", "vector -> one score per token", "good")
    c.arrow(732, 216, 732, 246, accent=True)
    c.box(624, 250, 216, 44, "softmax", "scores -> probabilities", "good")

    c.box(40, 362, 800, 58,
          "Attention decides WHICH earlier tokens matter. The feed-forward layers hold "
          "WHAT the model knows. Depth lets it compose both, repeatedly.", role="default")
    return "hb-transformer-block.svg", c.render(
        "Token ids become embeddings, pass through N identical blocks of "
        "self-attention and a feed-forward network with residual connections, "
        "then a final layer and softmax produce token probabilities."
    ), "Inside a transformer"


@diagram
def attention():
    c = Canvas(880, 380)
    c.title("What attention does",
            "Resolving a word by looking back at the right earlier words.")

    tokens = ["The", "student", "opened", "her", "laptop", "because", "it", "was"]
    x = 40
    positions = []
    for tok in tokens:
        w = max(62, len(tok) * 11)
        role = "accent" if tok == "it" else "default"
        c.box(x, 96, w, 40, tok, role=role, mono=True)
        positions.append((x + w / 2, w))
        x += w + 10

    it_x = positions[6][0]
    # Attention weights from "it" back to earlier tokens.
    weights = [0.02, 0.06, 0.03, 0.04, 0.71, 0.02]
    for i, weight in enumerate(weights):
        src = positions[i][0]
        thickness = weight > 0.3
        c.path(f"M {it_x:.0f} 96 C {it_x:.0f} 40, {src:.0f} 40, {src:.0f} 92",
               accent=thickness, dashed=not thickness, arrow=False)
        c.text(src, 74, f"{weight:.0%}", "small" + (" t-accent" if thickness else ""))

    c.text(440, 176, '"it" attends mostly to "laptop" - that is how the model '
                     'resolves the pronoun.', "small")

    c.box(40, 206, 390, 80, "Why this replaced recurrence",
          "Every position is computed in parallel, and any token can reach any "
          "earlier token in one step.", "good")
    c.box(450, 206, 390, 80, "What it costs",
          "Comparing every token with every other is quadratic - which is why "
          "context windows are expensive.", "warn")

    c.text(40, 322, "Multi-head attention runs several of these in parallel, so one head can "
                    "track grammar while", "small", anchor="start")
    c.text(40, 342, "another tracks subject matter. Their outputs are concatenated.",
           "small", anchor="start")
    return "hb-attention.svg", c.render(
        "The word 'it' attends 71 percent to 'laptop' and only a few percent to "
        "other tokens, which is how the model resolves the pronoun."
    ), "What attention does"


# ---------------------------------------------------------------------------
# Chapter 3 - training
# ---------------------------------------------------------------------------


@diagram
def training_lifecycle():
    c = Canvas(880, 470)
    c.title("The full training lifecycle",
            "Four phases. Only the last one is yours - and it is the cheapest by far.")

    phases = [
        (40, "0. Data", "crawl, filter,\ndedupe, tokenize", "months of\nengineering", "default"),
        (250, "1. Pre-training", "predict the next\ntoken, 10-15T of them", "$1M - $100M+\nweeks on 1000s of GPUs", "bad"),
        (460, "2. Post-training", "SFT, then preference\noptimisation", "$10k - $1M\ndays", "warn"),
        (670, "3. Adaptation", "prompt, RAG, tools,\nLoRA", "$0 - $100\nminutes to hours", "good"),
    ]
    for i, (x, name, what, cost, role) in enumerate(phases):
        c.box(x, 92, 170, 124, role=role)
        c.text(x + 85, 118, name, "lbl")
        c.text(x + 85, 144, what.split("\n")[0], "small")
        c.text(x + 85, 160, what.split("\n")[1], "small")
        c.text(x + 85, 186, cost.split("\n")[0], "small t-" + ("bad" if role == "bad" else "warn" if role == "warn" else "good" if role == "good" else "accent"))
        c.text(x + 85, 202, cost.split("\n")[1], "small")
        if i < 3:
            c.arrow(x + 174, 154, x + 246, 154, accent=True)

    c.text(125, 244, "a base model", "small")
    c.text(545, 244, "an instruct model", "small")
    c.text(755, 244, "your application", "small")

    c.box(40, 268, 390, 104, role="ghost")
    c.text(60, 292, "What each phase actually buys", "lbl", anchor="start")
    for i, line in enumerate([
        "Pre-training  ->  knowledge and language",
        "Post-training ->  following instructions, tone, refusal",
        "Adaptation    ->  your data, your format, your tools",
    ]):
        c.text(60, 316 + i * 20, line, "small", anchor="start")

    c.box(450, 268, 390, 104, role="accent")
    c.text(470, 292, "The practical consequence", "lbl", anchor="start")
    for i, line in enumerate([
        "You will almost never pre-train.",
        "You will rarely post-train.",
        "You will constantly adapt - and that is",
        "where nearly all the value is created.",
    ]):
        c.text(470, 314 + i * 17, line, "small", anchor="start")

    c.box(40, 388, 800, 62,
          "A base model completes text. It does NOT follow instructions - that behaviour is "
          "installed in phase 2. Download a -Base checkpoint expecting a chatbot and it "
          "will ramble at you.", role="warn")
    return "hb-training-lifecycle.svg", c.render(
        "Four training phases - data, pre-training, post-training, adaptation - "
        "with their costs falling from tens of millions of dollars to nearly nothing."
    ), "The full training lifecycle"


@diagram
def pretraining_data():
    c = Canvas(880, 400)
    c.title("Phase 0: making the data",
            "Most of the work, and the part nobody publishes.")

    steps = [
        (40, "Raw crawl", "~100 TB of\nweb pages", "default"),
        (215, "Quality filter", "drop spam, boilerplate,\nmachine text", "accent"),
        (390, "Deduplicate", "near-duplicates teach\nmemorisation", "accent"),
        (565, "Decontaminate", "remove benchmark\ntest sets", "warn"),
        (740, "Tokenize", "text -> ids\n~15T tokens", "good"),
    ]
    for i, (x, name, sub, role) in enumerate(steps):
        c.box(x, 96, 160, 86, name, sub.replace("\n", " "), role)
        if i < 4:
            c.arrow(x + 164, 139, x + 211, 139, accent=True)

    # A funnel showing how much survives.
    c.text(40, 216, "Roughly how much survives:", "small", anchor="start")
    widths = [800, 320, 210, 205, 200]
    labels = ["100%", "~40%", "~26%", "~25%", "kept"]
    y = 230
    for w, label, (_x, _n, _s, role) in zip(widths, labels, steps, strict=True):
        c.box(40, y, w, 18, "", role=role, radius=3)
        c.text(50 + w, y + 14, label, "small", anchor="start")
        y += 24

    c.box(40, 356, 390, 40, "Why dedupe matters",
          "duplicated text is memorised, not learned", "accent")
    c.box(450, 356, 390, 40, "Why decontaminate matters",
          "otherwise your benchmark score is a lie", "warn")
    return "hb-pretraining-data.svg", c.render(
        "The data pipeline: raw crawl, quality filtering, deduplication, "
        "decontamination and tokenization, with roughly a quarter of the "
        "original text surviving."
    ), "Phase 0: making the data"


@diagram
def scaling_laws():
    c = Canvas(880, 400)
    c.title("Scaling laws: how big, trained on how much?",
            "Given a fixed compute budget, there is a right answer.")

    c.box(40, 92, 250, 96, "The question",
          "You can afford N GPU-hours. Spend them on a bigger model, or more data?",
          "default")
    c.box(310, 92, 250, 96, "Chinchilla (2022)",
          "Most models were too big and undertrained. Scale BOTH, roughly "
          "20 tokens per parameter.", "accent")
    c.box(580, 92, 260, 96, "What changed since",
          "Inference cost dominates, so models are now deliberately "
          "overtrained: smaller, on far more data.", "good")

    c.text(40, 226, "Tokens per parameter, in practice:", "lbl", anchor="start")
    rows = [
        ("GPT-3 (2020)", "175B params", "300B tokens", "~2", 24, "bad"),
        ("Chinchilla-optimal", "-", "-", "~20", 96, "accent"),
        ("Llama 3 8B (2024)", "8B params", "15T tokens", "~1875", 700, "good"),
    ]
    y = 248
    for name, params, tokens, ratio, width, role in rows:
        c.text(40, y + 17, name, "small", anchor="start")
        c.text(190, y + 17, params, "small", anchor="start")
        c.text(300, y + 17, tokens, "small", anchor="start")
        c.box(410, y, width, 22, "", role=role, radius=3)
        c.text(418 + width, y + 17, f"{ratio} tok/param", "small", anchor="start")
        y += 34

    c.box(40, 356, 800, 40,
          "A small model trained far past 'optimal' is cheap to run forever. That is why "
          "a good 4B model exists at all.", role="accent")
    return "hb-scaling-laws.svg", c.render(
        "Scaling laws: GPT-3 used about 2 tokens per parameter, Chinchilla found "
        "20 to be compute-optimal, and modern small models like Llama 3 8B use "
        "nearly 2000 because inference cost now dominates."
    ), "Scaling laws"


@diagram
def post_training():
    c = Canvas(880, 470)
    c.title("Phase 2: post-training",
            "Turning a text completer into something that follows instructions.")

    c.box(40, 92, 150, 60, "Base model", "completes text", "muted")
    c.arrow(194, 122, 232, 122, accent=True)

    c.box(236, 88, 200, 68, "SFT", "supervised fine-tuning on "
          "instruction/response pairs", "accent")
    c.arrow(440, 122, 478, 122, accent=True)

    c.box(482, 88, 200, 68, "Preference optimisation",
          "learn which answer people prefer", "accent")
    c.arrow(686, 122, 724, 122, accent=True)
    c.box(728, 92, 112, 60, "Instruct model", "", "good")

    # Preference optimisation, expanded.
    c.text(40, 200, "Two ways to do the preference step:", "lbl", anchor="start")

    c.box(40, 220, 390, 140, "", role="ghost")
    c.text(235, 244, "RLHF  (2022)", "lbl")
    c.box(60, 256, 160, 40, "Reward model", "learns to score", "accent")
    c.arrow(224, 276, 254, 276)
    c.box(258, 256, 152, 40, "RL (PPO)", "optimise against it", "accent")
    c.path("M 334 298 L 334 322 L 140 322 L 140 300", dashed=True)
    c.text(235, 342, "powerful, fiddly, unstable", "small")

    c.box(450, 220, 390, 140, "", role="ghost")
    c.text(645, 244, "DPO and friends  (2023+)", "lbl")
    c.box(470, 256, 350, 40, "One training objective on (preferred, rejected) pairs",
          role="good")
    c.text(645, 316, "no reward model, no RL loop", "small")
    c.text(645, 342, "simpler, cheaper, now the default", "small")

    c.box(40, 382, 800, 76, role="warn")
    c.text(440, 406, "Reasoning models add a third step: RL against answers that can be "
                     "checked automatically", "lbl")
    c.text(440, 426, "- maths with a known result, code that passes tests. No human "
                     "labels, so it scales.", "small")
    c.text(440, 446, "This is what produced thinking mode.", "small")
    return "hb-post-training.svg", c.render(
        "Post-training: a base model gets supervised fine-tuning, then preference "
        "optimisation by either RLHF with a reward model or the simpler DPO "
        "objective, producing an instruct model."
    ), "Phase 2: post-training"


@diagram
def adaptation_ladder():
    c = Canvas(880, 420)
    c.title("Phase 3: your options, cheapest first",
            "Climb only as far as you must.")

    rungs = [
        ("Prompting", "minutes", "free", "change the words you send", "good", 0),
        ("Few-shot examples", "minutes", "free", "show it what you mean", "good", 1),
        ("RAG", "hours", "~free", "give it your documents", "good", 2),
        ("Tools", "hours", "~free", "let it act and compute", "accent", 3),
        ("LoRA fine-tune", "hours", "$1-$50", "change its style and format", "warn", 4),
        ("Full fine-tune", "days", "$1k+", "rarely the right answer", "bad", 5),
        ("Pre-train", "months", "$1M+", "essentially never", "bad", 6),
    ]
    y = 92
    for name, time, cost, what, role, i in rungs:
        # The bars widen down the ladder. Keep the trailing label short so the
        # widest row still fits: 40 + 660 + a two-word cost stays on canvas.
        width = 300 + i * 60
        c.box(40, y, width, 34, "", role=role, radius=4)
        c.text(56, y + 23, f"{name}  -  {what}", "small", anchor="start")
        c.text(40 + width + 14, y + 23, f"{time} · {cost}", "small", anchor="start")
        y += 42

    c.box(40, 386, 800, 30, "", role="accent")
    c.text(440, 406, "Each rung costs about 10x the one below it. Most teams jump "
                     "three rungs too far.", "small")
    return "hb-adaptation-ladder.svg", c.render(
        "A ladder of adaptation options from prompting, few-shot, RAG and tools "
        "through LoRA to full fine-tuning and pre-training, each roughly ten "
        "times more expensive than the last."
    ), "Adaptation options, cheapest first"


@diagram
def distillation_moe():
    c = Canvas(880, 380)
    c.title("Two tricks that shape the models you download",
            "Why a 4B model is good, and why a 30B model can be fast.")

    c.text(40, 92, "Distillation", "lbl", anchor="start")
    c.box(40, 106, 170, 60, "Large teacher", "expensive, capable", "muted")
    c.arrow(214, 136, 254, 136, accent=True)
    c.box(258, 106, 190, 60, "Generate answers", "millions of them", "accent")
    c.arrow(452, 136, 492, 136, accent=True)
    c.box(496, 106, 170, 60, "Train a student", "small, fast", "good")
    c.text(690, 130, "The small model learns", "small", anchor="start")
    c.text(690, 148, "from the big one's output.", "small", anchor="start")

    c.text(40, 216, "Mixture of Experts (MoE)", "lbl", anchor="start")
    c.box(40, 230, 150, 60, "A token", "", "default")
    c.arrow(194, 260, 230, 260, accent=True)
    c.box(234, 230, 120, 60, "Router", "picks 2 of 8", "accent")
    for i in range(4):
        role = "good" if i in (1, 2) else "ghost"
        c.box(390 + i * 90, 230, 78, 60, f"Expert {i + 1}", "", role)
    c.text(752, 250, "only the chosen", "small", anchor="start")
    c.text(752, 268, "experts run", "small", anchor="start")

    c.box(40, 316, 800, 50,
          "MoE: memory of a 30B model, speed closer to a 3B one. You still need RAM for "
          "ALL the experts - only the compute is saved.", role="warn")
    return "hb-distillation-moe.svg", c.render(
        "Distillation trains a small student model on a large teacher's outputs. "
        "Mixture of Experts routes each token to a couple of experts, saving "
        "compute but not memory."
    ), "Distillation and Mixture of Experts"


# ---------------------------------------------------------------------------
# Chapter 4 - the model landscape
# ---------------------------------------------------------------------------


@diagram
def model_landscape():
    c = Canvas(880, 452)
    c.title("Which kind of model is this?",
            "Four independent questions. People confuse them constantly.")

    pairs = [
        (40, "What stage?", [("Base", "completes text"), ("Instruct", "follows instructions"),
                             ("Reasoning", "thinks first")], "accent"),
        (250, "How available?", [("Open weights", "you run it"), ("API only", "they run it"),
                                 ("Fully open", "data too")], "good"),
        (460, "How big?", [("Small 0.5-8B", "your laptop"), ("Mid 8-70B", "a server"),
                           ("Large 100B+", "a cluster")], "warn"),
        (670, "How capable?", [("Foundation", "general purpose"), ("Frontier", "the current best"),
                               ("Specialist", "one domain")], "bad"),
    ]
    for x, heading, items, role in pairs:
        c.box(x, 92, 170, 190, "", role="ghost")
        c.text(x + 85, 116, heading, "lbl")
        for i, (name, sub) in enumerate(items):
            c.box(x + 14, 132 + i * 56, 142, 50, name, sub, role)

    c.box(40, 320, 390, 110, role="accent")
    c.text(60, 346, "Foundation vs frontier", "lbl", anchor="start")
    c.text(60, 370, "Foundation = trained broadly, meant to be", "small", anchor="start")
    c.text(60, 388, "adapted. A category. Qwen3-8B is one.", "small", anchor="start")
    c.text(60, 408, "Frontier = the most capable models that", "small", anchor="start")
    c.text(60, 424, "exist right now. A moving target, not a design.", "small", anchor="start")

    c.box(450, 320, 390, 110, role="warn")
    c.text(470, 346, "The mistake to avoid", "lbl", anchor="start")
    c.text(470, 370, "\"Frontier\" is not a quality tier you can buy", "small", anchor="start")
    c.text(470, 388, "into - today's frontier model is next year's", "small", anchor="start")
    c.text(470, 408, "ordinary one. Judge a model on YOUR task,", "small", anchor="start")
    c.text(470, 424, "not on which tier a press release claims.", "small", anchor="start")
    return "hb-model-landscape.svg", c.render(
        "Four independent axes: training stage, availability, size and capability "
        "tier, with foundation models being a category and frontier models a "
        "moving target."
    ), "Which kind of model is this?"


# ---------------------------------------------------------------------------
# Chapter 5 - the tool ecosystem
# ---------------------------------------------------------------------------


@diagram
def ecosystem_stack():
    c = Canvas(880, 500)
    c.title("The ecosystem, as a stack",
            "Every tool sits in one of these layers. Knowing the layer tells you what it replaces.")

    layers = [
        ("Your application", "Gradio · Streamlit · Open WebUI · your own code", "good"),
        ("Orchestration", "LangChain · LlamaIndex · smolagents · Qwen-Agent · DSPy · your own loop", "accent"),
        ("Protocol", "OpenAI API · MCP · tool-calling schemas", "accent"),
        ("Serving runtime", "vLLM · Ollama · llama.cpp · TGI · SGLang · LM Studio", "warn"),
        ("Model library", "transformers · PEFT · TRL · sentence-transformers", "warn"),
        ("Model weights", "Hugging Face · ModelScope · safetensors · GGUF", "default"),
        ("Numerics & hardware", "PyTorch · CUDA / Metal · bitsandbytes · Flash Attention", "muted"),
    ]
    y = 92
    for name, examples, role in layers:
        c.box(40, y, 250, 48, name, "", role)
        c.box(300, y, 540, 48, examples, "", "ghost")
        y += 54

    c.box(40, 474, 800, 24, "", role="accent")
    c.text(440, 492, "Tools in the SAME layer are alternatives. Tools in different "
                     "layers compose.", "small")
    return "hb-ecosystem-stack.svg", c.render(
        "A seven-layer stack from hardware and numerics up through model weights, "
        "libraries, serving runtimes, protocols, orchestration and the application."
    ), "The ecosystem as a stack"


@diagram
def mcp():
    c = Canvas(880, 452)
    c.title("MCP: why a protocol was needed",
            "The same problem USB solved for peripherals.")

    c.text(40, 90, "Before: every client needs custom code for every tool",
           "lbl t-bad", anchor="start")
    clients = ["App A", "App B", "App C"]
    tools = ["Files", "Database", "Calendar"]
    for i, name in enumerate(clients):
        c.box(40, 110 + i * 56, 110, 44, name, role="default")
    for i, name in enumerate(tools):
        c.box(310, 110 + i * 56, 110, 44, name, role="default")
    for i in range(3):
        for j in range(3):
            c.path(f"M 154 {132 + i * 56} L 306 {132 + j * 56}", arrow=False)
    c.text(230, 290, "3 x 3 = 9 integrations", "small t-bad")
    c.text(230, 310, "add one tool -> write 3 more", "small")

    c.text(470, 90, "After: one protocol in the middle", "lbl t-good", anchor="start")
    for i, name in enumerate(clients):
        c.box(470, 110 + i * 56, 110, 44, name, role="default")
    c.box(620, 138, 74, 100, "MCP", "", "good")
    for i, name in enumerate(tools):
        c.box(730, 110 + i * 56, 110, 44, name, role="default")
    for i in range(3):
        c.path(f"M 584 {132 + i * 56} L 616 180", arrow=False)
        c.path(f"M 698 188 L 726 {132 + i * 56}", arrow=False)
    c.text(655, 290, "3 + 3 = 6 integrations", "small t-good")
    c.text(655, 310, "add one tool -> write 1", "small")

    c.box(40, 336, 390, 100, role="accent")
    c.text(60, 360, "What an MCP server exposes", "lbl", anchor="start")
    for i, line in enumerate(["tools    - actions the model can call",
                              "resources - data it can read",
                              "prompts  - reusable templates"]):
        c.text(60, 384 + i * 20, line, "small", anchor="start")

    c.box(450, 336, 390, 100, role="default")
    c.text(470, 360, "Why you should care", "lbl", anchor="start")
    for i, line in enumerate(["Write your capstone's tools as an MCP",
                              "server and they work from any client -",
                              "an IDE, a desktop app, your own agent -",
                              "with no rewriting."]):
        c.text(470, 380 + i * 18, line, "small", anchor="start")
    return "hb-mcp.svg", c.render(
        "Without a protocol, three clients and three tools need nine custom "
        "integrations. MCP puts one protocol in the middle, reducing it to six, "
        "and each new tool needs only one."
    ), "MCP: why a protocol was needed"


@diagram
def choosing_tools():
    c = Canvas(880, 440)
    c.title("Choosing a tool: the questions that matter",
            "Most tool choices are reversible. Pick one and move.")

    questions = [
        ("What problem does it solve?", "If you cannot state it in one sentence, you do not need it yet.", "accent"),
        ("What did people do before?", "That tells you what it actually replaces.", "accent"),
        ("What does it lock in?", "A format, a schema, a hosting model - or nothing?", "warn"),
        ("Can I read the source?", "For a teaching project and for debugging, this matters more than features.", "good"),
        ("Is it maintained?", "Check the commit history, not the star count.", "good"),
    ]
    y = 92
    for question, why, role in questions:
        c.box(40, y, 330, 52, question, role=role)
        c.box(390, y, 450, 52, why, role="ghost")
        y += 62

    c.box(40, 408, 800, 28, "", role="default")
    c.text(440, 427, "Default to the smallest thing that works. You can always "
                     "add a framework; removing one is painful.", "small")
    return "hb-choosing-tools.svg", c.render(
        "Five questions for choosing a tool: what problem it solves, what it "
        "replaces, what it locks in, whether you can read the source, and "
        "whether it is maintained."
    ), "Choosing a tool"
