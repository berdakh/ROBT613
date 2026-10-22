---
title: Diagram index
parent: Guides
nav_order: 12
---

# Diagram index
{: .no_toc }

Every diagram in the workshop and the handbook, in one place — useful for
lecturing from, or for revision.

All of them are **generated**, not drawn by hand: `tools/make_diagrams.py`
builds them with the small kit in `tools/svgkit.py`. Change the generator and
run `make diagrams`; CI fails if a committed SVG has drifted from its source.

They are plain SVG, so they scale to any projector, work offline, and adapt to
light and dark themes. Reuse them in your own slides — code is MIT, prose and
figures are CC BY 4.0.

1. TOC
{:toc}

---

## Foundations

### The next-token generation loop
Notebook 01. The one idea everything rests on.

<img src="../assets/diagrams/next-token-loop.svg" alt="Text is tokenized, the model produces a probability for every token, one is sampled, appended, and the whole sequence runs through again" width="100%">

### Tokenization
Notebook 01. Why "how many r's in strawberry" is hard, and why a tool fixes it.

<img src="../assets/diagrams/tokenization.svg" alt="The word strawberry splits into three tokens, so the letters are invisible to the model" width="100%">

### Open weights vs. open source
Notebook 01, and the [open-weight models guide](open-weight-models.html).

<img src="../assets/diagrams/open-weights.svg" alt="Open weights gives parameters and inference code but not training data or reproducibility" width="100%">

---

## Hardware and efficiency

### Where the memory goes
Notebook 02. Weights dominate — until the context gets long.

<img src="../assets/diagrams/memory-budget.svg" alt="Memory per model and precision, showing weights against KV cache" width="100%">

### How 4-bit quantization works
Notebook 05.

<img src="../assets/diagrams/quantization.svg" alt="A group of weights stored as 4-bit integers plus one shared scale" width="100%">

### How LoRA works
Notebook 11.

<img src="../assets/diagrams/lora.svg" alt="LoRA freezes the original weight matrix and learns two thin matrices beside it" width="100%">

---

## Generating text

### The four steps from messages to text
Notebook 03. Step 4 is the one everybody forgets.

<img src="../assets/diagrams/chat-pipeline.svg" alt="Messages, chat template, generate, then slice off the prompt before decoding" width="100%">

### Sampling
Notebook 04. Why top-p usually beats top-k.

<img src="../assets/diagrams/sampling.svg" alt="Temperature reshapes the odds, top-k and top-p cut the tail, then one token is picked at random" width="100%">

### Thinking mode
Notebook 04.

<img src="../assets/diagrams/thinking-mode.svg" alt="A hidden reasoning block sits between the question and the visible answer" width="100%">

---

## Building systems

### One API, four runtimes
Notebook 06, and the [deployment guide](deployment.html).

<img src="../assets/diagrams/serving.svg" alt="One OpenAI-compatible API served interchangeably by Ollama, vLLM, llama.cpp or LM Studio" width="100%">

### The tool-calling round trip
Notebook 07. The model never executes anything.

<img src="../assets/diagrams/tool-calling.svg" alt="The model requests a call, your code executes it, the result returns, the model answers" width="100%">

### Embeddings place meaning in space
Notebook 08.

<img src="../assets/diagrams/embedding-space.svg" alt="Paraphrases and translations cluster together; embeddings fail on exact identifiers, negation and numbers" width="100%">

### The four stages of RAG
Notebook 09. Measure retrieval and answers separately.

<img src="../assets/diagrams/rag-pipeline.svg" alt="Chunk, embed, retrieve, generate, with retrieval and answer quality measured apart" width="100%">

---

## Agents

### The agent loop
Notebook 10, and the [agentic AI guide](agentic-ai.html). This is the whole idea.

<img src="../assets/diagrams/agent-loop.svg" alt="THINK either answers or calls a tool; ACT runs it, OBSERVE appends the result, and control returns to THINK" width="100%">

### The chain-to-agent spectrum
Notebook 10. Most production "agents" should have been chains.

<img src="../assets/diagrams/agent-spectrum.svg" alt="A spectrum from prompt and chain through router, tool loop and planner to multi-agent" width="100%">

### Prompt injection
Notebook 10, and [privacy and safety](privacy-and-safety.html).

<img src="../assets/diagrams/prompt-injection.svg" alt="Hostile instructions inside a document reach the agent; defences are least privilege, confirmation gates, trust separation and logging" width="100%">

---

## Planning your work

### Prompt, RAG, tools, or fine-tune?
Notebook 11. Work down the list and stop at the first yes.

<img src="../assets/diagrams/decision-tree.svg" alt="Try prompting, then RAG, then structured output, then tools, and only then fine-tuning" width="100%">

### The four-day roadmap

<img src="../assets/diagrams/roadmap.svg" alt="Day 1 foundations, day 2 practical, day 3 knowledge and agents, day 4 ship" width="100%">

---

## Handbook figures

These accompany [the handbook](../handbook/), which is background reading rather
than hands-on work.

### How we got here
[Chapter 1](../handbook/how-we-got-here.html). Why each piece was invented.

<img src="../assets/diagrams/hb-timeline.svg" alt="A timeline from the 2017 transformer to reasoning models and agents" width="100%">

### What attention does
[Chapter 2](../handbook/how-models-work.html).

<img src="../assets/diagrams/hb-attention.svg" alt="The word 'it' attends mostly to 'laptop', resolving the pronoun" width="100%">

### Inside a transformer
[Chapter 2](../handbook/how-models-work.html).

<img src="../assets/diagrams/hb-transformer-block.svg" alt="Embeddings, N blocks of attention and feed-forward, then a final layer and softmax" width="100%">

### The full training lifecycle
[Chapter 3](../handbook/how-llms-are-trained.html). The single most useful figure
in the handbook.

<img src="../assets/diagrams/hb-training-lifecycle.svg" alt="Four phases - data, pre-training, post-training, adaptation - with falling costs" width="100%">

### Phase 0: making the data
[Chapter 3](../handbook/how-llms-are-trained.html).

<img src="../assets/diagrams/hb-pretraining-data.svg" alt="Crawl, filter, deduplicate, decontaminate, tokenize" width="100%">

### Scaling laws
[Chapter 3](../handbook/how-llms-are-trained.html). Why good small models exist.

<img src="../assets/diagrams/hb-scaling-laws.svg" alt="Tokens per parameter from GPT-3 through Chinchilla to modern overtrained small models" width="100%">

### Phase 2: post-training
[Chapter 3](../handbook/how-llms-are-trained.html). SFT, RLHF and DPO.

<img src="../assets/diagrams/hb-post-training.svg" alt="Base model to instruct model via supervised fine-tuning and preference optimisation" width="100%">

### Adaptation options, cheapest first
[Chapter 3](../handbook/how-llms-are-trained.html).

<img src="../assets/diagrams/hb-adaptation-ladder.svg" alt="A ladder from prompting through RAG and tools to fine-tuning and pre-training" width="100%">

### Distillation and Mixture of Experts
[Chapter 3](../handbook/how-llms-are-trained.html).

<img src="../assets/diagrams/hb-distillation-moe.svg" alt="A small student trained on a teacher's outputs; a router sending tokens to a couple of experts" width="100%">

### Which kind of model is this?
[Chapter 4](../handbook/the-model-landscape.html). Foundation vs frontier, and
the axes people conflate.

<img src="../assets/diagrams/hb-model-landscape.svg" alt="Four axes: training stage, availability, size and capability tier" width="100%">

### The ecosystem as a stack
[Chapter 5](../handbook/the-tool-ecosystem.html). Tools in the same layer are
alternatives; tools in different layers compose.

<img src="../assets/diagrams/hb-ecosystem-stack.svg" alt="Seven layers from hardware up to the application" width="100%">

### Choosing a tool
[Chapter 5](../handbook/the-tool-ecosystem.html).

<img src="../assets/diagrams/hb-choosing-tools.svg" alt="Five questions for evaluating a tool" width="100%">

### MCP: why a protocol was needed
[Chapter 6](../handbook/protocols-and-standards.html).

<img src="../assets/diagrams/hb-mcp.svg" alt="Nine custom integrations become six once a protocol sits in the middle" width="100%">

---

## The training track

### One training step
Notebook 13. Every trainer you will ever use is this loop with more bookkeeping,
and it reports exactly four numbers.

<img src="../assets/diagrams/tr-training-loop.svg" alt="A batch goes through a forward pass, a loss, a backward pass and a weight update, then the loop repeats; clipping and the schedule modify the update" width="100%">

### Loss masking
Notebook 14. Which tokens the model is graded on, and which it only reads.

<img src="../assets/diagrams/tr-loss-masking.svg" alt="System and user segments are labelled -100 and contribute no loss; only the assistant segment is graded" width="100%">

### RLHF versus DPO
Notebook 15, and [chapter 3](../handbook/how-llms-are-trained.html). Same data;
DPO removes the reward model and the RL loop.

<img src="../assets/diagrams/dpo-vs-rlhf.svg" alt="RLHF trains a reward model then loops sample-score-update with PPO; DPO scores each pair with the policy and a frozen reference and optimises one logistic loss" width="100%">

### Five loss curves, five decisions
Notebook 16. In three of the five, the train curve alone looks acceptable.

<img src="../assets/diagrams/loss-curve-patterns.svg" alt="Healthy, undertrained, overfitting, stuck at chance and diverged loss curves, with the decision each one implies" width="100%">


---

## The agents track

### Four ways to remember
Notebook 17. The model has no memory; these are four ways your code fakes it.

<img src="../assets/diagrams/ag-memory-strategies.svg" alt="Four memory strategies as rows, with what each one loses and whether it survives a restart" width="100%">

### A plan is a graph, not a list
Notebook 18. Which is why one failure blocks its descendants and nothing else.

<img src="../assets/diagrams/ag-plan-graph.svg" alt="A plan in three waves; a failed search blocks its descendants while the parallel branch completes" width="100%">

### Four ways to wire several agents
Notebook 19. Costs rise left to right. Start at the left.

<img src="../assets/diagrams/ag-team-patterns.svg" alt="Router, orchestrator, pipeline and critic loop side by side with the model-call cost of each" width="100%">

### Outcome is half the picture
Notebook 20. The dangerous quadrant is the one an outcome-only suite calls a pass.

<img src="../assets/diagrams/ag-eval-quadrants.svg" alt="A two-by-two of answer-right against path-sane; right answer with a wrong path is the lucky pass" width="100%">

---

## The mathematics

### One attention operation, end to end
Notebook 21 and [chapter 7](../handbook/the-mathematics.html). Every arrow is a
matrix multiply or an elementwise step.

<img src="../assets/diagrams/mt-attention-pipeline.svg" alt="X projects to Q, K and V; Q times K transpose gives scores, which are scaled, masked and softmaxed into weights that multiply V" width="100%">

### RoPE: position as rotation
Why the score ends up depending on the gap between tokens, not on where they are.

<img src="../assets/diagrams/mt-rope.svg" alt="A vector rotated further at each successive position, with its length unchanged, so the inner product depends only on the difference of positions" width="100%">

### A pre-norm block
The residual stream runs straight down; everything else is a correction to it.

<img src="../assets/diagrams/mt-block-anatomy.svg" alt="A transformer block as a vertical residual stream, with RMSNorm plus attention and RMSNorm plus the SwiGLU MLP each adding a correction back" width="100%">
