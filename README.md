# Open-Weight LLMs & Agentic AI — a hands-on workshop

**ROBT613** · Four days · Everything runs on your own machine.

Learn to run [Qwen](https://huggingface.co/Qwen) open-weight models locally,
give them tools and your own documents, build an agent, measure whether it
works, and ship something you actually use — with no API key, no subscription,
and no data leaving your laptop.

📖 **[Read the full course site →](https://berdakh.github.io/ROBT613/)**

---

<img src="docs/assets/diagrams/roadmap.svg" alt="The four-day roadmap" width="100%">

## What you will be able to do

By the end of four days:

1. **Explain** what a language model computes, and why it behaves as it does.
2. **Download and run** open-weight models on a laptop, GPU or Colab.
3. **Choose** the right model size, quantization and runtime for given hardware.
4. **Get structured, validated output** your program can rely on.
5. **Give the model tools** it can call — safely.
6. **Build retrieval** over your own notes, with citations and honest refusals.
7. **Write an agent loop** from scratch and debug it when it misbehaves.
8. **Fine-tune** with LoRA — and know when not to.
9. **Measure** all of the above instead of guessing.

## Who it is for

You need comfortable Python and a terminal. You do **not** need machine-learning
background, a GPU, or any paid service.

| You have | You can do |
|---|---|
| Any laptop, 8 GB RAM | Everything, with Qwen3-0.6B / 1.7B |
| 16 GB RAM or Apple Silicon | Everything, comfortably, with Qwen3-4B |
| An NVIDIA GPU | Everything, plus fine-tuning and vLLM |
| None of the above | Everything, in free Google Colab |

---

## Quick start

**Teaching this?** Send students
[docs/guides/before-you-start.md](docs/guides/before-you-start.md) a week ahead.
Setup failures are the biggest waste of day 1.

```bash
git clone https://github.com/berdakh/ROBT613.git
cd ROBT613

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt                # pinned; use for teaching

python scripts/check_env.py                 # what can this machine run?
python scripts/download_models.py --set core   # ~3 GB, do this on good wifi
jupyter lab notebooks/01_llm_foundations.ipynb
```

**No install at all?** Click any *Colab* badge in the table below — the first
cell clones the repo and installs everything for you.
See [docs/guides/colab.md](docs/guides/colab.md).

Full local setup, including Windows: **[docs/guides/setup.md](docs/guides/setup.md)**

---

## The notebooks

**You can read every notebook on the [course site](https://berdakh.github.io/ROBT613/notebooks/)
and open it in Colab with one click** — no clone, no install, no GPU to begin.


| # | Notebook | You learn to | Time | Colab |
|---|---|---|---|---|
| **Day 1 — Foundations** |||||
| 01 | [LLM foundations](notebooks/01_llm_foundations.ipynb) | Tokenization; what a model really computes; open weights vs. open source | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/01_llm_foundations.ipynb) |
| 02 | [Environment & hardware](notebooks/02_environment_and_hardware.ipynb) | Compute memory needs; choose a model; control your cache | 45 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/02_environment_and_hardware.ipynb) |
| 03 | [First generation](notebooks/03_first_generation.ipynb) | Load Qwen3, generate, stream, converse, measure tok/s | 75 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/03_first_generation.ipynb) |
| 04 | [Decoding & prompting](notebooks/04_decoding_and_prompting.ipynb) | Temperature, top-p, thinking mode, prompt patterns that work | 75 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/04_decoding_and_prompting.ipynb) |
| **Day 2 — Making it practical** |||||
| 05 | [Quantization & runtimes](notebooks/05_quantization_and_local_runtimes.ipynb) | GGUF, Ollama, llama.cpp, bitsandbytes; measure the quality cost | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/05_quantization_and_local_runtimes.ipynb) |
| 06 | [Serving & the OpenAI API](notebooks/06_serving_and_openai_api.ipynb) | Run a local server; write portable client code | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/06_serving_and_openai_api.ipynb) |
| 07 | [Structured output & tools](notebooks/07_structured_output_and_tools.ipynb) | Reliable JSON, Pydantic validation, tool calling, tool security | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/07_structured_output_and_tools.ipynb) |
| **Day 3 — Knowledge & agents** |||||
| 08 | [Embeddings & search](notebooks/08_embeddings_and_search.ipynb) | Semantic search, hybrid search, and when embeddings fail | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/08_embeddings_and_search.ipynb) |
| 09 | [RAG over your notes](notebooks/09_rag_over_your_notes.ipynb) | Full RAG pipeline with citations, refusals and **evaluation** | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/09_rag_over_your_notes.ipynb) |
| 10 | [Agents from scratch](notebooks/10_agents_from_scratch.ipynb) | Write the agent loop; test it with no model; six failure modes; injection | 120 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/10_agents_from_scratch.ipynb) |
| **Day 4 — Adapt & ship** |||||
| 11 | [Fine-tuning with LoRA](notebooks/11_finetuning_lora.ipynb) | Train an adapter — and decide whether you should | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/11_finetuning_lora.ipynb) |
| 12 | [Capstone](notebooks/12_capstone_daily_assistant.ipynb) | Build, evaluate and present a daily-life assistant | rest | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/12_capstone_daily_assistant.ipynb) |

| **Optional track — how training actually works** |||||
| 13 | [Pre-train a tiny model](notebooks/13_pretrain_a_tiny_model.ipynb) | Write a transformer and train it from scratch on CPU | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/13_pretrain_a_tiny_model.ipynb) |
| 14 | [Building a training dataset](notebooks/14_building_a_training_dataset.ipynb) | Loss masking, dedup, leak-free splits, validation | 75 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/14_building_a_training_dataset.ipynb) |
| 15 | [Preference tuning with DPO](notebooks/15_preference_tuning_dpo.ipynb) | Implement the DPO loss; build pairs; run `DPOTrainer` | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/15_preference_tuning_dpo.ipynb) |
| 16 | [Training diagnostics](notebooks/16_training_diagnostics.ipynb) | Read loss curves, find the LR, know when to stop | 75 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/16_training_diagnostics.ipynb) |
| **Optional track — agents in depth** |||||
| 17 | [Tools and memory](notebooks/17_tools_and_memory.ipynb) | Sandboxed tools; four memory strategies; the context budget | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/17_tools_and_memory.ipynb) |
| 18 | [Planning and decomposition](notebooks/18_planning_and_decomposition.ipynb) | Plans as graphs, waves, blocking, replanning | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/18_planning_and_decomposition.ipynb) |
| 19 | [Multi-agent systems](notebooks/19_multi_agent_systems.ipynb) | Routers, orchestrators, critic loops — and when not to | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/19_multi_agent_systems.ipynb) |
| 20 | [Evaluating agents](notebooks/20_evaluating_agents.ipynb) | Outcome vs trajectory; lucky passes; why the harness moves the score; a CI suite | 2 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/20_evaluating_agents.ipynb) |
| **Optional — the mathematics** |||||
| 21 | [Attention from scratch](notebooks/21_attention_from_scratch.ipynb) | Derive attention, prove the causal mask, implement RoPE | 2 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/21_attention_from_scratch.ipynb) |
| **Optional — building in Kazakh or Russian** |||||
| 22 | [Kazakh and Russian](notebooks/22_kazakh_and_russian.ipynb) | Measure the tokenizer tax; build a tokenizer to see why | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/22_kazakh_and_russian.ipynb) |

Short on time? Priority order: **01, 03, 07, 09, 10** are the essential five.

**Notebooks 13–16 are an optional fifth day.** They are where chapter 3 of
the handbook stops being reading and starts being code: you pre-train a
language model from scratch, prepare a dataset properly, implement the DPO
loss yourself, and learn to read a training run. Only 15's final cell needs a
GPU — everything else runs on a laptop in seconds. See
[the training track](docs/days/training-track.md).

**Notebooks 17–20 are an optional agents track.** Notebook 10 builds an
agent; this track gives it memory, plans and teammates, and then teaches you
to prove it works. **No server, no GPU, no downloads** — every notebook runs
against a scripted model, which is also how you test a real agent. See
[the agents track](docs/days/agents-track.md).

## The handbook — background reading

The notebooks teach you to build. The **[handbook](docs/handbook/)** explains
what you are building on. It is self-contained: no code to run, readable in an
evening, and it answers the questions students actually ask.

| | Chapter | About |
|---|---|---|
| 1 | [How we got here](docs/handbook/how-we-got-here.md) | Why each piece was invented, in order |
| 2 | [How models work](docs/handbook/how-models-work.md) | Tokens, embeddings, attention, context windows |
| 3 | [How LLMs are trained](docs/handbook/how-llms-are-trained.md) | Data, pre-training, post-training, RLHF/DPO, reasoning, and your part |
| 4 | [The model landscape](docs/handbook/the-model-landscape.md) | Base vs instruct vs reasoning · foundation vs frontier · reading a model card |
| 5 | [The tool ecosystem](docs/handbook/the-tool-ecosystem.md) | Every tool worth knowing, the problem it solves, and when not to use it |
| 6 | [Protocols and standards](docs/handbook/protocols-and-standards.md) | OpenAI API, tool-calling schemas, and MCP in depth |
| 7 | [The mathematics](docs/handbook/the-mathematics.md) | **The deep dive.** Attention derived, RoPE, the residual stream, the costs |

## Deep-dive guides

| Guide | About |
|---|---|
| [Open-weight models](docs/guides/open-weight-models.md) | What "open weights" means, licences, formats, quantization, sizing |
| [The Qwen family](docs/guides/qwen-family.md) | Which model for which job |
| [**Agentic AI**](docs/guides/agentic-ai.md) | The long version: loops, tools, memory, planning, multi-agent, MCP, failure modes, evaluation |
| [Privacy and safety](docs/guides/privacy-and-safety.md) | Using your own data responsibly |
| [Deployment](docs/guides/deployment.md) | From notebook to service |
| [Capstone ideas](docs/guides/capstone-ideas.md) | Seven projects, scoped for one afternoon |
| [Before you start](docs/guides/before-you-start.md) | The day-0 checklist to send students |
| [Capstone rubric](docs/guides/capstone-rubric.md) | How the capstone is marked, 100 points |
| [Solutions](solutions/) | Exercise and checkpoint answers (instructors) |
| [Troubleshooting](docs/guides/troubleshooting.md) | When it breaks |
| [Cheat sheet](docs/guides/cheatsheet.md) | The snippets you will reuse |
| [Glossary](docs/guides/glossary.md) | Every term, defined plainly |
| [Diagram index](docs/guides/diagrams.md) | All 31 figures in one place |

---

## What is in this repository

```
notebooks/        22 teaching notebooks (12 core + training, agents, maths, language)
notebook_src/     their source (percent format) - edit these, not the .ipynb
docs/handbook/    8-chapter background handbook (no code required)
src/qwen_workshop/
    config.py     model catalogue and sampling presets
    env.py        hardware and package detection
    loading.py    load models with sensible defaults and readable errors
    chat.py       chat templates, generation, streaming, thinking mode
    client.py     OpenAI-compatible client for Ollama / vLLM / llama.cpp
    tools.py      turn Python functions into tools an LLM can call, safely
    agent.py      the agent loop (~200 lines, heavily commented)
    scripted.py   a scripted stand-in for the model, so agents run in tests
    memory.py     conversation buffer, window, summariser, fact store
    planning.py   plans as dependency graphs, with replanning
    multiagent.py routers, orchestrators, critic loops
    agenteval.py  outcome + trajectory evaluation for agents
    rag.py        chunking, embeddings, vector index, RAG prompts
    demo_tools.py ready-made tools for the agent notebooks
    parsing.py    tolerant JSON extraction from chatty model output
data/             sample notes, pantry, expenses, inbox, eval questions
scripts/          check_env, download_models, smoke_test
solutions/        exercise + checkpoint answers for instructors
tests/            294 tests that run without a model, GPU or network
docs/             the GitHub Pages site
    assets/diagrams/  46 generated SVG figures, used by notebooks and handbook
tools/
    nbbuild.py        notebook_src/*.py  ->  notebooks/*.ipynb
    nbpages.py        notebooks/*.ipynb  ->  docs/notebooks/*.md (the site)
    make_diagrams.py  workshop diagrams; diagrams_handbook.py and
                      diagrams_training.py
                      and diagrams_agents.py add the handbook's and the tracks'
                      (svgkit.py is the shared drawing kit)
    check_notebooks.py, check_links.py   validation run in CI
```

The library is small and meant to be **read**, not just imported. Every module
is commented for a student who is seeing this for the first time.

---

## Design principles

**Everything is measurable.** Every technique is paired with a way to check
whether it worked — recall@k for retrieval, test cases for agents, before/after
for fine-tuning. "It looks good" is not an evaluation.

**Nothing is magic.** You write the agent loop yourself before using a
framework. You implement quantization in NumPy before using `bitsandbytes`. You
see the raw `<tool_call>` text before using a parser.

**Everything is generated and checked.** Notebooks come from `notebook_src/*.py`
and the 46 figures come from `tools/make_diagrams.py`, so both stay reviewable
in a diff. CI fails if either drifts, if a code cell stops parsing, or if any
link or image path breaks.

**Security is taught, not bolted on.** No `eval()` on model output — the
`calculate` tool parses an AST. File tools check path containment. Irreversible
actions need confirmation. There is a live prompt-injection demo in notebook 10.

**Honest about limits.** A 0.6B model will get things wrong. The workshop says
so, shows where, and teaches you to close the gap with retrieval and tools
rather than pretending the gap is not there.

---

## For instructors

Worked answers to all 62 exercises and 129 checkpoint questions are in
[`solutions/`](solutions/) — with confidence markers, since the ones that depend
on a real run are estimates until you teach it once.

```bash
make install          # latest compatible versions
make install-locked   # the exact versions you teach with
make lock             # regenerate the lock file from a verified environment
make test             # 294 tests, no model or network needed
make check       # tests + notebooks built and valid
make lint        # ruff
make notebooks   # rebuild .ipynb from notebook_src/
make diagrams    # regenerate docs/assets/diagrams/*.svg
```

Notebooks are **generated** from `notebook_src/*.py` (jupytext percent format)
so that pull requests show readable diffs instead of JSON blobs. Edit the `.py`
file, run `make notebooks`, commit both.

Diagrams work the same way: edit `tools/make_diagrams.py`, run `make diagrams`,
commit both. They are self-authored SVG rather than downloaded images, so there
is no licensing question when you redistribute the course, nothing to break when
a link rots, and they still render with no network — which matters, since the
whole workshop is meant to run offline.

CI runs the tests and verifies the notebooks are in sync on every push.

### Publishing the site

The site lives in `docs/`. A repository admin must switch Pages on **once**:

**Settings → Pages → Source: "GitHub Actions"**

After that, `.github/workflows/pages.yml` publishes it on every push to
`master` that touches `docs/`.

This one step cannot be automated: creating a Pages site requires admin
permission, and a workflow's `GITHUB_TOKEN` cannot be granted it — `pages: write`
covers deploying to an existing site, not creating one. Until it is switched on,
the deploy workflow fails at `configure-pages` with
`Resource not accessible by integration`.

### Adapting it

- Swap `data/notes/` for material from your own course — the RAG notebooks get
  much better when the corpus is something students care about.
- Stand up a shared endpoint as insurance and give students one line:
  `export WORKSHOP_LAB_URL=<url>`. Any notebook then works with
  `BACKEND = "lab"`, so a broken laptop costs one student a setup, not three days.
- The workshop is Qwen-based but the concepts are not. Changing `DEFAULT_CHAT_MODEL`
  in `src/qwen_workshop/config.py` is most of the work needed to use Llama or Mistral.
- Each day is independent enough to run as a standalone 6-hour session.

---

## Licence

Code is [MIT](LICENSE). Prose and teaching materials are CC BY 4.0 — use them in
your own course, with attribution.

**Model weights are governed by their own licences.** The Qwen3 main line is
Apache 2.0, but always check the model card for the specific checkpoint you
download. See [Open-weight models](docs/guides/open-weight-models.md#3-licences-read-them).

## Acknowledgements

Built on the work of the [Qwen team](https://huggingface.co/Qwen) at Alibaba
Cloud, and the open-source ecosystem around
[transformers](https://github.com/huggingface/transformers),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[Ollama](https://ollama.com) and [vLLM](https://github.com/vllm-project/vllm).
