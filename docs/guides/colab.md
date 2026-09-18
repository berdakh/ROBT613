---
title: Run in Colab
parent: Guides
nav_order: 2
---

# Run it all in Google Colab
{: .no_toc }

No GPU? No install? Every notebook runs on a free Colab runtime. Click a badge
and go.

1. TOC
{:toc}

---

## The notebooks

| Day | # | Notebook | Needs | In Colab | Open |
|:---|:---|:---|:---|:---|:---|
| 1 | 01 | LLM foundations | tokenizer only |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/01_llm_foundations.ipynb) |
| 1 | 02 | Environment & hardware | nothing |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/02_environment_and_hardware.ipynb) |
| 1 | 03 | First generation | Qwen3-0.6B | GPU helps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/03_first_generation.ipynb) |
| 1 | 04 | Decoding & prompting | Qwen3-0.6B | GPU helps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/04_decoding_and_prompting.ipynb) |
| 2 | 05 | Quantization & runtimes | optional Ollama |  | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/05_quantization_and_local_runtimes.ipynb) |
| 2 | 06 | Serving & the OpenAI API | a server | runs Ollama for you | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/06_serving_and_openai_api.ipynb) |
| 2 | 07 | Structured output & tools | a server | runs Ollama for you | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/07_structured_output_and_tools.ipynb) |
| 3 | 08 | Embeddings & search | embedding model | GPU helps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/08_embeddings_and_search.ipynb) |
| 3 | 09 | RAG over your notes | server + embeddings | runs Ollama for you | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/09_rag_over_your_notes.ipynb) |
| 3 | 10 | Agents from scratch | a server | runs Ollama for you | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/10_agents_from_scratch.ipynb) |
| 4 | 11 | Fine-tuning with LoRA | GPU | **use Colab** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/11_finetuning_lora.ipynb) |
| 4 | 12 | Capstone | a server | runs Ollama for you | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/12_capstone_daily_assistant.ipynb) |

## What happens when you click

Colab fetches **only the notebook file** from GitHub — not the repository. So
the first code cell in every notebook does the rest:

```python
if "google.colab" in sys.modules:
    # clone the repo so src/, data/ and the sample files exist
    # pip install -r requirements.txt
    # chdir into it
```

It is safe to re-run, and it does nothing on your own machine. That is why the
same notebook works in both places.

{: .note }
> The first run takes about a minute while it clones and installs. Subsequent
> cells are fast.

## Turn on the GPU

Colab gives you CPU by default, which makes notebooks 03, 04, 08 and 11 slow.

**Runtime → Change runtime type → T4 GPU**, then confirm with:

```python
!nvidia-smi
```

A free T4 has 16 GB of VRAM — enough for Qwen3-4B or Qwen3-8B in int4, and
enough to do the LoRA fine-tuning in notebook 11.

## Notebooks that need a server

Notebooks 06, 07, 09, 10 and 12 talk to a local OpenAI-compatible server.
Colab has none, so those notebooks include a cell that builds one for you:

```python
from qwen_workshop.colab import in_colab, start_ollama

if in_colab():
    start_ollama("qwen3:0.6b")
```

That installs Ollama, starts it in the background, and pulls the model — two to
four minutes the first time. On a CPU-only runtime stay with `qwen3:0.6b`.

## Things Colab will do to you

| Behaviour | What it means | What to do |
|---|---|---|
| Disconnects when idle | You lose the runtime and its disk | Re-run from the first cell; it re-clones |
| Wipes the disk each session | Models re-download every time | Expect a few minutes of setup per session |
| Free GPU quota is limited | You may get CPU-only at busy times | Use Qwen3-0.6B, or come back later |
| Edits are not saved to GitHub | Your changes vanish on disconnect | **File → Save a copy in Drive** before you start working |

{: .warning }
> **Save a copy in Drive first** if you intend to do the exercises. A Colab
> notebook opened from GitHub is read-only scratch space — closing the tab
> loses everything you typed.

## Using your own documents

Notebook 09 gets far more interesting with your own notes. In Colab:

```python
from google.colab import files
uploaded = files.upload()          # pick your .md or .txt files
```

Then move them into `data/notes/` and re-run the indexing cell. They are
deleted when the runtime ends, which is either a privacy feature or an
annoyance depending on your mood.

## Prefer to work locally?

Colab is the fallback, not the point. The workshop is about running models on
hardware you control — see [Setup](setup.html) for the local path, which is
faster after the first install and works offline.
