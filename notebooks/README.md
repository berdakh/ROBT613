# Notebooks

Work through these in order. Each builds on the last.

| # | Notebook | Needs | Time | Colab |
|---|---|---|---|---|
| 01 | `01_llm_foundations.ipynb` | tokenizer only (~10 MB) | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/01_llm_foundations.ipynb) |
| 02 | `02_environment_and_hardware.ipynb` | nothing | 45 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/02_environment_and_hardware.ipynb) |
| 03 | `03_first_generation.ipynb` | Qwen3-0.6B | 75 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/03_first_generation.ipynb) |
| 04 | `04_decoding_and_prompting.ipynb` | Qwen3-0.6B | 75 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/04_decoding_and_prompting.ipynb) |
| 05 | `05_quantization_and_local_runtimes.ipynb` | Ollama (optional) | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/05_quantization_and_local_runtimes.ipynb) |
| 06 | `06_serving_and_openai_api.ipynb` | **a running server** | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/06_serving_and_openai_api.ipynb) |
| 07 | `07_structured_output_and_tools.ipynb` | a running server | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/07_structured_output_and_tools.ipynb) |
| 08 | `08_embeddings_and_search.ipynb` | Qwen3-Embedding-0.6B | 60 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/08_embeddings_and_search.ipynb) |
| 09 | `09_rag_over_your_notes.ipynb` | embedding model + server | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/09_rag_over_your_notes.ipynb) |
| 10 | `10_agents_from_scratch.ipynb` | a running server | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/10_agents_from_scratch.ipynb) |
| 11 | `11_finetuning_lora.ipynb` | GPU recommended | 90 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/11_finetuning_lora.ipynb) |
| 12 | `12_capstone_daily_assistant.ipynb` | a running server | rest of day 4 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/12_capstone_daily_assistant.ipynb) |

"A running server" means Ollama (or vLLM / llama.cpp). Start it with:

```bash
ollama serve && ollama pull qwen3:0.6b
```

## Before you start

Every notebook begins with a cell that puts `src/` on the path. **Run it.**
Skipping it is the single most common cause of `ModuleNotFoundError`.

## The figures

Each notebook embeds SVG diagrams from `../docs/assets/diagrams/`. They render
in JupyterLab, on GitHub and on the course site, and adapt to light or dark
themes. All 18 are collected on one page:
[`docs/guides/diagrams.md`](../docs/guides/diagrams.md).

## Editing these notebooks

Don't — edit `notebook_src/*.py` instead and run `make notebooks`. The `.ipynb`
files are generated, so that pull requests show readable diffs rather than JSON.

## If a cell fails

See [`../docs/guides/troubleshooting.md`](../docs/guides/troubleshooting.md).
