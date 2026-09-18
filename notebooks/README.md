# Notebooks

Work through these in order. Each builds on the last.

| # | Notebook | Needs | Time |
|---|---|---|---|
| 01 | `01_llm_foundations.ipynb` | tokenizer only (~10 MB) | 60 min |
| 02 | `02_environment_and_hardware.ipynb` | nothing | 45 min |
| 03 | `03_first_generation.ipynb` | Qwen3-0.6B | 75 min |
| 04 | `04_decoding_and_prompting.ipynb` | Qwen3-0.6B | 75 min |
| 05 | `05_quantization_and_local_runtimes.ipynb` | Ollama (optional) | 60 min |
| 06 | `06_serving_and_openai_api.ipynb` | **a running server** | 60 min |
| 07 | `07_structured_output_and_tools.ipynb` | a running server | 90 min |
| 08 | `08_embeddings_and_search.ipynb` | Qwen3-Embedding-0.6B | 60 min |
| 09 | `09_rag_over_your_notes.ipynb` | embedding model + server | 90 min |
| 10 | `10_agents_from_scratch.ipynb` | a running server | 90 min |
| 11 | `11_finetuning_lora.ipynb` | GPU recommended | 90 min |
| 12 | `12_capstone_daily_assistant.ipynb` | a running server | rest of day 4 |

"A running server" means Ollama (or vLLM / llama.cpp). Start it with:

```bash
ollama serve && ollama pull qwen3:0.6b
```

## Before you start

Every notebook begins with a cell that puts `src/` on the path. **Run it.**
Skipping it is the single most common cause of `ModuleNotFoundError`.

## Editing these notebooks

Don't — edit `notebook_src/*.py` instead and run `make notebooks`. The `.ipynb`
files are generated, so that pull requests show readable diffs rather than JSON.

## If a cell fails

See [`../docs/guides/troubleshooting.md`](../docs/guides/troubleshooting.md).
