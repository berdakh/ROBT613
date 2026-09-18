---
title: Cheat sheet
parent: Guides
nav_order: 9
---

# Cheat sheet
{: .no_toc }

The commands and snippets you will actually reuse.

1. TOC
{:toc}

---

## Environment

```bash
python scripts/check_env.py          # hardware + packages report
python scripts/download_models.py --set core
python scripts/smoke_test.py         # end-to-end check
python -m pytest -q                  # the workshop's tests

export HF_HOME=/path/with/space      # move the model cache
export HF_ENDPOINT=https://hf-mirror.com   # use a mirror
export HF_HUB_OFFLINE=1              # work entirely offline

huggingface-cli scan-cache           # what is using disk
huggingface-cli delete-cache         # clean up
```

## Ollama

```bash
ollama pull qwen3:4b
ollama run qwen3:4b "your prompt" --verbose
ollama list                          # installed models
ollama ps                            # currently loaded
ollama rm qwen3:4b                   # free the disk
ollama serve                         # start the server
```

## llama.cpp

```bash
llama-cli   -hf Qwen/Qwen3-4B-GGUF:Q4_K_M            # interactive
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 8192 --jinja --port 8080
```

| Flag | Meaning |
|---|---|
| `-c 8192` | context length |
| `-ngl 99` | GPU layers (`0` = pure CPU) |
| `-t 8` | CPU threads |
| `--jinja` | use the model's chat template (needed for tools) |

## vLLM

```bash
vllm serve Qwen/Qwen3-8B \
  --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

---

## transformers

### Load and generate

```python
from qwen_workshop.loading import load_model
from qwen_workshop.chat import chat

qwen = load_model("Qwen/Qwen3-0.6B")             # add quantize_4bit=True on CUDA
result = chat(qwen, "Explain tokens in one sentence.")
print(result.text, result.tokens_per_second)
```

### Raw, without helpers

```python
prompt = tok.apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
inputs = tok([prompt], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                     top_p=0.8, top_k=20, do_sample=True)
text = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
```

{: .tip }
> The slice `out[0][inputs["input_ids"].shape[-1]:]` is the step everyone
> forgets. Without it you echo the prompt back.

### Streaming

```python
from qwen_workshop.chat import stream_chat, print_stream
print_stream(stream_chat(qwen, "Write a haiku."))
```

---

## Sampling presets

| Task | Settings |
|---|---|
| Extraction, JSON, classification | `do_sample=False` |
| Factual Q&A, RAG | `temperature=0.3, top_p=0.8` |
| Chat (Qwen3 non-thinking) | `temperature=0.7, top_p=0.8, top_k=20` |
| Reasoning (Qwen3 thinking) | `temperature=0.6, top_p=0.95, top_k=20` |
| Brainstorming | `temperature=1.0, top_p=0.95` |

```python
from qwen_workshop.config import THINKING_SAMPLING, NON_THINKING_SAMPLING, GREEDY_SAMPLING
chat(qwen, "...", sampling=GREEDY_SAMPLING.replace(max_new_tokens=128))
```

---

## OpenAI-compatible API

```python
from qwen_workshop.client import get_client, complete, is_up

if is_up("ollama"):
    client, backend = get_client("ollama", model="qwen3:4b")
    reply = complete(client, backend,
                     [{"role": "user", "content": "hello"}],
                     temperature=0.7, max_tokens=200, enable_thinking=False)
    print(reply.choices[0].message.content)
```

Always check `reply.choices[0].finish_reason` — `"length"` means truncated.

---

## Tools

```python
from qwen_workshop.tools import tool, ToolRegistry, ToolError
from typing import Literal

@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get today's weather for a city.

    Args:
        city: City name, e.g. "Astana".
        unit: Temperature unit.
    """
    if not city:
        raise ToolError("city must not be empty")
    return {"city": city, "temp": 12, "unit": unit}

registry = ToolRegistry().add(get_weather)
registry.schemas()                       # pass as tools=... to the API
registry.call("get_weather", {"city": "Astana"})   # never raises
```

## Agent

```python
from qwen_workshop.agent import Agent

agent = Agent(client, backend, registry, max_steps=6)
run = agent.run("What is the weather in Astana?")
print(run.answer)
print(run.trace())        # every step and observation - use this to debug
```

---

## RAG

```python
from qwen_workshop.rag import Embedder, VectorIndex, load_corpus, build_rag_prompt, cited_sources

chunks = load_corpus("data/notes", max_chars=700, overlap=120)
embedder = Embedder("Qwen/Qwen3-Embedding-0.6B")
index = VectorIndex(chunks, embedder.encode_documents([c.text for c in chunks]))
index.save("data/.index/notes")          # do this once

hits = index.search(embedder.encode_queries(["when is my exam?"])[0], k=3)
hits = [(c, s) for c, s in hits if s >= 0.35]        # the threshold matters
prompt = build_rag_prompt("when is my exam?", hits)
```

Check for invented citations:

```python
retrieved = {c.citation for c, _ in hits}
invented = set(cited_sources(answer)) - retrieved     # should be empty
```

---

## LoRA fine-tuning

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"])
args = SFTConfig(output_dir="out", max_steps=60, learning_rate=2e-4,
                 per_device_train_batch_size=2, gradient_accumulation_steps=2,
                 max_length=512, bf16=True, report_to=[])
SFTTrainer(model="Qwen/Qwen3-0.6B", args=args,
           train_dataset=dataset, peft_config=lora).train()
```

Dataset format — one `messages` list per row, identical in shape to inference input:

```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

---

## Memory rules of thumb

| Precision | Bytes/param | 8B model |
|---|---|---|
| bf16 | 2 | 16 GB |
| int8 | 1 | 8 GB |
| int4 | ~0.55 | ~4.5 GB |

Add ~25% for the KV cache and overhead. Long contexts add much more.

## Repository tasks

```bash
make help            # all targets
make test            # pytest
make notebooks       # rebuild .ipynb from notebook_src/
make check           # tests + notebook validation
make clean           # remove caches and generated indexes
```
