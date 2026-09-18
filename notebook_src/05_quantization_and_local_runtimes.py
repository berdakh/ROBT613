# %% [markdown]
# # 05 · Quantization and local runtimes
#
# **Day 2 · ~60 minutes**
#
# Quantization is how a 14B model fits on a gaming laptop. It stores each
# weight in fewer bits, trading a small amount of quality for a large amount of
# memory.
#
# By the end you will be able to:
#
# 1. explain what quantization does and what it costs;
# 2. read a GGUF quant name like `Q4_K_M` and choose sensibly;
# 3. run Qwen through Ollama and llama.cpp;
# 4. load a 4-bit model with bitsandbytes on a GPU;
# 5. pick the right runtime for a given job.
#
# **Much of this notebook runs shell commands.** If a runtime is not installed,
# the cell tells you what to install and moves on — nothing here is required
# for the later notebooks.

# %%
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def run(command: str, timeout: int = 300) -> str:
    """Run a shell command and return its output (never raises)."""
    try:
        done = subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as exc:  # noqa: BLE001
        return f"(failed: {exc})"


for binary in ("ollama", "llama-cli", "llama-server"):
    print(f"{binary:<14}{'found' if have(binary) else 'not installed'}")

# %% [markdown]
# <img src="../docs/assets/diagrams/quantization.svg" alt="Quantization stores a group of weights as small integers plus one shared scale" width="100%">

# %% [markdown]
# ## 5.1 · What quantization actually does
#
# A weight is a number like `0.4173829`. Stored as bf16 that costs 2 bytes.
#
# Quantization to 4 bits stores weights in **groups** (say, 32 at a time). For
# each group you keep:
#
# - a **scale** (and often a zero-point) at full precision,
# - each weight as a 4-bit integer index into that group's range.
#
# To use a weight you reconstruct `value ≈ scale × integer + zero`. The result
# is close to the original but not identical — that difference is the quality
# cost.
#
# Here is the idea in 15 lines of NumPy.

# %%
import numpy as np

rng = np.random.default_rng(613)
weights = rng.normal(0, 0.1, size=4096).astype(np.float32)


def quantize_group(values: np.ndarray, bits: int = 4, group_size: int = 32):
    """Symmetric group-wise quantization - the core of every 4-bit scheme."""
    groups = values.reshape(-1, group_size)
    max_int = 2 ** (bits - 1) - 1                      # 7 for 4-bit signed
    scales = np.abs(groups).max(axis=1, keepdims=True) / max_int
    scales = np.clip(scales, 1e-8, None)
    quantized = np.round(groups / scales).astype(np.int8)
    restored = (quantized * scales).reshape(-1)
    return quantized, scales, restored


for bits in (8, 4, 3, 2):
    _q, _s, restored = quantize_group(weights, bits=bits)
    error = np.abs(restored - weights).mean()
    bits_per_weight = bits + (16 / 32)  # payload + one fp16 scale per 32 weights
    print(f"{bits}-bit: mean abs error {error:.6f} "
          f"({error / np.abs(weights).mean():.2%} of average magnitude), "
          f"~{bits_per_weight:.2f} bits/weight")

# %% [markdown]
# Notice the cliff: 8-bit is nearly lossless, 4-bit is a reasonable trade,
# and 2-bit error explodes. That is exactly what people report about model
# quality at those precisions.
#
# > **Exercise 1.** Change `group_size` to 128 and to 16. Smaller groups mean
# > better accuracy but more scales to store. That trade-off is precisely what
# > the `_K_S` / `_K_M` / `_K_L` suffixes below encode.

# %% [markdown]
# ## 5.2 · Reading GGUF quant names
#
# **GGUF** is llama.cpp's single-file format — the standard for CPU and
# Apple Silicon inference. Filenames look like `Qwen3-4B-Q4_K_M.gguf`.
#
# | Name | Bits/weight | Size for 8B | Use it when |
# |---|---|---|---|
# | `Q8_0` | 8.5 | ~8.5 GB | you have room and want near-lossless |
# | `Q6_K` | 6.6 | ~6.6 GB | high quality, moderate size |
# | **`Q5_K_M`** | 5.7 | ~5.7 GB | better than Q4, if it fits |
# | **`Q4_K_M`** | 4.8 | ~4.9 GB | **the default recommendation** |
# | `Q4_K_S` | 4.6 | ~4.6 GB | slightly smaller, slightly worse |
# | `Q3_K_M` | 3.9 | ~4.0 GB | desperate for space |
# | `Q2_K` | 3.4 | ~3.4 GB | usually not worth it |
#
# - `K` = "k-quants", a smarter scheme that spends more bits on the layers that
#   matter most.
# - `S`/`M`/`L` = small/medium/large variants of that mix.
#
# **Rule of thumb: a larger model at Q4 beats a smaller model at Q8.**
# Qwen3-8B at Q4_K_M generally outperforms Qwen3-4B at Q8_0, at similar size.

# %% [markdown]
# ## 5.3 · Ollama — the easiest path
#
# Install from [ollama.com](https://ollama.com), then:
#
# ```bash
# ollama pull qwen3:0.6b     # ~500 MB
# ollama run qwen3:0.6b      # interactive chat
# ollama list                # what you have
# ollama ps                  # what is loaded right now
# ollama rm qwen3:0.6b       # free the disk
# ```
#
# Ollama picks a quant for you (usually Q4_K_M), manages the download, and
# exposes an OpenAI-compatible server on port 11434 automatically. For most
# students this is the shortest route from zero to a working local model.

# %%
if have("ollama"):
    print(run("ollama list"))
else:
    print("Ollama not installed.")
    print("  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
    print("  Windows    : download the installer from ollama.com")
    print("\nThen:  ollama pull qwen3:0.6b")

# %%
# Pull the small model (uncomment; ~500 MB).
# if have("ollama"):
#     print(run("ollama pull qwen3:0.6b", timeout=900))

# %%
if have("ollama"):
    print(run('ollama run qwen3:0.6b "Reply with exactly: hello from ollama" --verbose',
              timeout=180))

# %% [markdown]
# `--verbose` prints timing: prompt eval rate, generation rate, total duration.
# Compare the generation rate with what you measured in notebook 03 — a
# quantized model on CPU is often *faster* than the bf16 one, because you are
# moving a quarter of the bytes.

# %% [markdown]
# ## 5.4 · llama.cpp — more control
#
# llama.cpp is the engine underneath Ollama, LM Studio and much else. Use it
# directly when you want to choose the exact quant, control GPU layer
# offloading, or run on unusual hardware.
#
# ```bash
# # Install (macOS)
# brew install llama.cpp
#
# # Or build from source
# git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j
# ```
#
# Then run a model straight from Hugging Face:
#
# ```bash
# # Interactive chat; downloads the GGUF on first use
# llama-cli -hf Qwen/Qwen3-0.6B-GGUF:Q4_K_M
#
# # OpenAI-compatible server on :8080
# llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q4_K_M -c 8192 --port 8080
# ```
#
# Flags worth knowing:
#
# | Flag | Meaning |
# |---|---|
# | `-c 8192` | context length. Bigger = more KV cache memory. |
# | `-ngl 99` | offload up to 99 layers to the GPU. `-ngl 0` = pure CPU. |
# | `-t 8` | CPU threads. Set it to your physical core count, not logical. |
# | `--jinja` | use the model's own chat template (needed for tool calling) |

# %%
if have("llama-cli"):
    print(run("llama-cli --version"))
else:
    print("llama.cpp not installed - see the commands above. It is optional;")
    print("Ollama covers the same ground for this workshop.")

# %% [markdown]
# ### Partial GPU offload: the trick for a small GPU
#
# With a 6 GB GPU and a 10 GB model you are not stuck. `-ngl N` puts the first
# `N` layers on the GPU and leaves the rest on the CPU. Fill your VRAM, and the
# remainder runs on CPU. It is much faster than pure CPU, and it is why GGUF is
# so popular on consumer hardware.

# %% [markdown]
# ## 5.5 · bitsandbytes — 4-bit inside transformers
#
# If you have an NVIDIA GPU and want to stay in Python, `bitsandbytes` loads
# any HF model in 4-bit with one config object.
#
# ```bash
# pip install bitsandbytes
# ```

# %%
import torch

if torch.cuda.is_available():
    from qwen_workshop.loading import load_model

    full = load_model("Qwen/Qwen3-0.6B")
    print(f"bf16 : {full.memory_footprint_gb():.2f} GB")

    quantized = load_model("Qwen/Qwen3-0.6B", quantize_4bit=True)
    print(f"int4 : {quantized.memory_footprint_gb():.2f} GB")
    print(f"ratio: {full.memory_footprint_gb() / quantized.memory_footprint_gb():.1f}x smaller")
else:
    print("No CUDA GPU -> bitsandbytes is unavailable.")
    print("On CPU or Apple Silicon, use GGUF via Ollama or llama.cpp instead.")
    print("\nThe config you would use:")
    print("""
    from transformers import BitsAndBytesConfig
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4: tuned for weight distributions
        bnb_4bit_use_double_quant=True,     # quantize the scales too, saves ~0.4 bits/param
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in int4
    )
    """)

# %% [markdown]
# Note `bnb_4bit_compute_dtype`: weights are *stored* in 4 bits but
# *dequantized to bf16* for each matrix multiply. You save memory, not compute.
# That is why 4-bit is not automatically faster on a GPU — though it often is
# in practice, because these workloads are memory-bandwidth bound.

# %% [markdown]
# ## 5.6 · Does quantization actually hurt? Measure it.
#
# Do not take anyone's word for it, including mine. Run a task you care about
# at two precisions and compare. Here is a tiny harness.

# %%
QUIZ = [
    ("What is the capital of Kazakhstan?", ["astana", "nur-sultan"]),
    ("What is 12 * 12?", ["144"]),
    ("Which is heavier, 1 kg of steel or 1 kg of feathers?", ["same", "equal", "neither"]),
    ("What language is primarily spoken in Brazil?", ["portuguese"]),
    ("How many days are in a leap year?", ["366"]),
]


def score_model(answer_fn) -> float:
    """Fraction of quiz questions whose answer contains an accepted string."""
    hits = 0
    for question, accepted in QUIZ:
        reply = answer_fn(question).lower()
        ok = any(a in reply for a in accepted)
        hits += ok
        print(f"  [{'ok ' if ok else 'MISS'}] {question} -> {reply[:60]!r}")
    return hits / len(QUIZ)


# Example against Ollama (needs `ollama serve` running):
# from qwen_workshop.client import get_client, complete, is_up
# if is_up("ollama"):
#     client, backend = get_client("ollama", model="qwen3:0.6b")
#     score = score_model(lambda q: complete(
#         client, backend, [{"role": "user", "content": q}],
#         temperature=0.0, max_tokens=50, enable_thinking=False
#     ).choices[0].message.content)
#     print(f"\nQ4_K_M score: {score:.0%}")

print("Uncomment the block above once Ollama is running.")
print(f"This quiz has {len(QUIZ)} questions - far too few to conclude anything.")
print("It shows the *method*: fix the task, vary one thing, measure.")

# %% [markdown]
# > **Exercise 2.** Extend `QUIZ` to 20 questions about something you know well.
# > Score Qwen3-0.6B and Qwen3-4B at Q4_K_M. Is the bigger quantized model
# > better than the smaller unquantized one? (It usually is.)

# %% [markdown]
# ## 5.7 · Which runtime should you use?
#
# | Runtime | Best at | Weak at | Choose it when |
# |---|---|---|---|
# | **transformers** | flexibility, training, research | slow serving, one request at a time | you are experimenting or fine-tuning |
# | **Ollama** | zero-friction local chat | limited tuning, hides details | you want it working in 2 minutes |
# | **llama.cpp** | CPU/Mac speed, GGUF control | more setup | no GPU, or you need exact control |
# | **vLLM** | throughput, many users | needs a real GPU, Linux-first | serving a class or an app |
# | **LM Studio** | GUI, model browser | not scriptable | you prefer clicking to typing |
#
# For the rest of this workshop we use **Ollama** as the default server,
# because it works identically on every student's machine. Everything we write
# also works against vLLM by changing one URL — which is the point of the next
# notebook.
#
# ## Checkpoint
#
# 1. What does 4-bit quantization store per group of weights?
# 2. What do `K` and `M` mean in `Q4_K_M`?
# 3. Bigger model at Q4 or smaller model at Q8 — which, and why?
# 4. Why does 4-bit save memory but not necessarily compute?
#
# ➡️ **Next:** [`06_serving_and_openai_api.ipynb`](06_serving_and_openai_api.ipynb) —
# put the model behind an API so real applications can use it.
