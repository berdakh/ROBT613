# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/02_environment_and_hardware.ipynb)
#
# *Click the badge to run this notebook in Google Colab - no install required. The first code cell sets everything up.*

# %% [markdown]
# # 02 · What can *your* machine run?
#
# **Day 1 · ~45 minutes**
#
# The fastest way to lose an afternoon is to download a 30 GB model onto a
# laptop that cannot load it. This notebook prevents that.
#
# By the end you will be able to:
#
# 1. report your own hardware and read the constraint off it;
# 2. compute the memory a given model needs, rather than guessing;
# 3. choose between Hugging Face, ModelScope, Ollama and a mirror;
# 4. control where models are cached, and clean the cache up.

# %%
# --- Setup: works on your laptop AND in Google Colab ------------------------
# In Colab this clones the workshop repo and installs the dependencies (about a
# minute, first run only) so that `src/`, `data/` and the sample files exist.
# On your own machine it just locates the repo. Safe to re-run either way.
import os
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    if not Path("/content/ROBT613").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/ROBT613.git",
                        "/content/ROBT613"], check=True)
    os.chdir("/content/ROBT613")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], check=True)

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "src" / "qwen_workshop").is_dir())
sys.path.insert(0, str(REPO_ROOT / "src"))
print("repo root:", REPO_ROOT)

from qwen_workshop.env import check_environment, describe_environment

report = check_environment()
print(describe_environment(report))

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT613/master/docs/assets/diagrams/memory-budget.svg" alt="Memory use per model and precision: weights dominate until the context gets long" width="100%">

# %% [markdown]
# ## 2.1 · The memory formula
#
# Almost all of the memory goes to the **weights**. For a model with $N$
# parameters stored at $b$ bytes each:
#
# $$\text{weights} \approx N \times b$$
#
# | Precision | Bytes/parameter | 8B model | Quality |
# |---|---|---|---|
# | fp32 | 4 | 32 GB | reference |
# | **bf16 / fp16** | 2 | 16 GB | the standard default, no visible loss |
# | int8 | 1 | 8 GB | very close to bf16 |
# | **int4** (Q4_K_M, NF4) | ~0.55 | ~4.4 GB | small but real loss; usually worth it |
#
# Then add:
#
# - **KV cache** — grows with conversation length. Roughly
#   $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{tokens} \times b$.
#   For small models it is modest; for long contexts it can exceed the weights.
# - **Activations and framework overhead** — budget 1–2 GB.
#
# **Practical rule: take the weight size and add 25%.**

# %%
def memory_estimate(n_params_billions: float, bits: int = 16, context_tokens: int = 4096,
                    n_layers: int = 28, n_kv_heads: int = 8, head_dim: int = 128) -> dict:
    """Estimate memory for a model. Defaults describe Qwen3-0.6B-ish geometry.

    Returns a dict of GB figures. Approximate by design - the point is to get
    the order of magnitude right before you start a download.
    """
    bytes_per_param = bits / 8
    weights_gb = n_params_billions * 1e9 * bytes_per_param / 1024**3
    # 2 = one tensor for keys, one for values. KV cache is usually fp16.
    kv_gb = 2 * n_layers * n_kv_heads * head_dim * context_tokens * 2 / 1024**3
    overhead_gb = 1.0
    return {
        "weights": weights_gb,
        "kv_cache": kv_gb,
        "overhead": overhead_gb,
        "total": weights_gb + kv_gb + overhead_gb,
    }


print(f"{'model':<18}{'bits':>5}{'weights':>10}{'kv@4k':>9}{'total':>9}")
print("-" * 51)
for name, params, layers, kv_heads in [
    ("Qwen3-0.6B", 0.6, 28, 8),
    ("Qwen3-1.7B", 1.7, 28, 8),
    ("Qwen3-4B", 4.0, 36, 8),
    ("Qwen3-8B", 8.0, 36, 8),
    ("Qwen3-14B", 14.0, 40, 8),
]:
    for bits in (16, 4):
        est = memory_estimate(params, bits, 4096, layers, kv_heads)
        print(f"{name:<18}{bits:>5}{est['weights']:>9.1f}G{est['kv_cache']:>8.2f}G"
              f"{est['total']:>8.1f}G")

# %% [markdown]
# > **Exercise 1.** Re-run with `context_tokens=32768`. At what model size does
# > the KV cache start to rival the weights? This is why long-context chat runs
# > out of memory even though the model "fits".

# %% [markdown]
# ## 2.2 · What should *you* download?

# %%
from qwen_workshop.config import recommend_model

choice = recommend_model(
    vram_gb=report.vram_gb,
    ram_gb=report.ram_gb,
    quantized=True,
)

print(f"Detected: {report.device.upper()}"
      + (f", {report.vram_gb:.1f} GB VRAM" if report.vram_gb else "")
      + f", {report.ram_gb:.1f} GB RAM\n")
print(f"Recommended starting model : {choice.repo_id}")
print(f"  parameters   : {choice.params}")
print(f"  int4 memory  : ~{choice.approx_ram_gb_q4:.1f} GB")
print(f"  why          : {choice.notes}")

# %% [markdown]
# ### Honest expectations for speed
#
# | Setup | Qwen3-0.6B | Qwen3-4B (int4) | Qwen3-8B (int4) |
# |---|---|---|---|
# | Modern GPU (≥8 GB) | 60–150 tok/s | 30–60 tok/s | 20–40 tok/s |
# | Apple Silicon (M1–M3) | 30–60 tok/s | 15–30 tok/s | 8–20 tok/s |
# | CPU only (4–8 cores) | 8–20 tok/s | 3–6 tok/s | 1–3 tok/s |
#
# Below ~5 tokens/second a live demo becomes painful. **If you are CPU-only,
# use Qwen3-0.6B or 1.7B for everything in this workshop** and do the bigger
# experiments in Colab.
#
# Reading comfortably is about 5 words/second ≈ 7 tok/s, which is why streaming
# (notebook 03) makes a slow model feel so much better than it is.

# %% [markdown]
# ## 2.3 · No GPU? Use Colab
#
# Every notebook here runs on a free Colab T4 (16 GB VRAM), which comfortably
# handles Qwen3-4B or Qwen3-8B in int4.
#
# ```python
# # First cell in Colab:
# !git clone https://github.com/berdakh/ROBT613.git
# %cd ROBT613
# !pip install -q -r requirements.txt
# import sys; sys.path.insert(0, "src")
# ```
#
# Then **Runtime → Change runtime type → T4 GPU**. Check you actually got one
# with `!nvidia-smi`. Colab disconnects after inactivity and wipes the disk, so
# save anything you care about to Drive.

# %% [markdown]
# ## 2.4 · Where models come from
#
# | Source | Best for | Command |
# |---|---|---|
# | **Hugging Face Hub** | the default; every format | `huggingface-cli download Qwen/Qwen3-0.6B` |
# | **ModelScope** | faster inside mainland China | `modelscope download --model Qwen/Qwen3-0.6B` |
# | **HF mirror** | when HF is slow/blocked | `export HF_ENDPOINT=https://hf-mirror.com` |
# | **Ollama** | one-command local chat | `ollama pull qwen3:0.6b` |
#
# A Hugging Face model repo contains:
#
# ```
# config.json                 architecture and hyperparameters
# model.safetensors           the weights  <- the big file
# tokenizer.json              the vocabulary
# tokenizer_config.json       includes the chat template
# generation_config.json      recommended sampling defaults
# README.md                   the model card: licence, benchmarks, usage
# ```
#
# **Always use `.safetensors`, never `.bin`.** The old PyTorch `.bin` format is
# a pickle: loading it executes arbitrary code from whoever uploaded it.
# `safetensors` is pure data and cannot execute anything.

# %% [markdown]
# ### Inspect a repo without downloading the weights
#
# `hf_hub_download` on a small file, or `HfApi`, lets you read the config and
# licence first — worth doing before committing to a 16 GB download.

# %%
def inspect_repo(repo_id: str = "Qwen/Qwen3-0.6B") -> None:
    """Print file list, size and config for a HF repo without fetching weights."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub to run this cell")
        return

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:  # offline, rate-limited, gated...
        print(f"Could not reach the Hub ({type(exc).__name__}). Skipping.")
        return

    total = sum(f.size or 0 for f in info.siblings)
    print(f"{repo_id}")
    print(f"  licence   : {(info.card_data or {}).get('license', 'see model card')}")
    print(f"  downloads : {info.downloads:,} (last 30 days)")
    print(f"  total size: {total / 1024**3:.2f} GB")
    print("  files:")
    for f in sorted(info.siblings, key=lambda s: -(s.size or 0))[:8]:
        size = f"{(f.size or 0) / 1024**2:8.1f} MB" if f.size else " " * 11
        print(f"    {size}  {f.rfilename}")


inspect_repo("Qwen/Qwen3-0.6B")

# %% [markdown]
# ## 2.5 · Control your cache before it eats your disk
#
# Downloaded models pile up fast. By default they land in
# `~/.cache/huggingface/hub`. On a laptop with a small system drive, move it:
#
# ```bash
# # Put this in ~/.bashrc or ~/.zshrc
# export HF_HOME=/path/with/space/huggingface
# ```
#
# Useful commands:
#
# ```bash
# huggingface-cli scan-cache      # what is taking up space
# huggingface-cli delete-cache    # interactive cleanup
# export HF_HUB_OFFLINE=1         # never hit the network; use the cache only
# ```
#
# `HF_HUB_OFFLINE=1` is genuinely useful for a workshop: download once over
# wifi, then work entirely offline. It also proves your demo has no hidden
# network dependency.

# %%
print(f"Cache location: {report.hf_home}")
print(f"Currently using: {report.hf_cache_gb:.2f} GB")
print()
print("Budget for this workshop:")
print("  tokenizer only (nb 01)     ~0.01 GB")
print("  Qwen3-0.6B  (bf16)          ~1.5 GB")
print("  Qwen3-Embedding-0.6B        ~1.2 GB")
print("  Qwen3-4B GGUF Q4_K_M        ~2.5 GB  (optional)")
print("  ------------------------------------")
print("  comfortable total           ~6 GB")

# %% [markdown]
# ## 2.6 · Pre-download everything now
#
# Run this **while you still have good wifi**. It fetches only what the core
# notebooks need. You can also run it from a terminal:
#
# ```bash
# python scripts/download_models.py --set core
# ```

# %%
# Uncomment to download now (a few minutes on a decent connection).
# import subprocess
# subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "download_models.py"),
#                 "--set", "core"], check=False)

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Cause | Fix |
# |---|---|---|
# | `CUDA out of memory` | model too big for VRAM | smaller model, or `quantize_4bit=True`, or restart the kernel — old models linger |
# | Download crawls or stalls | network/region | `export HF_ENDPOINT=https://hf-mirror.com`, or use ModelScope |
# | `401 Client Error` | gated repo | accept the licence on the model page, then `huggingface-cli login` |
# | `No space left on device` | cache filled the disk | `huggingface-cli delete-cache`, or move `HF_HOME` |
# | Works in terminal, fails in Jupyter | different environment | run `import sys; print(sys.executable)` in both and compare |
#
# More: [`docs/guides/troubleshooting.md`](../docs/guides/troubleshooting.md)
#
# ## Checkpoint
#
# 1. How much memory does Qwen3-8B need at int4, with overhead?
# 2. Which model did the recommender pick for you, and why?
# 3. Where is your cache, and how would you move it?
# 4. Why prefer `.safetensors` over `.bin`?
#
# ➡️ **Next:** [`03_first_generation.ipynb`](03_first_generation.ipynb) — generate your first tokens.
