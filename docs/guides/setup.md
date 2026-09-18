---
title: Setup
parent: Guides
nav_order: 1
---

# Setup
{: .no_toc }

Everything you need before day 1. Budget 30 minutes, plus download time.

1. TOC
{:toc}

---

## Quick start

```bash
git clone https://github.com/berdakh/ROBT613.git
cd ROBT613

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python scripts/check_env.py
```

If `check_env.py` prints a report and no errors, you are ready. Open
`notebooks/01_llm_foundations.ipynb`.

## Requirements

- **Python 3.10 or newer.** Check with `python --version`.
- **~10 GB of free disk** for models and caches.
- **8 GB RAM minimum.** More is better; a GPU is optional.

## Choosing your path

| Situation | Do this |
|---|---|
| Laptop with 8 GB+ RAM | Local install, use Qwen3-0.6B |
| Laptop with an NVIDIA GPU | Local install, everything works |
| Apple Silicon Mac | Local install; MPS is supported |
| No suitable machine | [Use Colab](#google-colab) |
| Restricted network | Local install + [a mirror](#slow-or-blocked-downloads) |

## Install the dependencies

The requirements are split so you only install what you need.

```bash
pip install -r requirements.txt          # core: days 1-3
pip install -r requirements-train.txt    # day 4 fine-tuning (GPU)
pip install -r requirements-dev.txt      # tests and notebook tooling
```

{: .note }
> `requirements.txt` deliberately does **not** pin exact versions, because the
> right `torch` build depends on your CUDA version. If you need a specific CUDA
> build, install torch first from
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/), then run
> `pip install -r requirements.txt`.

## Register the Jupyter kernel

A very common failure is Jupyter running a different Python from your terminal.
Avoid it:

```bash
python -m ipykernel install --user --name qwen-workshop --display-name "Python 3 (qwen-workshop)"
```

Then in Jupyter: **Kernel → Change kernel → Python 3 (qwen-workshop)**.

To verify, run this in a notebook cell and in your terminal, and compare:

```python
import sys; print(sys.executable)
```

## Install Ollama

Needed from day 2 onwards. It is the simplest way to run a local
OpenAI-compatible server.

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download the installer from ollama.com
```

Then:

```bash
ollama pull qwen3:0.6b     # ~500 MB, required
ollama pull qwen3:4b       # ~2.5 GB, recommended if you have the RAM
ollama list                # confirm
```

## Pre-download the models

Do this on good wifi, ideally the night before:

```bash
python scripts/download_models.py --set core     # ~3 GB, required
python scripts/download_models.py --set full     # ~8 GB, everything
python scripts/download_models.py --list         # see what each set contains
```

Once downloaded you can work entirely offline:

```bash
export HF_HUB_OFFLINE=1
```

## Google Colab

Every notebook runs on a free T4.

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub**, paste `berdakh/ROBT613`
3. **Runtime → Change runtime type → T4 GPU**
4. Run this as the first cell:

```python
!git clone https://github.com/berdakh/ROBT613.git
%cd ROBT613
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, "src")
```

{: .warning }
> Colab wipes the disk when the session ends and disconnects after inactivity.
> Save anything you want to keep to Google Drive, and expect to re-download
> models each session.

Ollama does not run well in Colab. Use the `transformers` path (notebooks 03,
04, 08, 09, 11) rather than the server path (06, 07, 10, 12), or install vLLM.

## Windows notes

- Use **PowerShell**, not `cmd`. Activate with `.venv\Scripts\Activate.ps1`.
- If activation is blocked:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- `bitsandbytes` support on Windows is patchy. Use Ollama/GGUF for quantized
  models, or WSL2 for the full Linux toolchain.
- Long-path errors during download: enable long paths in Windows, or set
  `HF_HOME` to a short path such as `C:\hf`.

## Slow or blocked downloads

```bash
# Option 1: a Hugging Face mirror
export HF_ENDPOINT=https://hf-mirror.com

# Option 2: ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B

# Option 3: Ollama, which uses its own CDN
ollama pull qwen3:0.6b
```

Downloads resume, so retrying after a failure does not start from zero.

## Control where models are cached

By default: `~/.cache/huggingface/hub`. On a laptop with a small system drive,
move it:

```bash
export HF_HOME=/path/with/space/huggingface     # add to ~/.bashrc or ~/.zshrc
```

Useful commands:

```bash
huggingface-cli scan-cache      # what is taking up space
huggingface-cli delete-cache    # interactive cleanup
```

## Verify everything

```bash
python scripts/check_env.py      # hardware, packages, cache
python -m pytest -q              # the workshop's own tests (no model needed)
python scripts/smoke_test.py     # end-to-end: loads a model and generates
```

`smoke_test.py` is the real check — if it prints generated text, everything
works.

## Still stuck?

See [Troubleshooting](troubleshooting.html), or open an issue on the repository
with the full output of `python scripts/check_env.py`.
