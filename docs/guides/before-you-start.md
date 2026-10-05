---
title: Before you start (day 0)
parent: Guides
nav_order: 0
---

# Before you start
{: .no_toc }

**Do this at least three days before the workshop.** Setup failures are the
single biggest waste of day 1, and almost all of them are fixable in advance —
but not in the five minutes before a session begins.

Budget 30 minutes, plus download time.

1. TOC
{:toc}

---

## The checklist

Work down it. Each step tells you what "working" looks like.

### 1. Python 3.10 or newer

```bash
python --version
```

Older than 3.10? Install a current Python before going further — nothing below
will work reliably otherwise.

### 2. Clone and install

```bash
git clone https://github.com/berdakh/ROBT613.git
cd ROBT613

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-lock.txt
```

{: .note }
> Use `requirements-lock.txt`, not `requirements.txt`. The lock file pins exact
> versions so that a release during the workshop week cannot break everyone at
> once.

### 3. Check your machine

```bash
python scripts/check_env.py
```

**Working looks like:** a report ending in `READY`, and a recommended model.

Write down two numbers from the report — you will need them on day 1:

- Your RAM: `________`
- Your GPU and VRAM, if any: `________`

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name qwen-workshop
```

This prevents the single most common confusing failure, where Jupyter silently
runs a different Python from your terminal.

### 5. Download the models — on good wifi

```bash
python scripts/download_models.py --set core
```

About 3 GB. **Do not leave this until the morning of day 1**, when thirty people
download it simultaneously over the same connection.

### 6. Install Ollama

Needed from day 2 onwards.

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows: installer from ollama.com

ollama pull qwen3:0.6b
ollama pull qwen3:4b       # if you have 8 GB+ RAM to spare - strongly recommended
```

**Working looks like:** `ollama list` shows both models.

### 7. Prove it end to end

```bash
python scripts/smoke_test.py
```

**Working looks like:** `ALL CHECKS PASSED`, with generated text and sensible
embedding similarities.

This is the real test. If it passes, you are ready.

---

## If something fails

1. Check [troubleshooting](troubleshooting.html) — it is organised by the error
   you are seeing.
2. Still stuck? **Report it now, not on the day.**
   [Open an issue](https://github.com/berdakh/ROBT613/issues/new/choose) with
   the full output of `python scripts/check_env.py`.

{: .warning }
> Reporting a setup failure three days early costs you five minutes. Discovering
> it at 09:05 on day 1 costs you a morning.

---

## No suitable machine?

You have two fallbacks, and both are fine:

**Google Colab.** Every notebook has an *Open in Colab* badge and runs on a free
T4. See [Run in Colab](colab.html). Nothing to install.

**The lab server.** If your instructor has set one up, you only need:

```bash
export WORKSHOP_LAB_URL=<the url they give you>
```

Then any notebook works with `BACKEND = "lab"`, and days 2–4 proceed normally
without a local model at all.

---

## Bring something to work on

Notebooks 09 and 12 get dramatically better with **your own documents**. Have a
folder ready:

- lecture notes or slides converted to text or markdown,
- documentation for a project you work on,
- your own reading notes, recipes, or journal.

Anything you genuinely want to be able to search. This is the moment the
workshop stops being an exercise — and because the model runs locally, your
files never leave your machine.

---

## For instructors

Send this page out a week ahead with a hard ask: **reply once `smoke_test.py`
passes.** The replies tell you how much of day 1 you need to reserve for
firefighting, and the non-replies tell you exactly who to help first.

If you can, stand up a shared server as insurance:

```bash
vllm serve Qwen/Qwen3-8B --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes --host 0.0.0.0
```

Then give students one line: `export WORKSHOP_LAB_URL=http://<host>:8000/v1`.
A student whose laptop defeats them keeps all four days instead of losing three.
