# %% [markdown]
# # 11 · Fine-tuning Qwen with LoRA
#
# **Day 4 · ~90 minutes · GPU strongly recommended (use Colab if you have none)**
#
# Fine-tuning changes how a model *behaves*. It is the right tool much less
# often than people think — but when it is right, nothing else works.
#
# By the end you will be able to:
#
# 1. decide correctly between prompting, RAG and fine-tuning;
# 2. explain what LoRA does and why it makes this affordable;
# 3. prepare a dataset in the right format;
# 4. train an adapter on Qwen3-0.6B and use it;
# 5. recognise overfitting and catastrophic forgetting.

# %% [markdown]
# ## 11.1 · Decide before you train
#
# Work down this list. **Stop at the first "yes".**
#
# 1. Can a better **prompt** fix it? → do that. Minutes, free, reversible.
# 2. Is the problem **missing knowledge**? → **RAG** (notebook 09). Updating a
#    file beats retraining, always.
# 3. Do you need a **structured output**? → constrained decoding (notebook 07).
# 4. Does it need to **do things**? → tools (notebook 10).
# 5. Do you need a consistent **style, format, or domain language** that
#    prompting cannot hold reliably across many turns? → **now** fine-tune.
#
# | Fine-tuning is good at | Fine-tuning is bad at |
# |---|---|
# | Consistent tone and format | Adding facts (they decay and cannot be cited) |
# | Domain vocabulary and conventions | Anything that changes weekly |
# | Following a niche output schema | Making a small model fundamentally smarter |
# | Making a small model imitate a big one on **one** narrow task | General capability |
#
# > The single most common mistake in industry: fine-tuning to inject company
# > knowledge. It half-works, cannot cite sources, and must be redone whenever
# > a document changes. Use RAG.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from qwen_workshop.env import describe_environment

print(describe_environment())
if not torch.cuda.is_available():
    print("\n" + "!" * 70)
    print("No CUDA GPU detected. Training Qwen3-0.6B on CPU is possible but slow")
    print("(30-60+ minutes for this tiny example). Options:")
    print("  1. Open this notebook in Colab with a T4 (free).")
    print("  2. Read through and run only the data-preparation sections.")
    print("  3. Reduce MAX_STEPS below to 20 and accept a weak adapter.")
    print("!" * 70)

# %% [markdown]
# ## 11.2 · What LoRA does
#
# Full fine-tuning updates every weight. For an 8B model in bf16 that needs
# roughly: 16 GB weights + 16 GB gradients + 64 GB optimiser state ≈ **96 GB**.
# Not happening on your laptop.
#
# **LoRA (Low-Rank Adaptation)** freezes the original weights and learns a
# small additive correction. For a weight matrix $W \in \mathbb{R}^{d \times k}$
# it learns two thin matrices:
#
# $$W' = W + \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll d$$
#
# With $r = 16$ you train well under 1% of the parameters. The adapter file is
# a few megabytes, you can keep dozens for different tasks, and you can remove
# one instantly by not loading it.

# %%
def lora_savings(d: int = 4096, k: int = 4096, rank: int = 16) -> None:
    full = d * k
    lora = rank * (d + k)
    print(f"one {d}x{k} weight matrix")
    print(f"  full fine-tune : {full:>12,} parameters")
    print(f"  LoRA (r={rank:<3})   : {lora:>12,} parameters")
    print(f"  trained         : {lora / full:>12.2%}")


for rank in (4, 16, 64):
    lora_savings(rank=rank)
    print()

# %% [markdown]
# **Choosing the rank:** `r=8–16` for style and format; `r=32–64` for a genuinely
# new task or domain. Higher rank means more capacity and more overfitting risk.
# `lora_alpha` is conventionally set to `2 * r`.

# %% [markdown]
# ## 11.3 · Install
#
# ```bash
# pip install peft trl datasets accelerate
# ```

# %%
def ensure(package: str, import_name: str | None = None) -> bool:
    import importlib

    try:
        importlib.import_module(import_name or package.replace("-", "_"))
        return True
    except ImportError:
        print(f"installing {package}...")
        import subprocess

        done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", package])
        return done.returncode == 0


ready = all([ensure("peft"), ensure("trl"), ensure("datasets"), ensure("accelerate")])
print("training stack ready:", ready)

# %% [markdown]
# ## 11.4 · Build a dataset
#
# Our task: teach the model to answer **as a terse Kazakhstani student budget
# assistant** — always in tenge, always with a one-line actionable tip, never
# with preamble. That is a *style and format* task, which is exactly what
# fine-tuning is for.
#
# Format: a list of message lists, the same shape you use at inference.
# **Training data must look exactly like production input.**

# %%
import json

TRAIN_EXAMPLES = [
    ("I spent 4200 on groceries today.",
     "Logged: 4,200 ₸ groceries. Tip: buy lentils and rice in bulk - same protein, half the cost."),
    ("How much should I budget for food each month?",
     "Target: 60,000-70,000 ₸ for one student. Tip: cook 3 portions at once; the third meal is free time."),
    ("I bought a coffee for 1500.",
     "Logged: 1,500 ₸ eating out. Tip: 20 coffees a month is 30,000 ₸ - a thermos pays for itself in a week."),
    ("Is 12000 a lot for a gym?",
     "12,000 ₸/month is mid-range in Astana. Tip: check if your university gym is free before renewing."),
    ("I need to save for a trip to Almaty.",
     "Train is ~8,000 ₸ vs 25,000+ ₸ flying. Tip: book the overnight train and save a hotel night too."),
    ("My electricity bill was 9800 this month.",
     "Logged: 9,800 ₸ utilities. Tip: that is a cold-month bill - drop the thermostat 1°C and save ~5%."),
    ("I spent 6000 on a textbook.",
     "Logged: 6,000 ₸ study. Tip: check the library and past-year students before buying new."),
    ("What is a reasonable monthly total?",
     "Under 180,000 ₸ is a solid student target. Tip: track eating out - it is the category that always creeps."),
    ("I paid 1800 for a taxi.",
     "Logged: 1,800 ₸ transport. Tip: a monthly metro pass beats 15 taxi rides."),
    ("I want to cut my spending by 20%.",
     "Cut 36,000 ₸ from a 180,000 ₸ month. Tip: eating out and fun are the two flexible categories - start there."),
    ("Bought lunch at the canteen, 900 tenge.",
     "Logged: 900 ₸ eating out. Tip: the canteen is the cheap option - this is fine, keep it."),
    ("How much do I waste on subscriptions?",
     "Typical student: 5,000-15,000 ₸/month. Tip: list every recurring charge today and cancel one."),
]

SYSTEM = ("You are a terse budget assistant for a student in Kazakhstan. "
          "Always use tenge (₸). Reply in at most two short lines: the logged "
          "amount or figure, then one actionable tip. Never add preamble.")

dataset_rows = [
    {"messages": [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user},
                  {"role": "assistant", "content": assistant}]}
    for user, assistant in TRAIN_EXAMPLES
]

train_path = REPO_ROOT / "data" / "budget_style_train.jsonl"
train_path.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in dataset_rows), encoding="utf-8"
)
print(f"wrote {len(dataset_rows)} examples to {train_path.relative_to(REPO_ROOT)}")
print(json.dumps(dataset_rows[0], ensure_ascii=False, indent=2))

# %% [markdown]
# ### How much data do you need?
#
# | Examples | What you get |
# |---|---|
# | 10–50 | Style and format transfer. Often enough! |
# | 100–1,000 | Reliable behaviour on a narrow task |
# | 1,000–10,000 | A genuinely specialised model |
# | 10,000+ | Diminishing returns without careful curation |
#
# **Quality beats quantity, by a lot.** Fifty consistent, carefully written
# examples beat a thousand scraped ones. Every inconsistency in your data is an
# instruction to the model to be inconsistent.
#
# Our 12 examples are a demonstration, not a real dataset. Expect a visible but
# imperfect effect.

# %% [markdown]
# ## 11.5 · Baseline first
#
# **Always measure before training.** Otherwise you cannot tell whether
# training helped.

# %%
from qwen_workshop.chat import chat
from qwen_workshop.loading import load_model

BASE_MODEL = "Qwen/Qwen3-0.6B"
base = load_model(BASE_MODEL)

TEST_PROMPTS = [
    "I spent 3500 on groceries.",
    "How much should I spend on transport?",
    "I bought new headphones for 25000.",
]

print("=== BEFORE fine-tuning ===")
for prompt in TEST_PROMPTS:
    reply = chat(base, prompt, system=SYSTEM,
                 sampling=None, enable_thinking=False)
    print(f"\nQ: {prompt}\nA: {reply.text[:300]}")

# %% [markdown]
# Note what is wrong: too long, preamble, inconsistent formatting, possibly the
# wrong currency. That is the gap we are closing.

# %% [markdown]
# ## 11.6 · Train the adapter

# %%
if ready:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_list(dataset_rows)

    lora_config = LoraConfig(
        r=16,                    # rank: capacity of the adapter
        lora_alpha=32,           # scaling, conventionally 2 * r
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Which matrices get adapters. Attention + MLP is the standard choice;
        # attention-only (q,k,v,o) is cheaper and often enough for style.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    MAX_STEPS = 60 if torch.cuda.is_available() else 20
    output_dir = REPO_ROOT / "outputs" / "qwen3-budget-lora"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,     # effective batch size 4
        learning_rate=2e-4,                # LoRA likes 1e-4 to 3e-4; much higher than full FT
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to=[],                      # no wandb prompt
        max_length=512,
        bf16=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable: {trainable:,} / {total:,} = {trainable / total:.2%}\n")

    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"\nadapter saved to {output_dir}")
else:
    print("Training stack unavailable - skipping. Read on for the results discussion.")

# %% [markdown]
# **Reading the loss.** It should fall and then flatten. If it drops to near
# zero on 12 examples, you have memorised them — which on a toy dataset is
# expected, and on a real one means stop and add data.

# %%
if ready:
    import glob
    import os

    files = glob.glob(str(output_dir / "*"))
    total_mb = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1024**2
    print(f"adapter size: {total_mb:.1f} MB")
    print("(compare with ~1,200 MB for the full model - this is why LoRA is practical)")
    for f in sorted(files):
        print("  ", os.path.basename(f))

# %% [markdown]
# ## 11.7 · Use the adapter

# %%
if ready:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qwen_workshop.loading import LoadedModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    backbone = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tuned_model = PeftModel.from_pretrained(backbone, str(output_dir))
    tuned_model = tuned_model.to(base.device).eval()

    tuned = LoadedModel(model=tuned_model, tokenizer=tokenizer, repo_id="qwen3-budget-lora",
                        device=base.device, dtype=str(tuned_model.dtype), quantized=False)

    print("=== AFTER fine-tuning ===")
    for prompt in TEST_PROMPTS:
        reply = chat(tuned, prompt, system=SYSTEM, sampling=None, enable_thinking=False)
        print(f"\nQ: {prompt}\nA: {reply.text[:300]}")

# %% [markdown]
# Compare with the baseline above. With only 12 examples expect partial
# success: shorter, more consistent, more likely to use ₸ and the two-line
# shape. That partial success *is* the lesson — it shows the direction and the
# cost.
#
# > **Exercise 1.** Write 40 more examples in the same style and retrain.
# > Measure: average reply length, fraction using "₸", fraction with a "Tip:"
# > line. Report the three numbers before and after.

# %% [markdown]
# ## 11.8 · Catastrophic forgetting
#
# A fine-tuned model can get *worse* at everything it was not trained on. Check.

# %%
if ready:
    OFF_TASK = [
        "What is the capital of France?",
        "Write a one-line Python function that reverses a string.",
        "Translate 'good morning' into Kazakh.",
    ]
    for prompt in OFF_TASK:
        before = chat(base, prompt, sampling=None, enable_thinking=False).text
        after = chat(tuned, prompt, sampling=None, enable_thinking=False).text
        print(f"\nQ: {prompt}")
        print(f"  base : {before[:140]}")
        print(f"  tuned: {after[:140]}")

# %% [markdown]
# If the tuned model now answers "What is the capital of France?" with a budget
# tip, that is catastrophic forgetting. Mitigations:
#
# - fewer training steps, or a lower learning rate;
# - a lower LoRA rank;
# - mix 10–20% general instruction data into your training set;
# - target fewer modules (attention only).
#
# LoRA is more resistant than full fine-tuning — the base weights are intact —
# but it is not immune.

# %% [markdown]
# ## 11.9 · Merging and deployment
#
# An adapter can be **merged** into the base weights to produce a standalone
# model with no inference overhead and no PEFT dependency.
#
# ```python
# merged = PeftModel.from_pretrained(backbone, adapter_dir).merge_and_unload()
# merged.save_pretrained("qwen3-budget-merged")
# tokenizer.save_pretrained("qwen3-budget-merged")
# ```
#
# Then convert to GGUF for Ollama/llama.cpp:
#
# ```bash
# python llama.cpp/convert_hf_to_gguf.py qwen3-budget-merged --outfile budget.gguf
# ./llama.cpp/build/bin/llama-quantize budget.gguf budget-q4.gguf Q4_K_M
# ```
#
# **Keep the adapter separate** if you want several behaviours from one base
# model — you can hot-swap adapters and serve them from one loaded backbone.
#
# ### Licensing
#
# Qwen3's Apache 2.0 licence permits commercial use and redistribution of
# derivatives. Check the specific model card: other families (and some Qwen 2.5
# sizes) use different licences with conditions attached.
#
# ## Checkpoint
#
# 1. Give the five-step decision list before fine-tuning.
# 2. What does LoRA learn, and why is it so much cheaper?
# 3. How would you detect overfitting on 12 examples?
# 4. Why measure the baseline before training?
# 5. When do you merge an adapter, and when do you keep it separate?
#
# ➡️ **Next:** [`12_capstone_daily_assistant.ipynb`](12_capstone_daily_assistant.ipynb) —
# put it all together and build something you will actually use.
