---
title: "Optional · How training works"
parent: Daily plan
nav_order: 5
---

# The training track — notebooks 13 to 16
{: .no_toc }

**Goal:** stop taking training on faith. By the end of this track you have
written a language model and trained it from nothing, prepared a dataset the way
it is actually done, implemented the preference-tuning loss by hand, and
diagnosed five broken training runs.

1. TOC
{:toc}

---

## Who this is for

Chapter 3 of the [handbook](../handbook/how-llms-are-trained.html) describes
pre-training, supervised fine-tuning and preference tuning. Notebook 11 lets you
run a LoRA fine-tune. Between those two there is a gap: you can *describe* the
phases and you can *invoke* a trainer, but you have never seen the loop.

This track closes that gap. Take it if you want to understand training rather
than call it — if you are heading toward research, toward a fine-tune of your
own, or simply find "and then it learns" an unsatisfying sentence.

**Skip it** if your goal is to build applications on top of models. Notebooks
01–12 are complete without this, and prompting, RAG and tools will take you
further than fine-tuning in nearly every product.

## What it costs you

| | |
|---|---|
| **Time** | About 5½ hours, so one more day |
| **Hardware** | A laptop. Only the last cell of notebook 15 wants a GPU, and it is skipped automatically without one. |
| **Prerequisites** | Notebooks 01, 03 and 11. Chapter 3 of the handbook helps but is not required — this track re-derives most of it. |

Notebook 13 trains a real transformer on CPU in a few minutes. Notebooks 14, 15
and 16 are mostly NumPy and plain Python, and run in seconds — deliberately, so
you can change a number and see the consequence immediately rather than waiting
for a GPU queue.

## Schedule

| Time | Activity | Notebook |
|---|---|---|
| 0:00–1:30 | Write a transformer; train it from scratch; watch it learn | [`13_pretrain_a_tiny_model`](../notebooks/13_pretrain_a_tiny_model.html) |
| 1:30–2:45 | Loss masking, dedup, leak-free splits, validation | [`14_building_a_training_dataset`](../notebooks/14_building_a_training_dataset.html) |
| 2:45–4:15 | Implement the DPO loss; build preference pairs; `DPOTrainer` | [`15_preference_tuning_dpo`](../notebooks/15_preference_tuning_dpo.html) |
| 4:15–5:30 | Read five real loss curves; find a learning rate; know when to stop | [`16_training_diagnostics`](../notebooks/16_training_diagnostics.html) |

## The notebooks

### 13 · Pre-train a tiny model

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/13_pretrain_a_tiny_model.ipynb)

A character-level tokenizer, causal self-attention written out in full, and a
~800k-parameter model trained on this repository's own documentation. It writes
recognisable English-shaped text after a few minutes on a CPU.

**The idea you take away:** GPT-4 and this model differ in scale, data and
engineering. They do not differ in objective. Predicting the next token, and
nothing else, is what produced everything you have used in the previous twelve
notebooks.

### 14 · Building a training dataset

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/14_building_a_training_dataset.ipynb)

No model, no GPU, no training run — which is itself the lesson. Loss masking,
near-duplicate detection, splitting without leaking, and a validator for the
format bugs that fail silently.

**The idea you take away:** the data decides whether a fine-tune works. Almost
every failed fine-tune failed here, and the failure was visible before training
started.

### 15 · Preference tuning with DPO

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/15_preference_tuning_dpo.ipynb)

You implement the DPO loss, then use it to answer questions about itself: what β
does, why the reference model is frozen, why a healthy run can report a negative
reward on the answer it is learning to prefer. Then the real `DPOTrainer` call,
which runs if you have a GPU and prints itself if you do not.

**The idea you take away:** preference tuning optimises *what annotators
picked*, which is a proxy for what is good — and when the proxy comes apart you
get sycophancy, verbosity and confident hedging. That is a data problem wearing
a hyperparameter costume.

### 16 · Training diagnostics

[Open in Colab](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/16_training_diagnostics.ipynb)

Five real training runs — healthy, undertrained, overfitting, stuck and diverged
— and a `diagnose()` function that labels them correctly. Then gradient norms,
the learning-rate range test, early stopping, and why perplexity is not the
metric.

**The idea you take away:** train loss going down is almost no evidence. Four of
the five runs have a falling train loss and only one of them is a model worth
keeping.

<img src="../assets/diagrams/loss-curve-patterns.svg" alt="Five loss curve shapes: healthy, undertrained, overfitting, stuck at chance, and diverged, with the decision each implies" width="100%">

## If you only do one

**Notebook 16.** It is the one you will use again. The others explain how
training works; 16 is a skill you apply to every training run you ever watch,
including runs of code you did not write.

## What this track deliberately does not cover

* **Distributed training.** Data, tensor and pipeline parallelism matter above
  one GPU and are irrelevant below it.
* **Online RL** (PPO, GRPO). Frontier post-training uses it; it needs a sampling
  loop and a scale this workshop cannot reach. Notebook 15 explains what it is
  and why DPO replaced it for everyone else.
* **Mixed precision, FlashAttention, gradient checkpointing.** Engineering that
  changes what fits in memory, not what is learned.
* **Training a model that is actually good.** That needs trillions of tokens and
  a cluster. What you can have on a laptop is the mechanism, in full, with
  nothing hidden — and that is what this track gives you.
