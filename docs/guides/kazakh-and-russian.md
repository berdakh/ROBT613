---
title: Kazakh and Russian
parent: Guides
nav_order: 6
---

# Building in Kazakh and Russian
{: .no_toc }

Every other page here is written for an audience that could be anywhere. This
one is not.

If you build in Kazakh or Russian you pay a tax the English-language tutorials
never mention, because their authors do not pay it. It is real, it is
measurable, and left unmeasured it will quietly wreck your context budget,
your costs and your retrieval quality.

1. TOC
{:toc}

---

{: .note }
> **The companion notebook does the measuring.**
> [`22_kazakh_and_russian`](../notebooks/22_kazakh_and_russian.html) builds a
> tokenizer from scratch to show *why* the tax exists, then turns it into
> context, money and latency. Most of it runs with no model at all.

## The short version

| | |
|---|---|
| **Cyrillic is 2 bytes per character in UTF-8** | exact, and the floor under everything else |
| **Tokenizers charge far more than that** | a toy BPE trained on English-dominant text splits Kazakh into single letters — roughly **3x** the tokens |
| **A bigger vocabulary makes it worse** | extra merges go to whatever is frequent, and that is English |
| **Expect ~1.5–2.5x with real Qwen3** | measure your own; notebook 01 and notebook 22 both do it |
| **Root cause: text scarcity, not script** | there is simply far less varied Kazakh text on the public internet |

## What a 2x multiplier costs you

Three consequences, in the order they will bite:

| | Effect |
|---|---|
| **Context** | A 32k window holds ~131,000 characters of English and ~65,000 of Kazakh. Your users get half the document, silently. |
| **Cost** | Twice the bill for the same content on any per-token pricing. |
| **Latency** | Roughly twice the generation time. Tokens come out one at a time, so a Kazakh answer streams at half the speed. |

The latency one is the most visible to users and the least discussed.

## The failure mode nobody tests for

The characteristic failure of a multilingual model on a lower-resource
language is **not** producing bad Kazakh. It is quietly answering in Russian
or English instead — which passes any automated test that only checks whether
the answer is *correct*.

Worse, "is the answer Cyrillic?" does not catch it, because Russian is Cyrillic
too. What does catch it is the nine letters Kazakh has and Russian does not:

```
ә ғ қ ң ө ұ ү һ і
```

Almost any real Kazakh sentence contains several; a Russian sentence contains
none. Notebook 22 turns that into a `script_report` check you can assert on in
a test suite, in about ten lines and with no dependencies.

## RAG in Kazakh degrades twice over

Two effects compound, and neither is obvious from the English tutorials:

1. **Chunks hold less.** If you chunk by *tokens*, a Kazakh chunk carries
   roughly half the content of an English one — so your retrieved context is
   half as informative at the same `k`.
2. **Embeddings are weaker.** Embedding models are trained on the same skewed
   corpora, so semantic similarity in Kazakh is noisier and recall@k drops —
   on top of the chunks already holding less.

Mitigations, in the order worth trying:

| Approach | Why it helps | Cost |
|---|---|---|
| **Chunk by characters, not tokens** | restores content parity per chunk | none — do this today |
| **Hybrid search** ([notebook 08](../notebooks/08_embeddings_and_search.html)) | BM25 is script-agnostic and strong on names and borrowed terms | a little code |
| **Raise `k`** | compensates for weaker ranking | consumes the context you can least afford |
| **Pivot the query through English** | search an English index | a translation call, and the errors it introduces |

Hybrid search is the one most people skip and should not. Kazakh is
**agglutinative** — one stem takes many suffixes — so keyword matching wants
stemming to work well; but for names, institutions and technical loanwords
BM25 is excellent and completely unaffected by the embedding model's English
bias.

## Four strategies

| Strategy | Use when | Watch out for |
|---|---|---|
| **Prompt in Kazakh, accept the tax** | the model is good enough and volume is low | code-switching — test for it |
| **Pivot through Russian** | Russian output is markedly better and users accept it | this is a product decision, not a technical one. Ask them. |
| **Pivot through English** | the task is reasoning-heavy and the model is far stronger in English | two translation steps, two chances to lose meaning; unusable where exact wording matters |
| **Fine-tune on Kazakh** | 500+ good examples and a narrow task | teaches form, not knowledge |

### On fine-tuning specifically

[Notebook 11's](../notebooks/11_finetuning_lora.html) decision list applies
unchanged, with two additions:

* **LoRA will not fix the tokenizer.** An adapter changes behaviour, not the
  vocabulary. Your Kazakh text costs exactly the same after fine-tuning.
  People expect otherwise and are disappointed.
* **Data is the bottleneck, not compute.** [Notebook
  14](../notebooks/14_building_a_training_dataset.html) matters more here than
  notebook 11. Fifty carefully written Kazakh examples beat a thousand
  machine-translated ones — and machine-translated training data teaches the
  model to produce translationese.

Legitimate sources of Kazakh text: Kazakh Wikipedia, open government
publications, university materials you have rights to, news archives with
permission, and your own institution's documents. Check the licence on each;
the [privacy and safety guide](privacy-and-safety.html) applies.

## What to do on Monday

1. **Measure your own multiplier** with the real tokenizer. One number, five
   minutes, and every budget decision downstream depends on it.
2. **Chunk by characters, not tokens**, in anything serving Kazakh.
3. **Add a script check** so code-switching cannot pass silently.
4. **Halve your assumed context budget** for Kazakh until measured otherwise.
5. **Try hybrid search before fine-tuning.** A day's work instead of a
   month's, and it addresses the failure you actually have.

## An open invitation

Notebook 22 ends with exercises that ask for measurements nobody has
published: a Kazakh-language evaluation of an open-weight model, and a
recall@k comparison for Kazakh RAG with and without hybrid search.

Those are genuinely scarce. A class working through this material could
produce the best public numbers that exist, and they would be worth sharing.
