# %% [markdown]
# # 09 · RAG: answering from your own documents
#
# **Day 3 · ~90 minutes**
#
# Retrieval-Augmented Generation is the highest-value technique in this
# workshop. It turns a small model that knows nothing about you into an
# assistant that answers accurately about *your* notes, *your* course, *your*
# documents — with citations.
#
# By the end you will be able to:
#
# 1. build a complete RAG pipeline end to end;
# 2. make the model cite its sources and refuse when it should;
# 3. **measure** retrieval and answer quality rather than eyeballing it;
# 4. diagnose which stage is broken when an answer is wrong.

# %%
import json
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_workshop.client import complete, get_client, require_backend

BACKEND, MODEL = "ollama", "qwen3:0.6b"
require_backend(BACKEND)
client, backend = get_client(BACKEND, model=MODEL)
print(backend)

# %% [markdown]
# ## 9.1 · Why RAG rather than fine-tuning?
#
# | | RAG | Fine-tuning |
# |---|---|---|
# | Add new facts | instantly, just add a file | retrain |
# | Remove a fact | delete the file | retrain, and hope |
# | Cite sources | natural | impossible |
# | Cost to update | seconds | GPU-hours |
# | Good for | **knowledge** | **behaviour, format, style** |
#
# **Fine-tuning teaches a model how to speak. RAG tells it what to say.**
# Reach for RAG first; nine times out of ten it is what you actually needed.
# (Notebook 11 covers the tenth case.)

# %% [markdown]
# ## 9.2 · The pipeline, one stage at a time

# %%
from qwen_workshop.rag import Embedder, VectorIndex, load_corpus

# Stage 1 - chunk.
chunks = load_corpus(REPO_ROOT / "data" / "notes", max_chars=700, overlap=120)
print(f"stage 1: {len(chunks)} chunks")

# Stage 2 - embed and index (reuse notebook 08's saved index if present).
index_path = REPO_ROOT / "data" / ".index" / "notes"
embedder = Embedder("Qwen/Qwen3-Embedding-0.6B")

if index_path.with_suffix(".npy").exists():
    index = VectorIndex.load(index_path)
    print(f"stage 2: loaded cached index ({len(index)} chunks)")
else:
    index = VectorIndex(chunks, embedder.encode_documents([c.text for c in chunks]))
    index.save(index_path)
    print(f"stage 2: built and saved index ({len(index)} chunks)")

# %%
# Stage 3 - retrieve.
from qwen_workshop.rag import build_rag_prompt, format_context

question = "When and where is my ROBT613 midterm?"
hits = index.search(embedder.encode_queries([question])[0], k=3)

print(f"stage 3: retrieved {len(hits)} chunks for {question!r}\n")
for chunk, score in hits:
    print(f"  [{score:.3f}] {chunk.citation}")

# %%
# Stage 4 - stuff the context into a prompt and generate.
prompt = build_rag_prompt(question, hits)
print("=== the full prompt sent to the model ===")
print(prompt[:1200])
print("...\n")

answer = complete(client, backend, [{"role": "user", "content": prompt}],
                  temperature=0.2, max_tokens=300, enable_thinking=False)
print("=== the answer ===")
print(answer.choices[0].message.content)

# %% [markdown]
# **Print the prompt.** Every RAG bug is visible there: wrong chunks retrieved,
# chunks truncated mid-sentence, or instructions buried under 4,000 tokens of
# context.

# %% [markdown]
# ## 9.3 · Wrap it into one function

# %%
from dataclasses import dataclass

from qwen_workshop.rag import cited_sources


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list
    refused: bool
    citations: list[str]
    invented_citations: list[str]

    @property
    def top_score(self) -> float:
        return self.hits[0][1] if self.hits else 0.0


def rag(question: str, k: int = 3, min_score: float = 0.35, temperature: float = 0.2) -> RagAnswer:
    """Retrieve, then answer strictly from what was retrieved.

    `min_score` is the guard from notebook 08: a vector index always returns k
    neighbours, so without a floor we would happily feed the model the three
    least-irrelevant chunks and invite it to invent an answer.
    """
    hits = index.search(embedder.encode_queries([question])[0], k=k)
    hits = [(chunk, score) for chunk, score in hits if score >= min_score]

    if not hits:
        return RagAnswer(question, "I cannot answer that from the provided documents.",
                         [], True, [], [])

    reply = complete(client, backend,
                     [{"role": "user", "content": build_rag_prompt(question, hits)}],
                     temperature=temperature, max_tokens=400, enable_thinking=False)
    text = (reply.choices[0].message.content or "").strip()

    retrieved = {chunk.citation for chunk, _ in hits}
    citations = cited_sources(text)
    return RagAnswer(
        question=question,
        answer=text,
        hits=hits,
        refused="cannot answer" in text.lower(),
        citations=citations,
        invented_citations=[c for c in citations if c not in retrieved],
    )


for q in [
    "When is my exam and in which room?",
    "How much water do I need per cup of rice?",
    "What is my monthly budget target?",
    "What is the capital of Brazil?",          # not in the notes - must refuse
]:
    result = rag(q)
    print(f"Q: {q}")
    print(f"A: {result.answer[:260]}")
    print(f"   (top score {result.top_score:.3f}, cited {result.citations or 'nothing'}"
          + (f", INVENTED {result.invented_citations}" if result.invented_citations else "")
          + ")\n")

# %% [markdown]
# The Brazil question is the important one. A good RAG system **refuses**.
# A bad one answers from the model's own memory, and you cannot tell the
# difference by looking.

# %% [markdown]
# ## 9.4 · Measure it
#
# "It looks good" is not an evaluation. `data/eval_questions.json` holds
# questions with their expected source files and expected keywords.
#
# Measure the two stages **separately** — they fail for different reasons and
# have different fixes.

# %%
eval_set = json.loads((REPO_ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
print(f"{len(eval_set)} evaluation questions")
print(json.dumps(eval_set[0], indent=2, ensure_ascii=False))

# %%
def evaluate_retrieval(k: int = 3) -> float:
    """Recall@k: did we retrieve at least one chunk from an expected file?"""
    hits_count = 0
    scorable = 0
    print(f"{'question':<52}{'expected':<14}{'retrieved':<22}{'ok'}")
    print("-" * 96)
    for item in eval_set:
        expected = set(item["expected_sources"])
        if not expected:
            continue  # the refusal question is scored in the answer evaluation
        scorable += 1
        retrieved = index.search(embedder.encode_queries([item["question"]])[0], k=k)
        sources = [c.source for c, _ in retrieved]
        ok = bool(expected & set(sources))
        hits_count += ok
        print(f"{item['question'][:50]:<52}{','.join(expected):<14}"
              f"{','.join(dict.fromkeys(sources))[:20]:<22}{'yes' if ok else 'NO'}")
    recall = hits_count / scorable
    print(f"\nRecall@{k} = {hits_count}/{scorable} = {recall:.0%}")
    return recall


recall_at_3 = evaluate_retrieval(k=3)

# %% [markdown]
# **If recall is low, no prompt engineering will save you.** The model cannot
# answer from text it never received. Fix retrieval first:
#
# - smaller chunks (more precise) or larger chunks (more context)
# - more overlap
# - a bigger `k`
# - hybrid search (notebook 08)
# - a reranker (section 9.6)

# %%
for k in (1, 2, 3, 5):
    retrieved_ok = sum(
        bool(set(item["expected_sources"]) &
             {c.source for c, _ in index.search(
                 embedder.encode_queries([item["question"]])[0], k=k)})
        for item in eval_set if item["expected_sources"]
    )
    total = sum(1 for item in eval_set if item["expected_sources"])
    print(f"Recall@{k} = {retrieved_ok}/{total} = {retrieved_ok / total:.0%}")

# %% [markdown]
# Recall rises with `k` — but so does the amount of irrelevant text in the
# prompt, which distracts a small model and costs latency. That trade-off is
# what a reranker exists to resolve.

# %%
def evaluate_answers() -> dict:
    """Score answers on keyword presence, refusal behaviour and citation honesty."""
    scores = {"correct": 0, "refused_correctly": 0, "hallucinated_citation": 0}
    total_answerable = 0

    for item in eval_set:
        result = rag(item["question"])
        expects_refusal = not item["expected_sources"]

        if expects_refusal:
            scores["refused_correctly"] += result.refused
            verdict = "refused (correct)" if result.refused else "ANSWERED (should refuse)"
        else:
            total_answerable += 1
            found = any(kw.lower() in result.answer.lower() for kw in item["expected_keywords"])
            scores["correct"] += found
            verdict = "correct" if found else f"MISS (wanted {item['expected_keywords']})"

        if result.invented_citations:
            scores["hallucinated_citation"] += 1
            verdict += f" + INVENTED CITATION {result.invented_citations}"

        print(f"  {item['question'][:54]:<56}{verdict}")

    print(f"\nAnswer accuracy    : {scores['correct']}/{total_answerable}")
    print(f"Correct refusals   : {scores['refused_correctly']}/"
          f"{sum(1 for i in eval_set if not i['expected_sources'])}")
    print(f"Invented citations : {scores['hallucinated_citation']} (want 0)")
    return scores


answer_scores = evaluate_answers()

# %% [markdown]
# > **Exercise 1.** Write 10 evaluation questions about **your own** documents,
# > with expected sources. Measure recall@3. Then change one thing — chunk
# > size, `k`, or the prompt — and measure again. Report which change helped.
# > This single exercise is worth more than any amount of prompt tweaking by feel.

# %% [markdown]
# ## 9.5 · Diagnosing a wrong answer
#
# When RAG gives a bad answer, there are exactly four possible causes. Work
# through them in order — do not skip to step 4.

# %%
def diagnose(question: str, k: int = 3) -> None:
    """Walk the four failure modes for one question."""
    query_vector = embedder.encode_queries([question])[0]
    hits = index.search(query_vector, k=k)

    print(f"Q: {question}\n")
    print("1. RETRIEVAL - what came back?")
    for chunk, score in hits:
        flag = "  <- weak" if score < 0.4 else ""
        print(f"     [{score:.3f}] {chunk.citation}{flag}")
        print(f"             {' '.join(chunk.text.split())[:90]}...")

    print("\n2. CHUNKING - is the answer split across a boundary?")
    print(f"     chunk sizes: {[len(c) for c, _ in hits]}")
    print("     (if the answer straddles two chunks, increase overlap)")

    print("\n3. PROMPT - how much context is the model juggling?")
    prompt = build_rag_prompt(question, hits)
    print(f"     prompt is ~{len(prompt) // 4} tokens")

    print("\n4. GENERATION - what did it do with it?")
    result = rag(question, k=k)
    print(f"     {result.answer[:220]}")
    print(f"\n   Verdict: "
          + ("retrieval failed - fix chunking/embedding/k"
             if not hits or hits[0][1] < 0.4
             else "retrieval looks fine - the generation step is at fault"))


diagnose("What did I write about multi-head attention?")

# %% [markdown]
# ## 9.6 · Reranking (optional, high value)
#
# Retrieve 20 cheaply, then rerank with a slower, more accurate cross-encoder
# and keep the best 3. The reranker reads the query and the document
# *together*, so it catches relevance an embedding comparison misses.

# %%
def rerank_with_reranker(question: str, candidates: list, top_k: int = 3):
    """Rerank with Qwen3-Reranker-0.6B. Downloads ~1.2 GB on first use."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("pip install sentence-transformers")
        return candidates[:top_k]

    try:
        reranker = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", trust_remote_code=True)
        pairs = [(question, chunk.text) for chunk, _ in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip([c for c, _ in candidates], scores), key=lambda r: -r[1])
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
    except Exception as exc:  # noqa: BLE001
        print(f"Reranker unavailable ({type(exc).__name__}); keeping embedding order.")
        return candidates[:top_k]


question = "what should I do about the radiator?"
wide = index.search(embedder.encode_queries([question])[0], k=min(8, len(index)))
print("embedding order:")
for chunk, score in wide[:4]:
    print(f"  [{score:.3f}] {chunk.citation}: {' '.join(chunk.text.split())[:70]}...")

print("\nafter reranking:")
for chunk, score in rerank_with_reranker(question, wide):
    print(f"  [{score:.3f}] {chunk.citation}: {' '.join(chunk.text.split())[:70]}...")

# %% [markdown]
# ## 9.7 · Practical tuning guide
#
# | Symptom | Likely cause | Try |
# |---|---|---|
# | Right file, wrong part | chunks too big | `max_chars` 400–600 |
# | Answer cut in half | boundary split | raise `overlap` to 200 |
# | Retrieves nothing relevant | vocabulary mismatch | hybrid search, or rewrite the query with the LLM first |
# | Model ignores the context | context too long | fewer chunks, instruction *after* the data |
# | Model invents facts | no refusal instruction | strengthen the prompt, raise `min_score` |
# | Slow | embedding every query | cache the index; batch queries |
#
# **Query rewriting** deserves a mention: ask the LLM to turn "what about the
# thing with the heating?" into "radiator boiler bleeding heating maintenance"
# before embedding. It is cheap and often the single biggest win on
# conversational questions.

# %%
def rewrite_query(question: str) -> str:
    """Expand a vague question into keyword-rich search text."""
    reply = complete(client, backend, [{"role": "user", "content":
        "Rewrite this question as a short list of search keywords. "
        "Output only the keywords, space separated, no explanation.\n\n"
        f"Question: {question}"}],
        temperature=0.0, max_tokens=40, enable_thinking=False)
    return (reply.choices[0].message.content or question).strip()


vague = "what about the thing with the heating?"
rewritten = rewrite_query(vague)
print(f"original : {vague}")
print(f"rewritten: {rewritten}\n")

for label, text in (("original", vague), ("rewritten", rewritten)):
    top = index.search(embedder.encode_queries([text])[0], k=1)[0]
    print(f"{label:<10} -> [{top[1]:.3f}] {top[0].citation}")

# %% [markdown]
# ## Checkpoint
#
# 1. What are the four stages of RAG?
# 2. Why measure retrieval separately from answer quality?
# 3. What does recall@k measure, and what do you do when it is low?
# 4. Why is a score threshold necessary?
# 5. When would you fine-tune instead of using RAG?
#
# ➡️ **Next:** [`10_agents_from_scratch.ipynb`](10_agents_from_scratch.ipynb) —
# let the model decide *which* tool to use, and in what order.
