# %% [markdown]
# # 08 · Embeddings and semantic search
#
# **Day 3 · ~60 minutes**
#
# A 0.6B model knows almost nothing about your life. Retrieval fixes that: give
# it the right paragraph at the right moment and it becomes genuinely useful.
#
# Retrieval starts with **embeddings** — turning text into vectors where
# *distance means meaning*.
#
# By the end you will be able to:
#
# 1. embed text with Qwen3-Embedding and inspect the vectors;
# 2. explain cosine similarity and compute it by hand;
# 3. build a working semantic search over your own notes;
# 4. say precisely when keyword search beats embeddings (it often does).

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

# %% [markdown]
# ## 8.1 · What an embedding is
#
# An embedding model maps a piece of text to a fixed-length vector — for
# Qwen3-Embedding-0.6B, 1024 numbers. Texts that *mean* similar things land
# near each other, even with no words in common.
#
# This is different from the chat model:
#
# | | Chat model | Embedding model |
# |---|---|---|
# | Input | tokens | tokens |
# | Output | next-token distribution | one vector for the whole text |
# | Used for | generating | comparing, searching, clustering |
#
# ```bash
# pip install sentence-transformers
# ```

# %%
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"],
                   check=True)
    from sentence_transformers import SentenceTransformer

from qwen_workshop.rag import Embedder

# Downloads ~1.2 GB on first run.
embedder = Embedder("Qwen/Qwen3-Embedding-0.6B")
print(f"model     : {embedder.model_name}")
print(f"device    : {embedder.device}")
print(f"dimensions: {embedder.dim}")

# %%
vector = embedder.encode_documents(["A cat sat on the mat."])[0]
print(f"shape : {vector.shape}")
print(f"first 8 values: {np.round(vector[:8], 4)}")
print(f"L2 norm: {np.linalg.norm(vector):.4f}  (we normalise, so it is 1.0)")

# %% [markdown]
# ## 8.2 · Cosine similarity
#
# For normalised vectors, similarity is just the dot product:
#
# $$\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} = a \cdot b$$
#
# It ranges from −1 (opposite) to 1 (identical). In practice, embedding models
# rarely produce negatives; treat the useful range as roughly 0.3–1.0.

# %%
sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",          # same meaning, no shared content words
    "My dog loves to play fetch.",          # same topic (pets), different meaning
    "The stock market closed lower today.", # unrelated
    "Мысық кілемде отырды.",                # the first sentence, in Kazakh
]

vectors = embedder.encode_documents(sentences)
similarity = vectors @ vectors.T

print("Similarity matrix:\n")
print("     " + "".join(f"{i:>7}" for i in range(len(sentences))))
for i, row in enumerate(similarity):
    print(f"{i:>3}  " + "".join(f"{v:>7.3f}" for v in row))
print()
for i, sentence in enumerate(sentences):
    print(f"{i}: {sentence}")

# %% [markdown]
# **Read row 0.** Sentence 1 ("a feline rested on the rug") should score high
# despite sharing no content words with sentence 0 — that is semantic matching,
# and it is what keyword search cannot do. Sentence 4, the Kazakh translation,
# should also score well: Qwen3-Embedding is multilingual, so you can search
# English notes with a Kazakh query.
#
# > **Exercise 1.** Add a sentence that is *lexically* similar but *semantically*
# > opposite, e.g. "The cat did not sit on the mat." How well does the model
# > separate them? (Most embedding models handle negation poorly. This is a
# > known, important limitation.)

# %% [markdown]
# ## 8.3 · Queries are encoded differently from documents
#
# Qwen3-Embedding is **instruction-aware**. Queries should carry a task
# instruction; documents should not. Skipping this costs accuracy for free.

# %%
documents = [
    "Rinse basmati rice until the water runs clear, then use 1.5 cups water per cup of rice.",
    "The ROBT613 midterm is on 14 March at 09:00 in room 3.204.",
    "Internet costs 6,000 KZT monthly, charged on the 5th.",
    "Bike tyres should be inflated to 4 bar; oil the chain every 300 km.",
]
query = "how much water for rice?"

doc_vectors = embedder.encode_documents(documents)

plain = embedder.model.encode([query], normalize_embeddings=True)[0]
instructed = embedder.encode_queries([query])[0]

print(f"{'document':<48}{'plain':>9}{'instructed':>12}")
print("-" * 69)
for doc, p, q in zip(documents, doc_vectors @ plain, doc_vectors @ instructed):
    print(f"{doc[:46]:<48}{p:>9.3f}{q:>12.3f}")

# %% [markdown]
# Both should rank the rice document first. What usually improves is the
# **margin** between the right answer and the rest — which is what makes
# retrieval robust when your corpus grows.

# %% [markdown]
# ## 8.4 · Build a search engine over the sample notes

# %%
from qwen_workshop.rag import VectorIndex, load_corpus

chunks = load_corpus(REPO_ROOT / "data" / "notes", max_chars=700, overlap=120)
print(f"{len(chunks)} chunks from {len({c.source for c in chunks})} files")
for chunk in chunks[:3]:
    print(f"  {chunk.citation:<16}{len(chunk):>4} chars  {chunk.text[:58]!r}")

# %%
chunk_vectors = embedder.encode_documents([c.text for c in chunks])
index = VectorIndex(chunks, chunk_vectors)
print(f"indexed {len(index)} chunks of {chunk_vectors.shape[1]} dimensions")


def search(query: str, k: int = 3) -> None:
    hits = index.search(embedder.encode_queries([query])[0], k=k)
    print(f"\nQ: {query}")
    for rank, (chunk, score) in enumerate(hits, 1):
        snippet = " ".join(chunk.text.split())[:110]
        print(f"  {rank}. [{score:.3f}] {chunk.citation:<16}{snippet}...")


search("when is my exam?")
search("how do I cook lentils?")
search("what do I spend money on?")
search("bike maintenance")

# %% [markdown]
# > **Exercise 2.** Replace `data/notes/` with a folder of your own notes,
# > lecture slides converted to text, or documentation. Search it. This is the
# > moment the workshop stops being an exercise.

# %% [markdown]
# ## 8.5 · When embeddings fail
#
# Semantic search is not universally better. It is reliably *worse* at:
#
# - **exact identifiers** — "invoice #88213", "ROBT613", a function name;
# - **negation** — "notes that do *not* mention the exam";
# - **numbers and dates** — "spending over 5,000 KZT";
# - **rare proper nouns** the model never saw in training.
#
# Watch it fail:

# %%
search("invoice 88213")   # no such note exists; look at how confident the scores still are

# %% [markdown]
# **Notice: it returns results anyway, with respectable scores.** A vector
# index always returns its `k` nearest neighbours — "nearest" does not mean
# "relevant". If you feed those to a model without a score threshold, you have
# built a hallucination machine.
#
# Two fixes, both used in notebook 09:
#
# 1. **A score threshold.** Below ~0.4, treat it as "nothing found".
# 2. **Hybrid search.** Combine embeddings with keyword (BM25) scoring.

# %%
def hybrid_search(query: str, k: int = 3, alpha: float = 0.6) -> list:
    """Blend semantic similarity with simple keyword overlap.

    alpha weights the semantic half. A real system would use BM25 for the
    keyword half; word overlap is enough to demonstrate the principle.
    """
    semantic = index.search(embedder.encode_queries([query])[0], k=len(index))
    query_words = {w.lower().strip(".,!?#") for w in query.split() if len(w) > 2}

    scored = []
    for chunk, sem_score in semantic:
        chunk_words = {w.lower().strip(".,!?#") for w in chunk.text.split()}
        overlap = len(query_words & chunk_words) / max(len(query_words), 1)
        scored.append((chunk, alpha * sem_score + (1 - alpha) * overlap, sem_score, overlap))

    scored.sort(key=lambda row: -row[1])
    return scored[:k]


for query in ("invoice 88213", "when is my exam?", "ROBT613"):
    print(f"\nQ: {query}")
    for chunk, blended, sem, kw in hybrid_search(query):
        print(f"  [{blended:.3f}] (sem {sem:.3f}, kw {kw:.2f}) {chunk.citation:<14}"
              f"{' '.join(chunk.text.split())[:70]}...")

# %% [markdown]
# For "ROBT613" the keyword component should pull the exam note up decisively.
# This is why production search is almost always hybrid.

# %% [markdown]
# ## 8.6 · Persist the index
#
# Embedding is the slow part. Do it once, save, reload instantly.

# %%
index_path = REPO_ROOT / "data" / ".index" / "notes"
index.save(index_path)

reloaded = VectorIndex.load(index_path)
print(f"saved and reloaded {len(reloaded)} chunks")
print("files:", [p.name for p in index_path.parent.iterdir()])

# %% [markdown]
# For bigger collections:
#
# | Scale | Use |
# |---|---|
# | < 10k chunks | NumPy, like this. Exact, zero dependencies. |
# | 10k – 1M | [FAISS](https://faiss.ai) — approximate, very fast |
# | 1M+, with filters and persistence | Qdrant, Chroma, pgvector, LanceDB |
#
# Do not reach for a vector database before you need one. A NumPy array you
# understand beats a service you do not.

# %% [markdown]
# ## 8.7 · Bonus: clustering your notes
#
# Embeddings are useful beyond search. Here we group chunks by similarity —
# no labels, no training.

# %%
from collections import defaultdict

similarity_matrix = chunk_vectors @ chunk_vectors.T
np.fill_diagonal(similarity_matrix, 0)

groups = defaultdict(list)
for i, chunk in enumerate(chunks):
    nearest = int(np.argmax(similarity_matrix[i]))
    key = tuple(sorted((chunk.source, chunks[nearest].source)))
    groups[key].append(chunk.citation)

print("Each chunk's nearest neighbour, grouped by source pair:\n")
for (a, b), members in sorted(groups.items()):
    link = f"{a} <-> {b}" if a != b else f"within {a}"
    print(f"  {link:<34}{len(members)} chunk(s)")

# %% [markdown]
# Most chunks should pair within their own file — a sanity check that the
# embeddings capture topic. Cross-file links are often genuinely interesting
# (the budget note and the bills note, for instance).
#
# ## Checkpoint
#
# 1. What does an embedding model output, and how does it differ from a chat model?
# 2. Why encode queries and documents differently?
# 3. Give three query types where keyword search beats embeddings.
# 4. Why is "the index always returns k results" dangerous?
#
# ➡️ **Next:** [`09_rag_over_your_notes.ipynb`](09_rag_over_your_notes.ipynb) —
# connect retrieval to generation, and measure whether it works.
