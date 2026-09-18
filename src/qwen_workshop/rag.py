"""Retrieval-Augmented Generation, built from parts you can see.

RAG is three steps and one discipline:

1. **Chunk**   - split documents into passages small enough to be specific.
2. **Embed**   - turn each passage into a vector; store them.
3. **Retrieve**- embed the question, find the nearest passages, paste them into
   the prompt.

The discipline: *the model must answer only from the retrieved text, and must
say so when the text does not contain the answer.* A RAG system that quietly
falls back on the model's own memory is worse than no RAG at all, because you
can no longer tell grounded answers from invented ones.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_EMBEDDING_MODEL


@dataclass
class Chunk:
    """A passage of text plus where it came from."""

    text: str
    source: str
    index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.index}"

    def __len__(self) -> int:
        return len(self.text)


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping markdown headings attached to their body."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    merged: list[str] = []
    for block in blocks:
        # A lone heading belongs with the paragraph that follows it.
        if merged and re.fullmatch(r"#{1,6} .+", merged[-1]):
            merged[-1] = merged[-1] + "\n" + block
        else:
            merged.append(block)
    return merged


def chunk_text(
    text: str,
    source: str = "document",
    *,
    max_chars: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Split text into overlapping chunks that respect paragraph boundaries.

    Why character counts and not tokens? Because it is transparent and close
    enough: ~4 characters per token for English, ~2-3 for Russian or Kazakh.
    Why overlap? So a sentence that straddles a boundary still appears whole
    in at least one chunk.

    Args:
        text: the raw document.
        source: label used in citations.
        max_chars: soft upper bound per chunk.
        overlap: characters of tail from the previous chunk to repeat.

    Returns:
        A list of :class:`Chunk` in document order.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(Chunk(text=buffer.strip(), source=source, index=len(chunks)))
        buffer = ""

    for paragraph in split_paragraphs(text):
        # A single oversized paragraph gets hard-split by sentence.
        if len(paragraph) > max_chars:
            flush()
            for piece in _split_long(paragraph, max_chars, overlap):
                chunks.append(Chunk(text=piece, source=source, index=len(chunks)))
            continue

        if len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        else:
            tail = buffer[-overlap:] if overlap else ""
            flush()
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    flush()
    return chunks


def _split_long(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    """Hard-split an over-long paragraph, preferring sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    buffer = ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # a single monstrous sentence
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars - overlap :]
        if len(buffer) + len(sentence) + 1 <= max_chars:
            buffer = f"{buffer} {sentence}".strip()
        else:
            pieces.append(buffer.strip())
            buffer = (buffer[-overlap:] + " " + sentence).strip() if overlap else sentence
    if buffer.strip():
        pieces.append(buffer.strip())
    return [p for p in pieces if p]


def load_corpus(
    folder: str | Path,
    *,
    patterns: tuple[str, ...] = ("*.md", "*.txt"),
    max_chars: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Read every matching file in a folder (recursively) and chunk it."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Corpus folder not found: {folder}")

    chunks: list[Chunk] = []
    for pattern in patterns:
        for path in sorted(folder.rglob(pattern)):
            text = path.read_text(encoding="utf-8", errors="replace")
            # Strip Jekyll front matter so it does not pollute retrieval.
            text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
            local = chunk_text(text, source=str(path.relative_to(folder)),
                               max_chars=max_chars, overlap=overlap)
            for chunk in local:
                chunk.metadata["path"] = str(path)
            chunks.extend(local)
    return chunks


class Embedder:
    """Wrapper around Qwen3-Embedding via sentence-transformers.

    Qwen3 embedding models are *instruction aware*: queries should be encoded
    with a task instruction, documents without one. Skipping this costs a few
    points of retrieval accuracy for free.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        **kwargs,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        from .env import pick_device

        self.model_name = model_name
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str], batch_size: int = 16, **kwargs):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64, **kwargs
        )

    def encode_queries(self, texts: list[str], **kwargs):
        # prompt_name="query" applies the model's built-in query instruction.
        try:
            return self.model.encode(
                texts, prompt_name="query", normalize_embeddings=True, **kwargs
            )
        except (ValueError, KeyError):
            return self.model.encode(texts, normalize_embeddings=True, **kwargs)


class VectorIndex:
    """A brute-force cosine-similarity index in ~20 lines of NumPy.

    For a few thousand chunks this is genuinely the right tool: exact results,
    no dependencies, no index build step. Reach for FAISS, Chroma or pgvector
    when you pass roughly 10^5 vectors or need on-disk persistence with
    filters - not before.
    """

    def __init__(self, chunks: list[Chunk], vectors) -> None:
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype="float32")
        # Normalise once so cosine similarity is a plain dot product.
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.clip(norms, 1e-12, None)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the `k` most similar chunks with their cosine scores."""
        import numpy as np

        query = np.asarray(query_vector, dtype="float32").reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self.vectors @ query
        k = min(k, len(self.chunks))
        # argpartition is O(n); full sort only on the k survivors.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path: str | Path) -> None:
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.vectors)
        payload = [
            {"text": c.text, "source": c.source, "index": c.index, "metadata": c.metadata}
            for c in self.chunks
        ]
        path.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        import numpy as np

        path = Path(path)
        vectors = np.load(path.with_suffix(".npy"))
        payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in payload]
        return cls(chunks, vectors)


RAG_PROMPT = """Answer the question using ONLY the context below.

Rules:
- Cite the source of every claim like this: [source#index].
- If the context does not contain the answer, reply exactly:
  "I cannot answer that from the provided documents."
- Do not use knowledge from outside the context, even if you are confident.

Context:
{context}

Question: {question}"""


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks with citation labels the model can copy."""
    return "\n\n".join(f"[{chunk.citation}]\n{chunk.text}" for chunk, _score in hits)


def build_rag_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    return RAG_PROMPT.format(context=format_context(hits), question=question)


def cited_sources(answer: str) -> list[str]:
    """Pull ``[file.md#3]`` style citations back out of an answer.

    Use this to check the model cited something that was actually retrieved -
    a cheap, automatic hallucination check. See notebook 09.
    """
    return sorted(set(re.findall(r"\[([^\[\]\s]+#\d+)\]", answer)))
