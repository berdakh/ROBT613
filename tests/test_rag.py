"""Chunking, retrieval maths and citation checking - all without a model."""

from __future__ import annotations

import pytest

from qwen_workshop.rag import (
    Chunk,
    VectorIndex,
    build_rag_prompt,
    chunk_text,
    cited_sources,
    format_context,
    split_paragraphs,
)

np = pytest.importorskip("numpy", reason="numpy is needed for the vector index tests")


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraph number {i}. " * 12 for i in range(40))
    chunks = chunk_text(text, "doc.md", max_chars=500, overlap=80)
    assert chunks
    assert max(len(c) for c in chunks) <= 500


def test_chunks_are_indexed_and_citable():
    chunks = chunk_text("a\n\nb\n\nc", "notes.md", max_chars=3, overlap=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].citation == "notes.md#0"


def test_overlap_repeats_context_across_boundary():
    text = "\n\n".join(["FIRST " * 20, "SECOND " * 20])
    chunks = chunk_text(text, "d.md", max_chars=140, overlap=60)
    assert len(chunks) >= 2
    # Some tail of chunk 0 must reappear at the head of chunk 1.
    assert "FIRST" in chunks[1].text


def test_oversized_paragraph_is_split():
    text = "One sentence. " * 200  # single paragraph, way over the limit
    chunks = chunk_text(text, "big.md", max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 300


def test_single_monstrous_sentence_still_terminates():
    chunks = chunk_text("x" * 5000, "huge.md", max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 200


def test_heading_stays_with_its_body():
    assert split_paragraphs("## Title\n\nThe body.") == ["## Title\nThe body."]


def test_empty_document_yields_no_chunks():
    assert chunk_text("   \n\n  ", "empty.md") == []


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("hello", "d.md", max_chars=100, overlap=100)


# --------------------------------------------------------------------------
# Vector index
# --------------------------------------------------------------------------


@pytest.fixture
def index():
    chunks = [Chunk(f"text {i}", "d.md", i) for i in range(4)]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    return VectorIndex(chunks, vectors)


def test_search_ranks_by_cosine_similarity(index):
    hits = index.search(np.array([1.0, 0.0, 0.0]), k=2)
    assert [c.index for c, _ in hits] == [0, 2]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_k_larger_than_corpus_is_safe(index):
    assert len(index.search(np.array([1.0, 0.0, 0.0]), k=99)) == 4


def test_unnormalised_vectors_are_handled(index):
    """Scaling a query must not change the ranking."""
    a = index.search(np.array([1.0, 0.0, 0.0]), k=3)
    b = index.search(np.array([50.0, 0.0, 0.0]), k=3)
    assert [c.index for c, _ in a] == [c.index for c, _ in b]


def test_index_rejects_length_mismatch():
    with pytest.raises(ValueError):
        VectorIndex([Chunk("a", "d.md", 0)], np.zeros((2, 3), dtype="float32"))


def test_index_round_trips_through_disk(index, tmp_path):
    index.save(tmp_path / "idx")
    reloaded = VectorIndex.load(tmp_path / "idx")
    assert len(reloaded) == len(index)
    assert reloaded.chunks[2].citation == "d.md#2"
    assert reloaded.search(np.array([0.0, 1.0, 0.0]), k=1)[0][0].index == 1


# --------------------------------------------------------------------------
# Prompt construction and citation checking
# --------------------------------------------------------------------------


def test_context_carries_citation_labels():
    hits = [(Chunk("Pasta needs salt.", "recipes.md", 3), 0.9)]
    assert "[recipes.md#3]" in format_context(hits)


def test_rag_prompt_includes_refusal_instruction():
    prompt = build_rag_prompt("Why?", [(Chunk("Because.", "a.md", 0), 0.5)])
    assert "cannot answer that from the provided documents" in prompt
    assert "Why?" in prompt


def test_cited_sources_extracts_only_valid_citations():
    answer = "Salt matters [recipes.md#3] and so does time [notes/b.md#12]. Ignore [TODO]."
    assert cited_sources(answer) == ["notes/b.md#12", "recipes.md#3"]


def test_citations_can_be_checked_against_retrieved_chunks():
    """The cheap hallucination check used in notebook 09."""
    retrieved = {"recipes.md#3"}
    invented = set(cited_sources("See [recipes.md#3] and [recipes.md#99].")) - retrieved
    assert invented == {"recipes.md#99"}
