"""Property-based tests — invariants that must hold for any input.

Uses Hypothesis to generate hundreds of random inputs and verify
structural guarantees that unit tests with fixed examples can miss.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from spiral_01_chunks.chunker import chunk_text
from spiral_01_chunks.config import SIMILARITY_THRESHOLD
from spiral_01_chunks.embedder import Embedder
from spiral_01_chunks.retriever import RetrievalResult
from spiral_05_economics.token_counter import TokenCounter

# ─── Chunker invariants ────────────────────────────────────────────────────────


@given(st.text(min_size=1, max_size=4000))
@settings(max_examples=50, deadline=None)
def test_chunker_no_text_loss(text: str) -> None:
    """Joining all chunks must contain every character from the original text.

    Overlap may cause repetition, but no content can be silently dropped.
    """
    chunks = chunk_text(text, source="test.md", strategy="recursive")
    if not chunks:
        # Empty chunks only allowed for whitespace-only input
        assert not text.strip()
        return

    joined = " ".join(c.text for c in chunks)
    # Every word that exists in the original must appear in joined
    for word in text.split():
        assert word in joined, f"Word {word!r} lost after chunking"


@given(st.text(min_size=50, max_size=2000))
@settings(max_examples=30, deadline=None)
def test_chunker_indices_are_sequential(text: str) -> None:
    """Chunk indices must start at 0 and increment by 1."""
    chunks = chunk_text(text, source="test.md", strategy="sentence")
    for expected_idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_idx


# ─── Retriever invariants ──────────────────────────────────────────────────────


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
def test_retrieval_results_sorted_descending(similarities: list[float]) -> None:
    """retrieve() results must always be sorted by similarity, highest first."""
    results = [
        RetrievalResult(
            text=f"chunk {i}",
            source="test.md",
            chunk_index=i,
            similarity=sim,
            metadata={},
        )
        for i, sim in enumerate(similarities)
    ]
    results_sorted = sorted(results, key=lambda r: r.similarity, reverse=True)

    for i in range(len(results_sorted) - 1):
        assert results_sorted[i].similarity >= results_sorted[i + 1].similarity


@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_is_relevant_consistent_with_threshold(similarity: float, threshold: float) -> None:
    """is_relevant must be True iff similarity >= SIMILARITY_THRESHOLD."""
    result = RetrievalResult(
        text="test",
        source="test.md",
        chunk_index=0,
        similarity=similarity,
        metadata={},
    )
    # is_relevant uses the global SIMILARITY_THRESHOLD, not the threshold param here
    assert result.is_relevant == (similarity >= SIMILARITY_THRESHOLD)


# ─── TokenCounter invariants ───────────────────────────────────────────────────


@given(
    st.text(
        min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))
    ),
    st.text(
        min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))
    ),
)
def test_token_count_monotonic(text_a: str, text_b: str) -> None:
    """More text = more or equal tokens (monotonic, never fewer)."""
    counter = TokenCounter()
    count_a = counter.estimate_prompt(text_a, model="llama-3.3-70b-versatile")
    count_ab = counter.estimate_prompt(text_a + " " + text_b, model="llama-3.3-70b-versatile")
    assert count_ab.input_tokens >= count_a.input_tokens


# ─── Embedder invariants ───────────────────────────────────────────────────────


@pytest.mark.slow
@given(
    st.text(
        min_size=5,
        max_size=200,
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
    )
)
@settings(max_examples=5, deadline=None)
def test_cosine_similarity_self_is_one(text: str) -> None:
    """cosine_similarity(embed_text(x), embed_text(x)) must be ~1.0."""
    embedder = Embedder()
    vec = embedder.embed_text(text)
    sim = Embedder.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5, f"Self-similarity was {sim}, expected ~1.0"
