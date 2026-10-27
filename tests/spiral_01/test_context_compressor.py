"""Tests for ContextCompressor."""

from __future__ import annotations

from unittest.mock import MagicMock

from spiral_01_chunks.context_compressor import (
    ContextCompressor,
    _split_sentences,
)
from spiral_01_chunks.retriever import RetrievalResult


def _make_result(text: str, source: str = "test.md", similarity: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        source=source,
        chunk_index=0,
        similarity=similarity,
        metadata={"strategy": "recursive"},
    )


def _mock_embedder(query_sim_map: dict[str, float]) -> MagicMock:
    """Build a mock embedder where cosine_similarity is controlled per sentence."""
    mock = MagicMock()
    # embed_text returns a unique "tag" per text so we can look it up
    mock.embed_text.side_effect = lambda text: [hash(text) % 1000 / 1000.0]
    mock.embed_batch.side_effect = lambda texts: [[hash(t) % 1000 / 1000.0] for t in texts]

    import spiral_01_chunks.context_compressor as mod

    original_cosine = mod.Embedder.cosine_similarity

    def fake_cosine(a, b):
        # We can't easily map back — use a real implementation
        return original_cosine(a, b)

    return mock


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        text = "First sentence. Second sentence."
        parts = _split_sentences(text)
        assert len(parts) == 2
        assert parts[0] == "First sentence."

    def test_splits_on_question_mark(self) -> None:
        parts = _split_sentences("What is RAG? It retrieves context.")
        assert len(parts) == 2

    def test_strips_empty(self) -> None:
        parts = _split_sentences("  ")
        assert parts == []

    def test_single_sentence(self) -> None:
        parts = _split_sentences("Just one sentence.")
        assert parts == ["Just one sentence."]


class TestContextCompressor:
    def test_returns_empty_for_empty_input(self) -> None:
        compressor = ContextCompressor()
        result = compressor.compress([], "query")
        assert result == []

    def test_preserves_source_and_metadata(self) -> None:
        """Compression must not change source, similarity, chunk_index, metadata."""
        chunk = _make_result(
            "FastAPI is a framework. Django is slower.",
            source="fastapi.md",
            similarity=0.75,
        )
        chunk.chunk_index = 3
        chunk.metadata = {"key": "value"}

        # Use real embedder with low threshold to keep everything
        compressor = ContextCompressor(threshold=0.0)
        results = compressor.compress([chunk], "FastAPI framework")

        assert len(results) == 1
        assert results[0].source == "fastapi.md"
        assert results[0].similarity == 0.75
        assert results[0].chunk_index == 3
        assert results[0].metadata == {"key": "value"}

    def test_with_threshold_zero_keeps_all_sentences(self) -> None:
        """threshold=0.0 keeps everything — no sentence has similarity < 0."""
        chunk = _make_result("First sentence. Second sentence. Third sentence.")
        compressor = ContextCompressor(threshold=0.0)
        results = compressor.compress([chunk], "anything")
        assert len(results) == 1
        # All three sentences must appear in the compressed text
        for sentence in ["First sentence.", "Second sentence.", "Third sentence."]:
            assert sentence in results[0].text

    def test_with_high_threshold_keeps_at_least_one_sentence(self) -> None:
        """Even with threshold=1.0, at least one sentence must be returned."""
        chunk = _make_result("This is about Python. Django is a web framework.")
        compressor = ContextCompressor(threshold=1.0)
        results = compressor.compress([chunk], "unrelated topic xyz")
        assert len(results) == 1
        assert len(results[0].text) > 0

    def test_chunk_with_no_sentences_passes_through(self) -> None:
        """A chunk with empty/whitespace text passes through unchanged."""
        chunk = _make_result("   ")
        chunk_copy_text = chunk.text
        compressor = ContextCompressor(threshold=0.3)
        results = compressor.compress([chunk], "query")
        assert len(results) == 1
        assert results[0].text == chunk_copy_text

    def test_multiple_chunks_all_processed(self) -> None:
        chunks = [
            _make_result("FastAPI is fast. It uses type hints.", similarity=0.9),
            _make_result("Django is batteries included. Flask is lightweight.", similarity=0.7),
        ]
        compressor = ContextCompressor(threshold=0.0)
        results = compressor.compress(chunks, "Python web frameworks")
        assert len(results) == 2

    def test_compressed_text_is_shorter_or_equal(self) -> None:
        """Compressed text must not be longer than original (no injection)."""
        chunk = _make_result(
            "FastAPI is modern. It is fast. It is based on type hints. "
            "Django is old. Flask is micro. They are all web frameworks."
        )
        compressor = ContextCompressor(threshold=0.3)
        results = compressor.compress([chunk], "FastAPI type hints")
        assert len(results[0].text) <= len(chunk.text)
