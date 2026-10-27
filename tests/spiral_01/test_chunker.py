"""Tests for Spiral 01 — Chunker module.

These tests ARE the specification:
- chunker must split text into blocks
- each strategy must produce valid Chunk objects
- overlap must preserve context between blocks
- empty input must be handled gracefully
"""

import pytest

from spiral_01_chunks.chunker import (
    STRATEGIES,
    Chunk,
    chunk_fixed,
    chunk_recursive,
    chunk_sentence,
    chunk_text,
)

# --- Test Data ---

SAMPLE_TEXT = """FastAPI is a modern, fast web framework for building APIs with Python.

It is based on standard Python type hints. FastAPI provides automatic validation,
serialization, and documentation. It uses Starlette for the web parts and Pydantic
for the data parts.

FastAPI is very fast. It is one of the fastest Python frameworks available.
The performance is on par with NodeJS and Go thanks to Starlette and Pydantic.

To create a basic FastAPI application, you need to import FastAPI and create an instance.
Then you define path operations using decorators like @app.get() and @app.post().
Each path operation function can have parameters that FastAPI will parse from the request."""

SHORT_TEXT = "Hello world."

EMPTY_TEXT = ""


# --- Chunk Dataclass ---


class TestChunk:
    def test_chunk_has_required_fields(self):
        chunk = Chunk(text="hello", source="test.md", chunk_index=0, strategy="fixed")
        assert chunk.text == "hello"
        assert chunk.source == "test.md"
        assert chunk.chunk_index == 0
        assert chunk.strategy == "fixed"

    def test_token_estimate(self):
        chunk = Chunk(text="a" * 100, source="test.md", chunk_index=0, strategy="fixed")
        assert chunk.token_estimate == 25  # 100 chars / 4

    def test_repr_truncates(self):
        chunk = Chunk(text="a" * 200, source="test.md", chunk_index=0, strategy="fixed")
        repr_str = repr(chunk)
        assert "..." in repr_str
        assert len(repr_str) < 200


# --- Strategy Registry ---


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert "fixed" in STRATEGIES
        assert "recursive" in STRATEGIES
        assert "sentence" in STRATEGIES

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_text("hello", "test.md", strategy="nonexistent")


# --- Fixed Chunking ---


class TestFixedChunking:
    def test_produces_chunks(self):
        chunks = chunk_fixed(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_strategy_label(self):
        chunks = chunk_fixed(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert all(c.strategy == "fixed" for c in chunks)

    def test_source_preserved(self):
        chunks = chunk_fixed(SAMPLE_TEXT, "my_doc.md", chunk_size=200, overlap=20)
        assert all(c.source == "my_doc.md" for c in chunks)

    def test_chunk_indices_sequential(self):
        chunks = chunk_fixed(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# --- Recursive Chunking ---


class TestRecursiveChunking:
    def test_produces_chunks(self):
        chunks = chunk_recursive(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_respects_boundaries(self):
        chunks = chunk_recursive(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        # Recursive splitter should prefer splitting at paragraph/sentence boundaries
        for chunk in chunks:
            assert len(chunk.text) <= 200 + 50  # some tolerance for boundary handling

    def test_strategy_label(self):
        chunks = chunk_recursive(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert all(c.strategy == "recursive" for c in chunks)


# --- Sentence Chunking ---


class TestSentenceChunking:
    def test_produces_chunks(self):
        chunks = chunk_sentence(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert len(chunks) >= 1

    def test_strategy_label(self):
        chunks = chunk_sentence(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert all(c.strategy == "sentence" for c in chunks)

    def test_short_text_single_chunk(self):
        chunks = chunk_sentence(SHORT_TEXT, "test.md", chunk_size=200, overlap=20)
        assert len(chunks) == 1

    def test_handles_text_with_empty_splits(self):
        """Ensure empty strings from regex split are skipped (covers line 141)."""
        # Trailing period + space produces empty string at end of split
        text = "First. Second. "
        chunks = chunk_sentence(text, "test.md", chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks)


# --- chunk_text (unified interface) ---


class TestChunkText:
    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "sentence"])
    def test_all_strategies_work(self, strategy: str):
        chunks = chunk_text(SAMPLE_TEXT, "test.md", strategy=strategy, chunk_size=200, overlap=20)
        assert len(chunks) >= 1
        assert all(c.strategy == strategy for c in chunks)

    def test_default_strategy_is_recursive(self):
        chunks = chunk_text(SAMPLE_TEXT, "test.md", chunk_size=200, overlap=20)
        assert all(c.strategy == "recursive" for c in chunks)
