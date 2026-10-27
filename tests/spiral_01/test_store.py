"""Tests for Spiral 01 — VectorStore module.

Key validations:
- Can add chunks and retrieve them
- Query returns results sorted by similarity
- Stats report correct counts
- Clear removes all documents
"""

from spiral_01_chunks.chunker import Chunk
from spiral_01_chunks.store import VectorStore


class TestVectorStore:
    def test_empty_store(self, temp_store: VectorStore):
        assert temp_store.count == 0

    def test_add_documents(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        added = temp_store.add_documents(sample_chunks)
        assert added == 4
        assert temp_store.count == 4

    def test_add_empty_list(self, temp_store: VectorStore):
        added = temp_store.add_documents([])
        assert added == 0

    def test_query_empty_store_returns_empty(self, temp_store: VectorStore):
        """Query on empty store should return empty list, not crash."""
        results = temp_store.query("anything")
        assert results == []

    def test_query_returns_results(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        temp_store.add_documents(sample_chunks)
        results = temp_store.query("What is FastAPI?", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_query_has_required_fields(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        temp_store.add_documents(sample_chunks)
        results = temp_store.query("FastAPI framework")
        assert len(results) > 0
        result = results[0]
        assert "text" in result
        assert "similarity" in result
        assert "metadata" in result

    def test_query_relevance(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        temp_store.add_documents(sample_chunks)
        results = temp_store.query("FastAPI web framework Python")

        # FastAPI chunk should be the most relevant
        assert len(results) > 0
        top_result = results[0]
        assert "FastAPI" in top_result["text"]

    def test_query_irrelevant_topic(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        temp_store.add_documents(sample_chunks)
        results = temp_store.query("machine learning artificial intelligence")

        # ML chunk should rank higher than web framework chunks
        assert len(results) > 0
        top_result = results[0]
        assert "learning" in top_result["text"].lower() or "machine" in top_result["text"].lower()

    def test_stats(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        temp_store.add_documents(sample_chunks)
        stats = temp_store.stats()
        assert stats["total_chunks"] == 4
        assert stats["similarity_metric"] == "cosine"
        assert "embedding_model" in stats

    def test_clear(self, temp_store: VectorStore, sample_chunks: list[Chunk]):
        temp_store.add_documents(sample_chunks)
        assert temp_store.count == 4
        temp_store.clear()
        assert temp_store.count == 0
