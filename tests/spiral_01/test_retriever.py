"""Tests for Spiral 01 — Retriever module.

Key validations:
- Retriever transforms raw store results into RetrievalResult objects
- Threshold filtering excludes low-similarity results
- Results are sorted by similarity (descending)
- format_context respects token limits
- retrieve_with_context returns correct metadata
"""

from spiral_01_chunks.retriever import RetrievalResult, Retriever

# --- RetrievalResult ---


class TestRetrievalResult:
    def test_has_required_fields(self):
        result = RetrievalResult(
            text="hello",
            source="test.md",
            chunk_index=0,
            similarity=0.85,
            metadata={},
        )
        assert result.text == "hello"
        assert result.source == "test.md"
        assert result.chunk_index == 0
        assert result.similarity == 0.85

    def test_is_relevant_above_threshold(self):
        result = RetrievalResult(
            text="hello", source="test.md", chunk_index=0, similarity=0.9, metadata={}
        )
        assert result.is_relevant is True

    def test_is_not_relevant_below_threshold(self):
        result = RetrievalResult(
            text="hello", source="test.md", chunk_index=0, similarity=0.1, metadata={}
        )
        assert result.is_relevant is False

    def test_repr_truncates(self):
        result = RetrievalResult(
            text="a" * 200, source="test.md", chunk_index=0, similarity=0.5, metadata={}
        )
        repr_str = repr(result)
        assert "sim=0.500" in repr_str
        assert "..." in repr_str


# --- Retriever.retrieve ---


class TestRetrieverRetrieve:
    def test_returns_retrieval_results(self, retriever):
        results = retriever.retrieve("FastAPI web framework")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_sorted_by_similarity_descending(self, retriever):
        results = retriever.retrieve("Python web framework", top_k=5)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_respects_top_k(self, retriever):
        results = retriever.retrieve("Python", top_k=2)
        assert len(results) <= 2

    def test_threshold_filters_low_similarity(self, retriever):
        # With a very high threshold, should return fewer (or no) results
        results_strict = retriever.retrieve("FastAPI", threshold=0.99)
        results_loose = retriever.retrieve("FastAPI", threshold=0.01)
        assert len(results_strict) <= len(results_loose)

    def test_result_has_source_metadata(self, retriever):
        results = retriever.retrieve("FastAPI")
        assert len(results) > 0
        top = results[0]
        assert top.source != "unknown"
        assert top.source.endswith(".md")

    def test_result_has_chunk_index(self, retriever):
        results = retriever.retrieve("FastAPI")
        assert len(results) > 0
        assert isinstance(results[0].chunk_index, int)

    def test_empty_query_returns_results(self, retriever):
        # Even an empty-ish query should not crash
        results = retriever.retrieve("", top_k=2)
        assert isinstance(results, list)


# --- Retriever.retrieve_with_context ---


class TestRetrieverWithContext:
    def test_returns_dict_with_required_keys(self, retriever):
        ctx = retriever.retrieve_with_context("FastAPI")
        assert "query" in ctx
        assert "top_k" in ctx
        assert "threshold" in ctx
        assert "total_results" in ctx
        assert "results" in ctx
        assert "avg_similarity" in ctx
        assert "store_stats" in ctx

    def test_query_preserved(self, retriever):
        ctx = retriever.retrieve_with_context("Docker containers")
        assert ctx["query"] == "Docker containers"

    def test_avg_similarity_computed(self, retriever):
        ctx = retriever.retrieve_with_context("Python framework", top_k=3)
        if ctx["total_results"] > 0:
            assert 0.0 < ctx["avg_similarity"] <= 1.0

    def test_store_stats_included(self, retriever):
        ctx = retriever.retrieve_with_context("FastAPI")
        stats = ctx["store_stats"]
        assert stats["total_chunks"] == 5
        assert stats["similarity_metric"] == "cosine"


# --- Retriever.format_context ---


class TestFormatContext:
    def test_formats_results_into_string(self, retriever):
        results = retriever.retrieve("FastAPI", top_k=3)
        context = Retriever.format_context(results)
        assert isinstance(context, str)
        assert len(context) > 0

    def test_includes_source_attribution(self, retriever):
        results = retriever.retrieve("FastAPI", top_k=1)
        context = Retriever.format_context(results)
        assert "[Source:" in context
        assert "Similarity:" in context

    def test_respects_max_tokens_limit(self, retriever):
        results = retriever.retrieve("Python", top_k=5)
        # With a very small token limit, should truncate
        context_small = Retriever.format_context(results, max_tokens=10)
        context_large = Retriever.format_context(results, max_tokens=20000)
        assert len(context_small) <= len(context_large)

    def test_empty_results_returns_empty_string(self):
        context = Retriever.format_context([])
        assert context == ""

    def test_separator_between_chunks(self, retriever):
        results = retriever.retrieve("Python", top_k=5)
        if len(results) >= 2:
            context = Retriever.format_context(results, max_tokens=20000)
            assert "---" in context


# --- Retriever.deduplicate ---


class TestDeduplicate:
    def test_empty_list_returns_empty(self):
        assert Retriever.deduplicate([]) == []

    def test_single_result_preserved(self):
        result = RetrievalResult(
            text="FastAPI is a web framework",
            source="test.md",
            chunk_index=0,
            similarity=0.9,
            metadata={},
        )
        deduped = Retriever.deduplicate([result])
        assert len(deduped) == 1
        assert deduped[0] is result

    def test_removes_near_duplicates(self, retriever):
        """Two chunks with nearly identical text should be deduplicated."""
        results = [
            RetrievalResult(
                text="FastAPI is a modern web framework for building APIs with Python.",
                source="a.md",
                chunk_index=0,
                similarity=0.95,
                metadata={},
            ),
            RetrievalResult(
                text="FastAPI is a modern web framework for building APIs with Python based on type hints.",
                source="b.md",
                chunk_index=0,
                similarity=0.90,
                metadata={},
            ),
        ]
        deduped = Retriever.deduplicate(results, threshold=0.85)
        assert len(deduped) == 1
        assert deduped[0].source == "a.md"  # keeps highest similarity

    def test_keeps_distinct_chunks(self, retriever):
        """Semantically different chunks should all be kept."""
        results = [
            RetrievalResult(
                text="FastAPI is a modern web framework for building APIs.",
                source="fastapi.md",
                chunk_index=0,
                similarity=0.9,
                metadata={},
            ),
            RetrievalResult(
                text="Machine learning uses neural networks to learn from data.",
                source="ml.md",
                chunk_index=0,
                similarity=0.5,
                metadata={},
            ),
        ]
        deduped = Retriever.deduplicate(results, threshold=0.85)
        assert len(deduped) == 2

    def test_preserves_order(self, retriever):
        """Deduplicated results maintain original order."""
        results = [
            RetrievalResult(
                text="FastAPI is a web framework.",
                source="a.md",
                chunk_index=0,
                similarity=0.9,
                metadata={},
            ),
            RetrievalResult(
                text="Docker packages applications.",
                source="b.md",
                chunk_index=0,
                similarity=0.7,
                metadata={},
            ),
            RetrievalResult(
                text="Python is a programming language.",
                source="c.md",
                chunk_index=0,
                similarity=0.5,
                metadata={},
            ),
        ]
        deduped = Retriever.deduplicate(results, threshold=0.85)
        similarities = [r.similarity for r in deduped]
        assert similarities == sorted(similarities, reverse=True)
