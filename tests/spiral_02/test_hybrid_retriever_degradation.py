"""Tests for HybridRetriever graceful degradation when Neo4j is offline.

Key validations:
- Graph failure returns vector-only results (no crash)
- Graph failure records metric via MetricsCollector
- No metrics object does not crash on degradation
"""

from unittest.mock import MagicMock, patch

from spiral_01_chunks.retriever import RetrievalResult
from spiral_02_graph.hybrid_retriever import HybridRetriever
from spiral_04_cognitive.metrics import InMemoryMetrics


def _make_vector_results() -> list[RetrievalResult]:
    """Create sample vector results for testing."""
    return [
        RetrievalResult(
            text="FastAPI is a modern web framework.",
            similarity=0.9,
            source="fastapi.md",
            chunk_index=0,
            metadata={"source": "fastapi.md"},
        ),
        RetrievalResult(
            text="Dependency Injection simplifies testing.",
            similarity=0.75,
            source="di.md",
            chunk_index=1,
            metadata={"source": "di.md"},
        ),
    ]


class TestHybridRetrieverDegradation:
    def test_graph_failure_returns_vector_only_results(self, mock_graph_store):
        """When graph search raises, retrieve() returns vector results only."""
        retriever = HybridRetriever(
            vector_store=MagicMock(),
            graph_store=mock_graph_store,
        )

        vector_results = _make_vector_results()
        with (
            patch.object(retriever._vector_retriever, "retrieve", return_value=vector_results),
            patch.object(
                retriever._graph_rag,
                "global_search",
                side_effect=ConnectionError("Neo4j is offline"),
            ),
        ):
            results = retriever.retrieve("What is FastAPI?")

        assert len(results) == 2
        assert all(r.source == "vector" for r in results)
        assert results[0].similarity == 0.9

    def test_graph_failure_records_metric(self, mock_graph_store):
        """When graph search fails, record_cache_miss('graph_degradation') is called."""
        metrics = InMemoryMetrics()
        retriever = HybridRetriever(
            vector_store=MagicMock(),
            graph_store=mock_graph_store,
            metrics=metrics,
        )

        vector_results = _make_vector_results()
        with (
            patch.object(retriever._vector_retriever, "retrieve", return_value=vector_results),
            patch.object(
                retriever._graph_rag,
                "global_search",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            retriever.retrieve("How does middleware work?")

        summary = metrics.summary()
        assert "graph_degradation" in summary["cache"]
        assert summary["cache"]["graph_degradation"]["misses"] == 1

    def test_no_metrics_no_crash_on_degradation(self, mock_graph_store):
        """When metrics is None and graph fails, no AttributeError is raised."""
        retriever = HybridRetriever(
            vector_store=MagicMock(),
            graph_store=mock_graph_store,
            metrics=None,
        )

        vector_results = _make_vector_results()
        with (
            patch.object(retriever._vector_retriever, "retrieve", return_value=vector_results),
            patch.object(
                retriever._graph_rag,
                "global_search",
                side_effect=Exception("timeout"),
            ),
        ):
            results = retriever.retrieve("Tell me about Starlette")

        # Should degrade gracefully — return vector results without crash
        assert len(results) == 2
        assert all(r.source == "vector" for r in results)
