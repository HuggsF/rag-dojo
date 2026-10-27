"""Tests for Spiral 02 — Hybrid Retriever module.

Key validations:
- Merge combines vector and graph results
- Combined scoring uses weights
- Deduplication detects overlap
- Format context includes graph path info
- Traversal info summary
"""

from unittest.mock import MagicMock

import pytest

from spiral_01_chunks.retriever import RetrievalResult
from spiral_02_graph.graph_rag import GraphResult
from spiral_02_graph.hybrid_retriever import HybridResult, HybridRetriever


class TestHybridResult:
    def test_basic_creation(self):
        r = HybridResult(
            text="Test",
            source="vector",
            similarity=0.8,
            graph_score=0.0,
            combined_score=0.4,
        )
        assert r.is_relevant  # 0.4 >= 0.3

    def test_irrelevant(self):
        r = HybridResult(
            text="Test",
            source="graph",
            similarity=0.0,
            graph_score=0.1,
            combined_score=0.05,
        )
        assert not r.is_relevant

    def test_repr(self):
        r = HybridResult(
            text="FastAPI is great",
            source="hybrid",
            similarity=0.8,
            graph_score=0.7,
            combined_score=0.75,
        )
        assert "0.75" in repr(r)
        assert "hybrid" in repr(r)


class TestMergeResults:
    def test_vector_only(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vector_weight = 0.5
        retriever._graph_weight = 0.5

        vector_results = [
            RetrievalResult(
                text="FastAPI is a web framework",
                source="fastapi.md",
                chunk_index=0,
                similarity=0.9,
                metadata={},
            ),
        ]
        merged = retriever._merge_results(vector_results, [])
        assert len(merged) == 1
        assert merged[0].source == "vector"
        assert merged[0].combined_score == pytest.approx(0.45, abs=0.01)

    def test_graph_only(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vector_weight = 0.5
        retriever._graph_weight = 0.5

        graph_results = [
            GraphResult(
                text="A -> B -> C",
                entities=["A", "B", "C"],
                path_length=2,
                relation_types=["USES", "DEPENDS_ON"],
                score=0.8,
            ),
        ]
        merged = retriever._merge_results([], graph_results)
        assert len(merged) == 1
        assert merged[0].source == "graph"
        assert merged[0].combined_score == pytest.approx(0.4, abs=0.01)

    def test_overlap_creates_hybrid(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vector_weight = 0.5
        retriever._graph_weight = 0.5

        vector_results = [
            RetrievalResult(
                text="FastAPI is a modern web framework for building APIs.",
                source="fastapi.md",
                chunk_index=0,
                similarity=0.85,
                metadata={},
            ),
        ]
        graph_results = [
            GraphResult(
                text="FastAPI -> uses -> Starlette",
                entities=["FastAPI", "Starlette"],
                path_length=1,
                relation_types=["USES"],
                score=0.75,
            ),
        ]
        merged = retriever._merge_results(vector_results, graph_results)

        # Should merge because "FastAPI" appears in vector text
        hybrid_results = [r for r in merged if r.source == "hybrid"]
        assert len(hybrid_results) == 1
        assert hybrid_results[0].similarity == 0.85
        assert hybrid_results[0].graph_score == 0.75

    def test_no_overlap_separate_results(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vector_weight = 0.5
        retriever._graph_weight = 0.5

        vector_results = [
            RetrievalResult(
                text="Python is a programming language.",
                source="python.md",
                chunk_index=0,
                similarity=0.7,
                metadata={},
            ),
        ]
        graph_results = [
            GraphResult(
                text="FastAPI -> uses -> Starlette",
                entities=["FastAPI", "Starlette"],
                path_length=1,
                relation_types=["USES"],
                score=0.6,
            ),
        ]
        merged = retriever._merge_results(vector_results, graph_results)
        assert len(merged) == 2  # separate results


class TestFormatContext:
    def test_includes_source_info(self, sample_hybrid_results):
        context = HybridRetriever.format_context(sample_hybrid_results)
        assert "hybrid" in context or "vector" in context or "graph" in context
        assert "Score:" in context

    def test_includes_path_info(self):
        results = [
            HybridResult(
                text="Some content",
                source="hybrid",
                similarity=0.8,
                graph_score=0.7,
                combined_score=0.75,
                path_info="A -> B -> C",
            ),
        ]
        context = HybridRetriever.format_context(results)
        assert "Graph path" in context
        assert "A -> B -> C" in context

    def test_respects_token_limit(self):
        long_text = "x" * 10000
        results = [
            HybridResult(
                text=long_text,
                source="vector",
                similarity=0.9,
                graph_score=0.0,
                combined_score=0.45,
            ),
            HybridResult(
                text="second result",
                source="vector",
                similarity=0.8,
                graph_score=0.0,
                combined_score=0.40,
            ),
        ]
        context = HybridRetriever.format_context(results, max_tokens=100)
        # Should only include first result (or part of it)
        assert "second result" not in context


class TestTraversalInfo:
    def test_with_results(self, sample_graph_results):
        info = HybridRetriever._build_traversal_info(sample_graph_results)
        assert "hops" in info
        assert "entities" in info
        assert "similarity" in info

    def test_empty_results(self):
        info = HybridRetriever._build_traversal_info([])
        assert "No graph paths found" in info


class TestFindOverlap:
    def test_finds_overlap(self):
        existing = [
            HybridResult(
                text="FastAPI is a modern framework",
                source="vector",
                similarity=0.8,
                graph_score=0.0,
                combined_score=0.4,
            ),
        ]
        graph_result = GraphResult(
            text="FastAPI -> uses -> Starlette",
            entities=["FastAPI"],
            path_length=1,
            relation_types=["USES"],
            score=0.7,
        )
        idx = HybridRetriever._find_overlap(existing, graph_result)
        assert idx == 0

    def test_no_overlap(self):
        existing = [
            HybridResult(
                text="Python is a language",
                source="vector",
                similarity=0.8,
                graph_score=0.0,
                combined_score=0.4,
            ),
        ]
        graph_result = GraphResult(
            text="Django -> uses -> ORM",
            entities=["Django", "ORM"],
            path_length=1,
            relation_types=["USES"],
            score=0.7,
        )
        idx = HybridRetriever._find_overlap(existing, graph_result)
        assert idx is None

    def test_skips_non_vector_sources(self):
        existing = [
            HybridResult(
                text="FastAPI is great",
                source="graph",
                similarity=0.0,
                graph_score=0.8,
                combined_score=0.4,
            ),
        ]
        graph_result = GraphResult(
            text="FastAPI -> uses -> X",
            entities=["FastAPI"],
            path_length=1,
            relation_types=["USES"],
            score=0.7,
        )
        idx = HybridRetriever._find_overlap(existing, graph_result)
        assert idx is None


class TestHybridRetrieverInit:
    def test_constructs_with_stores(self, mock_graph_store):
        mock_vector_store = MagicMock()
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.6,
            graph_weight=0.4,
        )
        assert retriever._vector_weight == 0.6
        assert retriever._graph_weight == 0.4


class TestRetrieve:
    @pytest.fixture
    def retriever(self, mock_graph_store):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vector_weight = 0.5
        retriever._graph_weight = 0.5

        mock_vector_retriever = MagicMock()
        mock_vector_retriever.retrieve.return_value = [
            RetrievalResult(
                text="FastAPI is a modern web framework",
                source="fastapi.md",
                chunk_index=0,
                similarity=0.85,
                metadata={},
            ),
        ]
        retriever._vector_retriever = mock_vector_retriever

        mock_graph_rag = MagicMock()
        mock_graph_rag.global_search.return_value = [
            GraphResult(
                text="FastAPI -> uses -> Starlette",
                entities=["FastAPI", "Starlette"],
                path_length=1,
                relation_types=["USES"],
                score=0.75,
            ),
        ]
        retriever._graph_rag = mock_graph_rag
        return retriever

    def test_retrieve_returns_hybrid_results(self, retriever):
        results = retriever.retrieve("what is FastAPI?", top_k=5)
        assert len(results) >= 1
        assert all(isinstance(r, HybridResult) for r in results)

    def test_retrieve_sorted_by_score(self, retriever):
        results = retriever.retrieve("what is FastAPI?", top_k=5)
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self, retriever):
        results = retriever.retrieve("query", top_k=1)
        assert len(results) <= 1

    def test_retrieve_with_context(self, retriever):
        ctx = retriever.retrieve_with_context("what is FastAPI?", top_k=5)
        assert "query" in ctx
        assert "results" in ctx
        assert "total_results" in ctx
        assert "vector_count" in ctx
        assert "graph_count" in ctx
        assert "avg_vector_similarity" in ctx
        assert "avg_graph_score" in ctx
        assert "traversal_info" in ctx
        assert ctx["vector_count"] == 1
        assert ctx["graph_count"] == 1

    def test_retrieve_with_context_empty_scores(self, mock_graph_store):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._vector_weight = 0.5
        retriever._graph_weight = 0.5

        mock_vector_retriever = MagicMock()
        mock_vector_retriever.retrieve.return_value = []
        retriever._vector_retriever = mock_vector_retriever

        mock_graph_rag = MagicMock()
        mock_graph_rag.global_search.return_value = []
        retriever._graph_rag = mock_graph_rag

        ctx = retriever.retrieve_with_context("nothing", top_k=5)
        assert ctx["avg_vector_similarity"] == 0.0
        assert ctx["avg_graph_score"] == 0.0
