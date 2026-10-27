"""Tests for Spiral 02 — Reranker module.

Key validations:
- Simple reranking sorts by score
- Hybrid source gets bonus
- Respects top_k limit
- Empty input returns empty output
- Rank assignment is correct
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spiral_02_graph.hybrid_retriever import HybridResult
from spiral_02_graph.reranker import RankedResult, Reranker


class TestRankedResult:
    def test_basic_creation(self):
        r = RankedResult(
            text="Test",
            source="vector",
            original_score=0.8,
            rerank_score=0.85,
            final_score=0.85,
            rank=1,
        )
        assert r.rank == 1
        assert r.final_score == 0.85

    def test_repr(self):
        r = RankedResult(
            text="FastAPI is a web framework",
            source="hybrid",
            original_score=0.8,
            rerank_score=0.9,
            final_score=0.9,
            rank=1,
        )
        assert "#1" in repr(r)
        assert "0.9" in repr(r)


class TestSimpleReranker:
    @pytest.fixture
    def reranker(self):
        return Reranker(use_cross_encoder=False)

    def test_empty_input(self, reranker):
        result = reranker.rerank("query", [])
        assert result == []

    def test_sorts_by_score(self, reranker, sample_hybrid_results):
        ranked = reranker.rerank("FastAPI middleware", sample_hybrid_results)
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_gets_bonus(self, reranker):
        results = [
            HybridResult(
                text="A",
                source="hybrid",
                similarity=0.8,
                graph_score=0.7,
                combined_score=0.5,
            ),
            HybridResult(
                text="B",
                source="vector",
                similarity=0.85,
                graph_score=0.0,
                combined_score=0.5,
            ),
        ]
        ranked = reranker.rerank("test", results)
        # Hybrid should rank higher due to 1.1x bonus
        assert ranked[0].source == "hybrid"
        assert ranked[0].final_score > ranked[1].final_score

    def test_respects_top_k(self, reranker, sample_hybrid_results):
        ranked = reranker.rerank("query", sample_hybrid_results, top_k=2)
        assert len(ranked) <= 2

    def test_rank_assignment(self, reranker, sample_hybrid_results):
        ranked = reranker.rerank("query", sample_hybrid_results)
        for i, r in enumerate(ranked):
            assert r.rank == i + 1

    def test_preserves_path_info(self, reranker):
        results = [
            HybridResult(
                text="Content",
                source="hybrid",
                similarity=0.8,
                graph_score=0.7,
                combined_score=0.75,
                path_info="A -> B -> C",
            ),
        ]
        ranked = reranker.rerank("test", results)
        assert ranked[0].path_info == "A -> B -> C"

    def test_preserves_metadata(self, reranker):
        results = [
            HybridResult(
                text="Content",
                source="vector",
                similarity=0.8,
                graph_score=0.0,
                combined_score=0.4,
                metadata={"key": "value"},
            ),
        ]
        ranked = reranker.rerank("test", results)
        assert ranked[0].metadata == {"key": "value"}


class TestCrossEncoderReranker:
    @pytest.fixture
    def ce_reranker(self):
        reranker = Reranker(use_cross_encoder=True)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.6])
        reranker._cross_encoder = mock_model
        return reranker

    def test_cross_encoder_sorts_by_combined_score(self, ce_reranker):
        results = [
            HybridResult(
                text="Result A",
                source="vector",
                similarity=0.8,
                graph_score=0.0,
                combined_score=0.4,
            ),
            HybridResult(
                text="Result B",
                source="graph",
                similarity=0.0,
                graph_score=0.7,
                combined_score=0.35,
            ),
            HybridResult(
                text="Result C",
                source="hybrid",
                similarity=0.6,
                graph_score=0.5,
                combined_score=0.55,
            ),
        ]
        ranked = ce_reranker.rerank("query", results)
        assert len(ranked) == 3
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_cross_encoder_combines_scores(self, ce_reranker):
        results = [
            HybridResult(
                text="Only one",
                source="vector",
                similarity=0.8,
                graph_score=0.0,
                combined_score=0.5,
            ),
        ]
        ce_reranker._cross_encoder.predict.return_value = np.array([0.8])
        ranked = ce_reranker.rerank("query", results)
        # final = 0.6 * 0.8 + 0.4 * 0.5 = 0.68
        assert ranked[0].final_score == pytest.approx(0.68, abs=0.01)
        assert ranked[0].rerank_score == pytest.approx(0.8, abs=0.01)

    def test_cross_encoder_preserves_path_info(self, ce_reranker):
        results = [
            HybridResult(
                text="Content",
                source="hybrid",
                similarity=0.8,
                graph_score=0.7,
                combined_score=0.75,
                path_info="A -> B",
            ),
        ]
        ce_reranker._cross_encoder.predict.return_value = np.array([0.9])
        ranked = ce_reranker.rerank("query", results)
        assert ranked[0].path_info == "A -> B"

    def test_cross_encoder_respects_top_k(self, ce_reranker):
        results = [
            HybridResult(
                text=f"R{i}", source="vector", similarity=0.5, graph_score=0.0, combined_score=0.25
            )
            for i in range(5)
        ]
        ce_reranker._cross_encoder.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        ranked = ce_reranker.rerank("query", results, top_k=2)
        assert len(ranked) == 2

    def test_cross_encoder_lazy_load(self):
        reranker = Reranker(use_cross_encoder=True)
        with patch("spiral_02_graph.reranker.CrossEncoder", create=True) as mock_ce_cls:
            mock_model = MagicMock()
            mock_ce_cls.return_value = mock_model

            with patch.dict(
                "sys.modules", {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_cls)}
            ):
                reranker._cross_encoder = None
                # Access the property to trigger lazy load
                with patch("spiral_02_graph.reranker.CrossEncoder", mock_ce_cls, create=True):
                    model = reranker.cross_encoder
                    assert model is not None
