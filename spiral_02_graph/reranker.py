"""Reranker — reorders hybrid results by relevance.

Analogy for a C dev:
- First pass (hybrid retrieve) is like a quick malloc scan
- Reranking is like compaction — move the best to the front
- Simple mode: weighted scoring (fast, zero deps)
- Cross-encoder mode: pairwise relevance scoring (slower, more accurate)

Default: simple weighted scoring (no extra model download needed).
Opt-in: cross-encoder via USE_CROSS_ENCODER=true in .env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spiral_02_graph.config import RERANKER_MODEL, TOP_K, USE_CROSS_ENCODER
from spiral_02_graph.hybrid_retriever import HybridResult


@dataclass
class RankedResult:
    """A reranked result with final score."""

    text: str
    source: str
    original_score: float
    rerank_score: float
    final_score: float
    rank: int = 0
    path_info: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ")
        return f"RankedResult(#{self.rank}, score={self.final_score:.3f}, '{preview}...')"


class Reranker:
    """Reranks hybrid results using simple scoring or cross-encoder."""

    def __init__(self, use_cross_encoder: bool = USE_CROSS_ENCODER):
        self._use_cross_encoder = use_cross_encoder
        self._cross_encoder = None  # lazy loaded

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_k: int = TOP_K,
    ) -> list[RankedResult]:
        """Rerank results using the configured strategy.

        Args:
            query: The original query.
            results: Hybrid results to rerank.
            top_k: Maximum results to return.

        Returns:
            List of RankedResult, sorted by final_score (highest first).
        """
        if not results:
            return []

        if self._use_cross_encoder:
            ranked = self._cross_encoder_rerank(query, results)
        else:
            ranked = self._simple_rerank(query, results)

        # Assign ranks
        for i, r in enumerate(ranked[:top_k]):
            r.rank = i + 1

        return ranked[:top_k]

    def _simple_rerank(
        self,
        query: str,
        results: list[HybridResult],
    ) -> list[RankedResult]:
        """Weighted scoring with source bonus.

        Hybrid sources (found by both vector AND graph) get a 1.1x bonus.
        This rewards convergence: if two different retrieval methods agree,
        the result is more likely relevant.
        """
        ranked: list[RankedResult] = []

        for r in results:
            source_bonus = 1.1 if r.source == "hybrid" else 1.0
            rerank_score = r.combined_score * source_bonus

            ranked.append(
                RankedResult(
                    text=r.text,
                    source=r.source,
                    original_score=r.combined_score,
                    rerank_score=round(rerank_score, 4),
                    final_score=round(rerank_score, 4),
                    path_info=r.path_info,
                    metadata=r.metadata,
                )
            )

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked

    def _cross_encoder_rerank(
        self,
        query: str,
        results: list[HybridResult],
    ) -> list[RankedResult]:
        """Use cross-encoder model for pairwise relevance scoring.

        The cross-encoder takes (query, document) pairs and outputs a
        relevance score. More accurate than bi-encoder similarity but
        slower (can't be pre-computed).
        """
        model = self.cross_encoder

        # Build pairs
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)

        ranked: list[RankedResult] = []
        for r, ce_score in zip(results, scores):
            # Combine: 60% cross-encoder, 40% original hybrid score
            final = 0.6 * float(ce_score) + 0.4 * r.combined_score

            ranked.append(
                RankedResult(
                    text=r.text,
                    source=r.source,
                    original_score=r.combined_score,
                    rerank_score=round(float(ce_score), 4),
                    final_score=round(final, 4),
                    path_info=r.path_info,
                    metadata=r.metadata,
                )
            )

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked

    @property
    def cross_encoder(self):
        """Lazy-load cross-encoder model (~100MB download on first use)."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(RERANKER_MODEL)
        return self._cross_encoder
