"""Hybrid Retriever — combines vector search (ChromaDB) + graph search (Neo4j).

Analogy for a C dev:
- Vector search = hash table lookup (find similar content by semantic hash)
- Graph search = pointer traversal (follow relationships between entities)
- Hybrid = both strategies, merged and ranked
- Like using BOTH an index scan and a sequential scan in a database,
  then picking the best results from both

This is the core of Spiral 02: two retrieval "blades" that complement each other.
Vectors find similar content. Graphs find connected content. Together = better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from spiral_01_chunks.config import SIMILARITY_THRESHOLD, TOP_K
from spiral_01_chunks.retriever import RetrievalResult, Retriever
from spiral_01_chunks.store import VectorStore
from spiral_02_graph.config import (
    DEAD_END_THRESHOLD,
    GRAPH_WEIGHT,
    VECTOR_WEIGHT,
)
from spiral_02_graph.graph_rag import GraphRAG, GraphResult
from spiral_02_graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A retrieval result that may come from vector, graph, or both."""

    text: str
    source: str  # "vector", "graph", or "hybrid"
    similarity: float  # vector similarity (0.0 if graph-only)
    graph_score: float  # graph score (0.0 if vector-only)
    combined_score: float  # weighted combination
    path_info: str = ""  # graph traversal description
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_relevant(self) -> bool:
        return self.combined_score >= DEAD_END_THRESHOLD

    def __repr__(self) -> str:
        preview = self.text[:40].replace("\n", " ")
        return f"HybridResult(score={self.combined_score:.3f}, src={self.source}, '{preview}...')"


class HybridRetriever:
    """Combines vector retrieval (spiral_01) with graph retrieval (spiral_02)."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        vector_weight: float = VECTOR_WEIGHT,
        graph_weight: float = GRAPH_WEIGHT,
        metrics: Any | None = None,
    ):
        self._vector_retriever = Retriever(vector_store or VectorStore())
        self._graph_rag = GraphRAG(graph_store)
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight
        self._metrics = metrics

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[HybridResult]:
        """Retrieve from both vector and graph, merge, and rank.

        Strategy:
        1. Get vector results from ChromaDB (spiral_01 Retriever)
        2. Get graph results from GraphRAG (local + global)
        3. Merge and deduplicate
        4. Score with weighted combination
        5. Sort by combined score
        6. Filter dead ends
        """
        # Vector retrieval
        vector_results = self._vector_retriever.retrieve(
            query,
            top_k=top_k,
            threshold=threshold,
        )

        # Graph retrieval (graceful degradation when Neo4j is offline)
        try:
            graph_results = self._graph_rag.global_search(query, top_k=top_k)
        except Exception:
            logger.warning(
                "Graph search failed — degrading to vector-only retrieval",
                exc_info=True,
            )
            graph_results = []
            if self._metrics is not None:
                self._metrics.record_cache_miss("graph_degradation")

        # Merge
        merged = self._merge_results(vector_results, graph_results)

        # Sort by combined score
        merged.sort(key=lambda r: r.combined_score, reverse=True)

        return merged[:top_k]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict[str, Any]:
        """Retrieve with full metadata for debugging/display."""
        vector_results = self._vector_retriever.retrieve(
            query,
            top_k=top_k,
            threshold=threshold,
        )
        graph_results = self._graph_rag.global_search(query, top_k=top_k)
        merged = self._merge_results(vector_results, graph_results)
        merged.sort(key=lambda r: r.combined_score, reverse=True)
        merged = merged[:top_k]

        # Compute stats
        vector_scores = [r.similarity for r in merged if r.similarity > 0]
        graph_scores = [r.graph_score for r in merged if r.graph_score > 0]

        return {
            "query": query,
            "results": merged,
            "total_results": len(merged),
            "vector_count": len(vector_results),
            "graph_count": len(graph_results),
            "avg_vector_similarity": (
                sum(vector_scores) / len(vector_scores) if vector_scores else 0.0
            ),
            "avg_graph_score": (sum(graph_scores) / len(graph_scores) if graph_scores else 0.0),
            "traversal_info": self._build_traversal_info(graph_results),
        }

    @staticmethod
    def format_context(results: list[HybridResult], max_tokens: int = 2000) -> str:
        """Format hybrid results into context for LLM.

        Includes source type and graph path info when available.
        """
        context_parts: list[str] = []
        total_tokens = 0

        for r in results:
            header = f"[Source: {r.source} | Score: {r.combined_score:.3f}]"
            if r.path_info:
                header += f"\n[Graph path: {r.path_info}]"
            chunk_text = f"{header}\n{r.text}"
            chunk_tokens = len(chunk_text) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    def _merge_results(
        self,
        vector_results: list[RetrievalResult],
        graph_results: list[GraphResult],
    ) -> list[HybridResult]:
        """Merge and deduplicate results from both sources.

        Deduplication: if a graph entity name appears in a vector chunk,
        they're merged into a single hybrid result with boosted score.
        """
        merged: list[HybridResult] = []

        # Add vector results
        for vr in vector_results:
            merged.append(
                HybridResult(
                    text=vr.text,
                    source="vector",
                    similarity=vr.similarity,
                    graph_score=0.0,
                    combined_score=round(self._vector_weight * vr.similarity, 4),
                    metadata=vr.metadata,
                )
            )

        # Add graph results, checking for overlap with vector results
        for gr in graph_results:
            # Check if any vector result contains the graph entities
            overlap_idx = self._find_overlap(merged, gr)
            if overlap_idx is not None:
                # Merge: boost the existing vector result with graph info
                existing = merged[overlap_idx]
                existing.graph_score = gr.score
                existing.combined_score = round(
                    self._vector_weight * existing.similarity + self._graph_weight * gr.score,
                    4,
                )
                existing.source = "hybrid"
                existing.path_info = gr.text
            else:
                # New result from graph only
                merged.append(
                    HybridResult(
                        text=gr.text,
                        source="graph",
                        similarity=0.0,
                        graph_score=gr.score,
                        combined_score=round(self._graph_weight * gr.score, 4),
                        path_info=gr.text,
                        metadata=gr.metadata,
                    )
                )

        return merged

    @staticmethod
    def _find_overlap(
        existing: list[HybridResult],
        graph_result: GraphResult,
    ) -> int | None:
        """Find if a graph result overlaps with an existing vector result.

        Overlap = any graph entity name appears in the vector chunk text.
        """
        for i, hr in enumerate(existing):
            if hr.source != "vector":
                continue
            text_lower = hr.text.lower()
            for entity in graph_result.entities:
                if entity.lower() in text_lower:
                    return i
        return None

    @staticmethod
    def _build_traversal_info(graph_results: list[GraphResult]) -> str:
        """Build a summary of graph traversal for display."""
        if not graph_results:
            return "No graph paths found"

        total_hops = sum(r.path_length for r in graph_results)
        all_entities: set[str] = set()
        for r in graph_results:
            all_entities.update(r.entities)

        avg_score = sum(r.score for r in graph_results) / len(graph_results)

        return (
            f"Traversal: {total_hops} hops, "
            f"{len(all_entities)} entities, "
            f"similarity: {avg_score:.2f}"
        )
