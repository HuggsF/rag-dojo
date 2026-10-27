"""Retriever — queries the vector store and returns ranked results.

Analogy for a C dev:
- The retriever is like a search function over a sorted array
- But instead of binary search (exact match), it does nearest-neighbor search
- The "distance" is semantic, not positional
- TOP_K is how many nearest neighbors to return
- SIMILARITY_THRESHOLD is the minimum quality bar (like a filter)

This is the R in RAG — Retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spiral_01_chunks.config import CONTEXT_BUDGET, SIMILARITY_THRESHOLD, TOP_K
from spiral_01_chunks.embedder import Embedder
from spiral_01_chunks.store import VectorStore

# Chunks with inter-similarity above this threshold are considered duplicates.
_DEDUP_THRESHOLD = 0.85


@dataclass
class RetrievalResult:
    """A single retrieval result with text and scoring."""

    text: str
    source: str
    chunk_index: int
    similarity: float
    metadata: dict[str, Any]

    @property
    def is_relevant(self) -> bool:
        """Check if this result passes the relevance threshold."""
        return self.similarity >= SIMILARITY_THRESHOLD

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Result(sim={self.similarity:.3f}, src={self.source}, '{preview}...')"


class Retriever:
    """Retrieves relevant chunks from the vector store."""

    def __init__(self, store: VectorStore | None = None):
        self._store = store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[RetrievalResult]:
        """Retrieve top-k relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Maximum results to return.
            threshold: Minimum similarity score.

        Returns:
            List of RetrievalResult, sorted by similarity (highest first).
        """
        raw_results = self._store.query(query, top_k=top_k)

        results = []
        for r in raw_results:
            result = RetrievalResult(
                text=r["text"],
                source=r.get("metadata", {}).get("source", "unknown"),
                chunk_index=r.get("metadata", {}).get("chunk_index", 0),
                similarity=r["similarity"],
                metadata=r.get("metadata", {}),
            )
            if result.similarity >= threshold:
                results.append(result)

        return sorted(results, key=lambda x: x.similarity, reverse=True)

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> dict[str, Any]:
        """Retrieve results with metadata for debugging/analysis.

        Returns a dict with results, query info, and stats.
        """
        results = self.retrieve(query, top_k, threshold)

        return {
            "query": query,
            "top_k": top_k,
            "threshold": threshold,
            "total_results": len(results),
            "results": results,
            "avg_similarity": (
                sum(r.similarity for r in results) / len(results) if results else 0.0
            ),
            "store_stats": self._store.stats(),
        }

    @staticmethod
    def deduplicate(
        results: list[RetrievalResult],
        threshold: float = _DEDUP_THRESHOLD,
    ) -> list[RetrievalResult]:
        """Remove near-duplicate chunks via pairwise cosine similarity.

        Keeps the first (highest-similarity) chunk when two chunks are
        too similar to each other, eliminating redundant context.
        """
        if len(results) <= 1:
            return list(results)

        embedder = Embedder()
        embeddings = embedder.embed_batch([r.text for r in results])

        kept: list[int] = [0]  # always keep the top result
        for i in range(1, len(results)):
            is_dup = False
            for j in kept:
                sim = Embedder.cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(i)

        return [results[i] for i in kept]

    @staticmethod
    def format_context(results: list[RetrievalResult], max_tokens: int = CONTEXT_BUDGET) -> str:
        """Format retrieval results into a context string for LLM.

        Concatenates chunks with source attribution, respecting token limit.

        Note: max_tokens here is the *context budget* (how much retrieved text
        we feed into the prompt), NOT LLM_MAX_TOKENS from config (which controls
        the LLM *output* length). Context budget > output budget because the
        prompt needs room for multiple chunks + source attribution.
        """
        context_parts: list[str] = []
        total_tokens = 0

        for r in results:
            chunk_text = f"[Source: {r.source} | Similarity: {r.similarity:.3f}]\n{r.text}"
            chunk_tokens = len(chunk_text) // 4  # rough estimate

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)
