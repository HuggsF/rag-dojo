"""Semantic cache — L0 cache using embedding similarity instead of exact hash.

Analogy for a C dev:
    Hash cache = memcmp() — exact byte match or miss.
    Semantic cache = fuzzy match with a distance threshold.
    "What is FastAPI?" and "Explain FastAPI" are semantically close
    and should hit the same cache entry.

The cache stores (embedding, answer) pairs. On lookup, it computes
cosine similarity between the query embedding and all cached embeddings.
If the best match exceeds the threshold, it's a hit.

Trade-off: slower than hash lookup (O(n) vs O(1)), but dramatically
higher hit rate for natural language queries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A single entry in the semantic cache."""

    query: str
    embedding: list[float]
    answer: str


class SemanticCache:
    """Embedding-based semantic cache for RAG queries.

    Uses cosine similarity to match queries that are semantically
    similar but lexically different.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 500,
    ) -> None:
        self._entries: list[CacheEntry] = []
        self._threshold = similarity_threshold
        self._max_entries = max_entries

    def store(self, query: str, embedding: list[float], answer: str) -> None:
        """Store a query/answer pair with its embedding."""
        self._entries.append(CacheEntry(query=query, embedding=embedding, answer=answer))

        # Evict oldest if over capacity
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def lookup(self, embedding: list[float]) -> str | None:
        """Find the best matching cached answer by cosine similarity.

        Returns the answer if similarity >= threshold, else None.
        """
        if not self._entries:
            return None

        best_sim = -1.0
        best_answer: str | None = None

        for entry in self._entries:
            sim = _cosine_similarity(embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_answer = entry.answer

        if best_sim >= self._threshold:
            return best_answer
        return None

    def size(self) -> int:
        """Return current cache size."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    @property
    def threshold(self) -> float:
        """Return the similarity threshold."""
        return self._threshold


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Inlined to avoid circular dependency on Embedder.
    """
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot / (norm_a * norm_b))
