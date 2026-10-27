"""Context compressor — reduces chunk text to only the most relevant sentences.

Analogy for a C dev:
    Like stripping dead code from a binary. The linker removes symbols that
    are never referenced. This module removes sentences that are semantically
    "unreferenced" by the query.

Strategy: sentence-level scoring
    1. Split each chunk into sentences
    2. Embed each sentence and the query
    3. Compute cosine similarity(sentence, query)
    4. Keep only sentences above a threshold
    5. Return new RetrievalResult with compressed .text (source/metadata intact)

This is an opt-in module — it is NOT wired into the retrieval pipeline
automatically. Use it when context window budget is tight.
"""

from __future__ import annotations

import re

from spiral_01_chunks.embedder import Embedder
from spiral_01_chunks.retriever import RetrievalResult

# Sentences with similarity below this are considered off-topic and removed.
DEFAULT_COMPRESSION_THRESHOLD = 0.3

# Minimum sentences to keep per chunk (avoid producing empty context).
_MIN_SENTENCES = 1


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, stripping empty strings."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


class ContextCompressor:
    """Compresses retrieved chunks to query-relevant sentences only.

    Usage:
        compressor = ContextCompressor()
        compressed = compressor.compress(results, query)
        context = Retriever.format_context(compressed)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        embedder: Embedder | None = None,
    ) -> None:
        self._threshold = threshold
        self._embedder = embedder or Embedder()

    def compress(
        self,
        chunks: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        """Compress each chunk to sentences relevant to the query.

        Args:
            chunks: Retrieved results from Retriever.retrieve().
            query: The original user query (used for sentence scoring).

        Returns:
            New list of RetrievalResult with .text reduced to relevant sentences.
            Source, similarity, chunk_index, and metadata are preserved unchanged.
            Chunks where ALL sentences score below threshold still return at least
            _MIN_SENTENCES sentence(s) (the highest-scoring one).
        """
        if not chunks:
            return []

        query_vec = self._embedder.embed_text(query)
        compressed: list[RetrievalResult] = []

        for chunk in chunks:
            sentences = _split_sentences(chunk.text)
            if not sentences:
                compressed.append(chunk)
                continue

            # Score each sentence against the query
            sentence_vecs = self._embedder.embed_batch(sentences)
            scored = [
                (sent, Embedder.cosine_similarity(query_vec, vec))
                for sent, vec in zip(sentences, sentence_vecs)
            ]

            # Keep sentences above threshold
            kept = [sent for sent, sim in scored if sim >= self._threshold]

            # Guarantee at least _MIN_SENTENCES to avoid empty context
            if not kept:
                best_sent = max(scored, key=lambda x: x[1])[0]
                kept = [best_sent]

            compressed_text = " ".join(kept)
            compressed.append(
                RetrievalResult(
                    text=compressed_text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    similarity=chunk.similarity,
                    metadata=chunk.metadata,
                )
            )

        return compressed
