"""Embedder — transforms text into N-dimensional vectors.

Analogy for a C dev:
- An embedding is a hash function, but instead of mapping to a single int,
  it maps to a vector of 384 floats (dimensions)
- Similar text → similar vectors (close in N-dimensional space)
- Unlike hash collisions, similar embeddings are FEATURES, not bugs
- cosine_similarity(vec_a, vec_b) tells you how "close" two texts are
  (1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite)

Model: all-MiniLM-L6-v2
- 384 dimensions
- ~22M parameters (tiny — runs on CPU)
- Trained on 1B+ sentence pairs
- Good enough for production RAG (not the best, but free and fast)
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from spiral_01_chunks.config import EMBEDDING_MODEL


class Embedder:
    """Wraps sentence-transformers for batch embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model (first call takes ~2s, subsequent calls instant)."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns a list of floats (vector in N-dimensional space).
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed multiple texts in batches (faster than one-by-one).

        Like batch processing in a pipeline — amortizes model loading cost.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()  # type: ignore[no-any-return]

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        cos(A, B) = (A · B) / (|A| × |B|)

        This is THE metric for semantic similarity in embedding space.
        - 1.0 = identical meaning
        - 0.7+ = semantically close (relevant for RAG)
        - 0.3-0.7 = vaguely related
        - <0.3 = unrelated
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))


# Singleton instance (model loads once, reuses)
_default_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get or create the default embedder instance."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder
