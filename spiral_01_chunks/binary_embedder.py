"""Binary embedder — quantizes float embeddings to binary (0/1) vectors.

Analogy for a C dev:
    Float embeddings are like 32-bit integers — precise but slow to compare.
    Binary embeddings are like bitsets — approximate but XOR/popcount fast.
    Hamming distance on binary vectors is ~50x faster than cosine on floats.

Trade-off:
    - Accuracy: ~2-5% recall degradation vs. float cosine
    - Speed: 50-100x faster similarity computation
    - Memory: 32x smaller (384 floats → 384 bits)

This is a proof-of-concept module for benchmarking speed vs. quality.
It does NOT modify VectorStore or Embedder — ChromaDB doesn't support
binary vectors natively. Use it for pre-filtering or approximate re-ranking.

Quantization strategy:
    threshold > 0.0 → bit = 1  (simple sign-based binarization)
    threshold ≤ 0.0 → bit = 0

This is the approach from: Yamada et al. "Binary Passage Retriever" (2021).
"""

from __future__ import annotations

from spiral_01_chunks.embedder import Embedder


class BinaryEmbedder:
    """Wraps an Embedder to produce binary (0/1) embeddings.

    Composes over Embedder — does NOT modify or inherit from it.
    """

    def __init__(self, base_embedder: Embedder | None = None) -> None:
        """Initialize with an optional base embedder.

        Args:
            base_embedder: The float embedder to delegate to.
                          If None, a new Embedder() is created.
        """
        self._base = base_embedder or Embedder()

    def embed(self, text: str) -> list[int]:
        """Compute a binary (0/1) embedding for text.

        Process:
            1. Get float embedding via base_embedder.embed_text()
            2. Binarize: 1 if value > 0.0, else 0

        Args:
            text: Input text to embed.

        Returns:
            List of int (0 or 1), same length as float embedding.
        """
        float_vec = self._base.embed_text(text)
        return [1 if v > 0.0 else 0 for v in float_vec]

    def embed_float(self, text: str) -> list[float]:
        """Return the underlying float embedding (delegates to base_embedder).

        Useful when you need the full-precision fallback without creating
        a separate Embedder instance.

        Args:
            text: Input text to embed.

        Returns:
            Float embedding from base_embedder.embed_text().
        """
        return self._base.embed_text(text)

    @staticmethod
    def hamming_similarity(a: list[int], b: list[int]) -> float:
        """Compute normalized Hamming similarity between two binary vectors.

        Hamming similarity = 1 - (hamming_distance / len)
        where hamming_distance = number of positions that differ.

        Args:
            a: First binary vector.
            b: Second binary vector.

        Returns:
            Float in [0.0, 1.0]. 1.0 = identical, 0.0 = fully opposite.

        Raises:
            ValueError: If vectors have different lengths or are empty.
        """
        if len(a) != len(b):
            raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
        if not a:
            raise ValueError("Cannot compute similarity of empty vectors")

        differing = sum(x != y for x, y in zip(a, b))
        return 1.0 - (differing / len(a))

    @staticmethod
    def hamming_distance(a: list[int], b: list[int]) -> int:
        """Count positions where a and b differ.

        This is the raw XOR popcount — lower is more similar.
        Use hamming_similarity() for a normalized [0, 1] score.
        """
        if len(a) != len(b):
            raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
        return sum(x != y for x, y in zip(a, b))
