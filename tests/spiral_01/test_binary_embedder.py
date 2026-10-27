"""Tests for BinaryEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spiral_01_chunks.binary_embedder import BinaryEmbedder


def _mock_embedder(float_vec: list[float]) -> MagicMock:
    """Create a mock Embedder that returns a fixed float vector."""
    mock = MagicMock()
    mock.embed_text.return_value = float_vec
    return mock


class TestBinaryEmbed:
    def test_returns_binary_values_only(self) -> None:
        mock = _mock_embedder([0.5, -0.3, 0.0, 1.2, -0.1])
        be = BinaryEmbedder(base_embedder=mock)
        result = be.embed("text")
        assert all(v in (0, 1) for v in result)

    def test_length_matches_float_embedding(self) -> None:
        float_vec = [0.1 * i for i in range(384)]
        mock = _mock_embedder(float_vec)
        be = BinaryEmbedder(base_embedder=mock)
        result = be.embed("text")
        assert len(result) == 384

    def test_positive_values_become_one(self) -> None:
        mock = _mock_embedder([0.1, 0.5, 1.0])
        be = BinaryEmbedder(base_embedder=mock)
        assert be.embed("t") == [1, 1, 1]

    def test_negative_and_zero_become_zero(self) -> None:
        mock = _mock_embedder([-0.1, -1.0, 0.0])
        be = BinaryEmbedder(base_embedder=mock)
        assert be.embed("t") == [0, 0, 0]

    def test_mixed_values(self) -> None:
        mock = _mock_embedder([0.3, -0.1, 0.0, 0.7])
        be = BinaryEmbedder(base_embedder=mock)
        assert be.embed("t") == [1, 0, 0, 1]

    def test_calls_embed_text_not_embed(self) -> None:
        """Must call embed_text(), never embed()."""
        mock = _mock_embedder([0.5])
        be = BinaryEmbedder(base_embedder=mock)
        be.embed("hello")
        mock.embed_text.assert_called_once_with("hello")
        mock.embed.assert_not_called()


class TestEmbedFloat:
    def test_delegates_to_embed_text(self) -> None:
        float_vec = [0.1, 0.2, 0.3]
        mock = _mock_embedder(float_vec)
        be = BinaryEmbedder(base_embedder=mock)
        result = be.embed_float("hello")
        assert result == float_vec
        mock.embed_text.assert_called_once_with("hello")


class TestHammingSimilarity:
    def test_identical_vectors_are_one(self) -> None:
        a = [1, 0, 1, 0, 1]
        assert BinaryEmbedder.hamming_similarity(a, a) == 1.0

    def test_opposite_vectors_are_zero(self) -> None:
        a = [1, 1, 1, 1]
        b = [0, 0, 0, 0]
        assert BinaryEmbedder.hamming_similarity(a, b) == 0.0

    def test_half_different(self) -> None:
        a = [1, 1, 0, 0]
        b = [1, 1, 1, 1]
        sim = BinaryEmbedder.hamming_similarity(a, b)
        assert abs(sim - 0.5) < 1e-9

    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            BinaryEmbedder.hamming_similarity([1, 0], [1, 0, 1])

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            BinaryEmbedder.hamming_similarity([], [])


class TestHammingDistance:
    def test_identical_is_zero(self) -> None:
        a = [1, 0, 1, 0]
        assert BinaryEmbedder.hamming_distance(a, a) == 0

    def test_counts_differences(self) -> None:
        a = [1, 1, 0, 0]
        b = [0, 1, 1, 0]
        assert BinaryEmbedder.hamming_distance(a, b) == 2

    def test_raises_on_mismatch(self) -> None:
        with pytest.raises(ValueError):
            BinaryEmbedder.hamming_distance([1], [1, 0])


class TestBinaryEmbedderIntegration:
    def test_default_embedder_created_when_none(self) -> None:
        """BinaryEmbedder() without args must not crash (lazy embedder)."""
        be = BinaryEmbedder()
        assert be._base is not None

    def test_similarity_is_symmetric(self) -> None:
        a = [1, 0, 1, 0, 1, 1]
        b = [0, 1, 1, 0, 1, 0]
        assert BinaryEmbedder.hamming_similarity(a, b) == BinaryEmbedder.hamming_similarity(b, a)
