"""Tests for Spiral 01 — Embedder module.

Key validations:
- Embeddings have correct dimensions (384 for MiniLM-L6-v2)
- Similar texts produce similar vectors (high cosine similarity)
- Different texts produce different vectors (low cosine similarity)
- Batch embedding produces same results as individual embedding
"""

from spiral_01_chunks.embedder import Embedder


class TestEmbedder:
    def test_embed_returns_list_of_floats(self, embedder: Embedder):
        result = embedder.embed_text("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embedding_dimensions(self, embedder: Embedder):
        result = embedder.embed_text("Hello world")
        assert len(result) == 384  # MiniLM-L6-v2

    def test_dimensions_property(self, embedder: Embedder):
        assert embedder.dimensions == 384

    def test_similar_texts_high_similarity(self, embedder: Embedder):
        vec_a = embedder.embed_text("How to create an API with FastAPI")
        vec_b = embedder.embed_text("Building APIs using the FastAPI framework")
        similarity = Embedder.cosine_similarity(vec_a, vec_b)
        assert similarity > 0.7, f"Similar texts should have high similarity, got {similarity}"

    def test_different_texts_low_similarity(self, embedder: Embedder):
        vec_a = embedder.embed_text("How to create an API with FastAPI")
        vec_b = embedder.embed_text("The weather in Tokyo is sunny today")
        similarity = Embedder.cosine_similarity(vec_a, vec_b)
        assert similarity < 0.5, f"Different texts should have low similarity, got {similarity}"

    def test_identical_texts_perfect_similarity(self, embedder: Embedder):
        text = "FastAPI middleware example"
        vec_a = embedder.embed_text(text)
        vec_b = embedder.embed_text(text)
        similarity = Embedder.cosine_similarity(vec_a, vec_b)
        assert similarity > 0.99, f"Identical texts should have ~1.0 similarity, got {similarity}"

    def test_batch_embedding(self, embedder: Embedder):
        texts = ["Hello", "World", "FastAPI"]
        results = embedder.embed_batch(texts)
        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    def test_batch_matches_individual(self, embedder: Embedder):
        texts = ["Hello world", "FastAPI is great"]
        batch = embedder.embed_batch(texts)
        individual = [embedder.embed_text(t) for t in texts]

        for b, i in zip(batch, individual):
            similarity = Embedder.cosine_similarity(b, i)
            assert similarity > 0.99, "Batch and individual should produce same embeddings"


class TestCosineSimilarity:
    def test_zero_vector(self):
        assert Embedder.cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(Embedder.cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_opposite_vectors(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        assert abs(Embedder.cosine_similarity(vec_a, vec_b) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self):
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert abs(Embedder.cosine_similarity(vec_a, vec_b)) < 1e-6
