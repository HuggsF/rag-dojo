"""Tests for spiral_05_economics.semantic_cache — embedding-based L0 cache."""

import math

import pytest

from spiral_05_economics.semantic_cache import SemanticCache, _cosine_similarity


class TestCosineSimlarity:
    """Tests for the inlined cosine similarity function."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert math.isclose(_cosine_similarity(v, v), 1.0, abs_tol=1e-6)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-6)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(_cosine_similarity(a, b), -1.0, abs_tol=1e-6)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestSemanticCache:
    """Tests for SemanticCache — the full cache logic."""

    @pytest.fixture
    def cache(self):
        return SemanticCache(similarity_threshold=0.9, max_entries=10)

    def test_empty_cache_returns_none(self, cache):
        assert cache.lookup([1.0, 0.0, 0.0]) is None

    def test_store_and_lookup_exact(self, cache):
        emb = [1.0, 0.0, 0.0]
        cache.store("What is FastAPI?", emb, "A web framework")
        assert cache.lookup(emb) == "A web framework"

    def test_similar_query_hits(self, cache):
        emb_store = [1.0, 0.1, 0.0]
        emb_query = [1.0, 0.05, 0.0]  # very close
        cache.store("What is FastAPI?", emb_store, "A web framework")
        result = cache.lookup(emb_query)
        assert result == "A web framework"

    def test_dissimilar_query_misses(self, cache):
        emb_store = [1.0, 0.0, 0.0]
        emb_query = [0.0, 1.0, 0.0]  # orthogonal
        cache.store("What is FastAPI?", emb_store, "A web framework")
        assert cache.lookup(emb_query) is None

    def test_best_match_returned(self, cache):
        cache.store("q1", [1.0, 0.0, 0.0], "answer1")
        cache.store("q2", [0.0, 1.0, 0.0], "answer2")
        # Query close to q2
        result = cache.lookup([0.05, 1.0, 0.0])
        assert result == "answer2"

    def test_max_entries_eviction(self):
        cache = SemanticCache(similarity_threshold=0.5, max_entries=3)
        for i in range(5):
            cache.store(f"q{i}", [float(i), 0.0], f"a{i}")
        assert cache.size() == 3

    def test_clear(self, cache):
        cache.store("q", [1.0], "a")
        cache.clear()
        assert cache.size() == 0

    def test_size(self, cache):
        assert cache.size() == 0
        cache.store("q1", [1.0], "a1")
        assert cache.size() == 1
        cache.store("q2", [0.0], "a2")
        assert cache.size() == 2

    def test_threshold_property(self, cache):
        assert cache.threshold == 0.9

    def test_threshold_boundary_miss(self):
        """Query just below threshold should miss."""
        cache = SemanticCache(similarity_threshold=0.99, max_entries=10)
        cache.store("q", [1.0, 0.0, 0.0], "answer")
        # Slightly different — cosine will be < 0.99
        result = cache.lookup([1.0, 0.3, 0.0])
        assert result is None

    def test_threshold_boundary_hit(self):
        """Query just above threshold should hit."""
        cache = SemanticCache(similarity_threshold=0.5, max_entries=10)
        cache.store("q", [1.0, 0.0, 0.0], "answer")
        # Close enough — cosine > 0.5
        result = cache.lookup([1.0, 0.5, 0.0])
        assert result == "answer"
