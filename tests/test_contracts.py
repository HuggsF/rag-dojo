"""Contract tests — verify real implementations satisfy Protocol interfaces.

Uses runtime_checkable isinstance checks + inspect.signature to ensure
structural subtyping is honored at both the type and method level.

Analogy for a C dev:
    Like checking that a .o file exports the correct symbols with the right
    types before the linker accepts it. Protocols = header files, impls = .o files.
"""

from __future__ import annotations

import inspect

import pytest

from pipeline_factory import EnhancedRetriever
from spiral_01_chunks.generator import Generator
from spiral_01_chunks.ollama_generator import OllamaGenerator
from spiral_02_graph.hybrid_retriever import HybridRetriever
from spiral_04_cognitive.cognitive_layer import CognitiveLayer
from spiral_04_cognitive.metrics import InMemoryMetrics, NoopMetrics
from spiral_05_economics.layer_pipeline import (
    CognitiveObserver,
    LLMBackend,
    MetricsObserver,
    RAGBackend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_method_params(cls: type, method_name: str) -> list[str]:
    """Return the parameter names (excluding 'self') for a method on cls."""
    method = getattr(cls, method_name)
    sig = inspect.signature(method)
    return [
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


# ---------------------------------------------------------------------------
# CognitiveObserver Protocol
# ---------------------------------------------------------------------------


class TestCognitiveObserverContract:
    """CognitiveLayer must satisfy CognitiveObserver Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(CognitiveLayer, CognitiveObserver)

    def test_enrich_method_exists(self) -> None:
        assert hasattr(CognitiveLayer, "enrich")
        assert callable(getattr(CognitiveLayer, "enrich"))

    def test_enrich_signature(self) -> None:
        params = _get_method_params(CognitiveLayer, "enrich")
        assert "bot_name" in params
        assert "command" in params
        assert "raw_output" in params
        assert "metrics" in params


# ---------------------------------------------------------------------------
# RAGBackend Protocol
# ---------------------------------------------------------------------------


class TestRAGBackendContract:
    """HybridRetriever must satisfy RAGBackend Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(HybridRetriever, RAGBackend)

    def test_retrieve_method_exists(self) -> None:
        assert hasattr(HybridRetriever, "retrieve")
        assert callable(getattr(HybridRetriever, "retrieve"))

    def test_retrieve_signature(self) -> None:
        params = _get_method_params(HybridRetriever, "retrieve")
        assert "query" in params
        assert "top_k" in params

    def test_format_context_exists(self) -> None:
        assert hasattr(HybridRetriever, "format_context")
        assert callable(getattr(HybridRetriever, "format_context"))

    def test_format_context_signature(self) -> None:
        params = _get_method_params(HybridRetriever, "format_context")
        assert "results" in params
        assert "max_tokens" in params


class TestEnhancedRetrieverContract:
    """EnhancedRetriever (pipeline_factory) must satisfy RAGBackend Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(EnhancedRetriever, RAGBackend)

    def test_retrieve_method_exists(self) -> None:
        assert hasattr(EnhancedRetriever, "retrieve")
        assert callable(getattr(EnhancedRetriever, "retrieve"))

    def test_retrieve_signature(self) -> None:
        params = _get_method_params(EnhancedRetriever, "retrieve")
        assert "query" in params
        assert "top_k" in params

    def test_format_context_exists(self) -> None:
        assert hasattr(EnhancedRetriever, "format_context")
        assert callable(getattr(EnhancedRetriever, "format_context"))

    def test_format_context_signature(self) -> None:
        params = _get_method_params(EnhancedRetriever, "format_context")
        assert "results" in params
        assert "max_tokens" in params


# ---------------------------------------------------------------------------
# LLMBackend Protocol
# ---------------------------------------------------------------------------


class TestGeneratorContract:
    """Generator must satisfy LLMBackend Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(Generator, LLMBackend)

    def test_generate_method_exists(self) -> None:
        assert hasattr(Generator, "generate")
        assert callable(getattr(Generator, "generate"))

    def test_generate_signature(self) -> None:
        params = _get_method_params(Generator, "generate")
        assert "context" in params
        assert "question" in params


class TestOllamaGeneratorContract:
    """OllamaGenerator must satisfy LLMBackend Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(OllamaGenerator, LLMBackend)

    def test_generate_method_exists(self) -> None:
        assert hasattr(OllamaGenerator, "generate")
        assert callable(getattr(OllamaGenerator, "generate"))

    def test_generate_signature(self) -> None:
        params = _get_method_params(OllamaGenerator, "generate")
        assert "context" in params
        assert "question" in params


# ---------------------------------------------------------------------------
# MetricsObserver Protocol
# ---------------------------------------------------------------------------


_METRICS_METHODS = [
    ("record_latency", ["operation", "seconds"]),
    ("record_tokens", ["model", "count"]),
    ("record_cache_hit", ["layer"]),
    ("record_cache_miss", ["layer"]),
]


class TestInMemoryMetricsContract:
    """InMemoryMetrics must satisfy MetricsObserver Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(InMemoryMetrics, MetricsObserver)

    @pytest.mark.parametrize("method_name,expected_params", _METRICS_METHODS)
    def test_method_exists_and_signature(
        self, method_name: str, expected_params: list[str]
    ) -> None:
        assert hasattr(InMemoryMetrics, method_name)
        params = _get_method_params(InMemoryMetrics, method_name)
        for p in expected_params:
            assert p in params, f"{method_name} missing param '{p}'"


class TestNoopMetricsContract:
    """NoopMetrics must satisfy MetricsObserver Protocol."""

    def test_isinstance_check(self) -> None:
        assert issubclass(NoopMetrics, MetricsObserver)

    @pytest.mark.parametrize("method_name,expected_params", _METRICS_METHODS)
    def test_method_exists_and_signature(
        self, method_name: str, expected_params: list[str]
    ) -> None:
        assert hasattr(NoopMetrics, method_name)
        params = _get_method_params(NoopMetrics, method_name)
        for p in expected_params:
            assert p in params, f"{method_name} missing param '{p}'"
