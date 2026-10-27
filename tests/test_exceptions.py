"""Tests for custom exception hierarchies."""

from __future__ import annotations

import pytest

from spiral_01_chunks.exceptions import (
    ChunkingError,
    EmbeddingError,
    GenerationError,
    RAGDojoError,
    RetrievalError,
)
from spiral_05_economics.exceptions import BudgetExceededError, CostCalculationError


class TestRAGDojoHierarchy:
    def test_retrieval_error_is_rag_dojo_error(self) -> None:
        with pytest.raises(RAGDojoError):
            raise RetrievalError("store unavailable")

    def test_generation_error_is_rag_dojo_error(self) -> None:
        with pytest.raises(RAGDojoError):
            raise GenerationError("LLM timeout")

    def test_embedding_error_is_rag_dojo_error(self) -> None:
        with pytest.raises(RAGDojoError):
            raise EmbeddingError("model not loaded")

    def test_chunking_error_is_rag_dojo_error(self) -> None:
        with pytest.raises(RAGDojoError):
            raise ChunkingError("file unreadable")

    def test_all_errors_are_exceptions(self) -> None:
        for exc_cls in (
            RAGDojoError,
            RetrievalError,
            GenerationError,
            EmbeddingError,
            ChunkingError,
        ):
            assert issubclass(exc_cls, Exception)

    def test_message_preserved(self) -> None:
        msg = "something went wrong"
        exc = RetrievalError(msg)
        assert str(exc) == msg


class TestEconomicsHierarchy:
    def test_budget_exceeded_is_rag_dojo_error(self) -> None:
        with pytest.raises(RAGDojoError):
            raise BudgetExceededError("daily limit hit")

    def test_cost_calculation_is_rag_dojo_error(self) -> None:
        with pytest.raises(RAGDojoError):
            raise CostCalculationError("unknown model")

    def test_budget_exceeded_importable_from_circuit_breaker(self) -> None:
        """BudgetExceededError must still be importable from circuit_breaker."""
        from spiral_05_economics.circuit_breaker import BudgetExceededError as CBE

        assert CBE is BudgetExceededError

    def test_budget_exceeded_message(self) -> None:
        msg = "Daily budget exceeded: $1.50 / $1.00"
        exc = BudgetExceededError(msg)
        assert str(exc) == msg
