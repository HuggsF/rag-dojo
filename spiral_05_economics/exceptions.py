"""Custom exception hierarchy for Spiral 05 — Economics Layer.

Extends the base RAGDojoError hierarchy from spiral_01_chunks.
"""

from __future__ import annotations

from spiral_01_chunks.exceptions import RAGDojoError


class BudgetExceededError(RAGDojoError):
    """Raised when an LLM call would exceed configured budget limits.

    Previously defined in spiral_05_economics.circuit_breaker — still
    importable from there via re-export for backwards compatibility.
    """


class CostCalculationError(RAGDojoError):
    """Raised when cost calculation fails (unknown model, invalid token count, etc.)."""
