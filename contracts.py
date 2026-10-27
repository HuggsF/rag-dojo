"""Cross-spiral contracts — formal interfaces each spiral must satisfy.

Each spiral is a module (.o file). These Protocols define the ABI.
If a spiral doesn't satisfy its Protocol, the linker (pipeline_factory) fails.

Usage:
    from contracts import SpiralModule, verify_spiral

    # Check at import time
    assert isinstance(my_module, SpiralModule)

    # Or verify all at once
    errors = verify_all_spirals()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpiralModule(Protocol):
    """Contract every spiral must satisfy."""

    def health(self) -> dict[str, object]:
        """Return health status. Must include 'healthy' (bool) key."""
        ...


@runtime_checkable
class RetrievalBackend(Protocol):
    """Contract for any retrieval component (spiral_01, spiral_02)."""

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve relevant documents."""
        ...

    @staticmethod
    def format_context(results: list, max_tokens: int = 2000) -> str:
        """Format results into context string."""
        ...


@runtime_checkable
class GenerationBackend(Protocol):
    """Contract for any LLM generation component."""

    def generate(self, context: str, question: str) -> object:
        """Generate answer from context. Returns GenerationResult-like."""
        ...


@runtime_checkable
class CostTracker(Protocol):
    """Contract for cost tracking (spiral_05)."""

    def record_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        query: str,
    ) -> None:
        """Record a cost event."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


@runtime_checkable
class SelfImprover(Protocol):
    """Contract for self-improvement loop (spiral_06)."""

    def run(self, scorecard: object, answer_fn: object, **kwargs: object) -> object:
        """Run improvement cycles."""
        ...


def verify_all_spirals() -> list[str]:
    """Verify all spiral implementations satisfy their contracts.

    Returns list of error messages. Empty = all contracts satisfied.
    """
    errors: list[str] = []

    # Spiral 01/02: Retrieval
    try:
        from spiral_01_chunks.retriever import Retriever

        if not issubclass(Retriever, RetrievalBackend):
            errors.append("Retriever does not satisfy RetrievalBackend")
    except ImportError as e:
        errors.append(f"Cannot import Retriever: {e}")

    try:
        from spiral_02_graph.hybrid_retriever import HybridRetriever

        if not issubclass(HybridRetriever, RetrievalBackend):
            errors.append("HybridRetriever does not satisfy RetrievalBackend")
    except ImportError as e:
        errors.append(f"Cannot import HybridRetriever: {e}")

    # Spiral 01: Generation
    try:
        from spiral_01_chunks.generator import Generator

        if not issubclass(Generator, GenerationBackend):
            errors.append("Generator does not satisfy GenerationBackend")
    except ImportError as e:
        errors.append(f"Cannot import Generator: {e}")

    try:
        from spiral_01_chunks.ollama_generator import OllamaGenerator

        if not issubclass(OllamaGenerator, GenerationBackend):
            errors.append("OllamaGenerator does not satisfy GenerationBackend")
    except ImportError as e:
        errors.append(f"Cannot import OllamaGenerator: {e}")

    # Spiral 05: Cost tracking
    try:
        from spiral_05_economics.budget_tracker import BudgetTracker

        if not issubclass(BudgetTracker, CostTracker):
            errors.append("BudgetTracker does not satisfy CostTracker")
    except ImportError as e:
        errors.append(f"Cannot import BudgetTracker: {e}")

    # Spiral 06: Self-improvement
    try:
        from spiral_06_state_of_art.cycle_manager import CycleManager

        if not issubclass(CycleManager, SelfImprover):
            errors.append("CycleManager does not satisfy SelfImprover")
    except ImportError as e:
        errors.append(f"Cannot import CycleManager: {e}")

    return errors
