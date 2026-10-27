"""Model router — routes queries to the cheapest adequate model.

Analogy for a C dev:
    Like choosing -O0 / -O1 / -O2 based on what you're compiling.
    A hello world doesn't need -O2. A real-time system doesn't survive -O0.
    Same here: simple question → cheap model, complex → expensive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from spiral_05_economics.config import (
    COMPLEXITY_FAST_MAX,
    COMPLEXITY_STANDARD_MAX,
    TIER_COMPLEX,
    TIER_DEFAULT_MODEL,
    TIER_FAST,
    TIER_STANDARD,
)
from spiral_05_economics.cost_calculator import CostCalculator


@dataclass
class RoutingDecision:
    """Result of routing a query to a model tier."""

    query: str
    tier: str
    model: str
    complexity_score: float
    reason: str

    def __repr__(self) -> str:
        return (
            f"RoutingDecision(tier={self.tier}, model={self.model}, "
            f"score={self.complexity_score:.2f}, reason={self.reason!r})"
        )


# Complexity signals — keywords that indicate higher complexity
_COMPLEX_KEYWORDS = {
    "design",
    "architect",
    "architecture",
    "scalab",
    "pattern",
    "trade-off",
    "tradeoff",
    "compare",
    "versus",
    "vs",
    "implement",
    "build",
    "create",
    "develop",
    "event-driven",
    "microservice",
    "distributed",
    "fault tolerance",
    "circuit breaker",
    "cqrs",
    "saga",
    "optimize",
    "performance",
    "benchmark",
}

_SIMPLE_KEYWORDS = {
    "what is",
    "what's",
    "define",
    "explain",
    "how to",
    "list",
    "show",
    "get",
    "which",
}


class ModelRouter:
    """Routes queries to appropriate model tiers based on complexity."""

    def __init__(
        self,
        calculator: CostCalculator | None = None,
        fast_max: float = COMPLEXITY_FAST_MAX,
        standard_max: float = COMPLEXITY_STANDARD_MAX,
    ) -> None:
        self._calculator = calculator or CostCalculator()
        self._fast_max = fast_max
        self._standard_max = standard_max

    def score_complexity(self, query: str) -> float:
        """Score query complexity from 0.0 (trivial) to 1.0 (very complex).

        Factors:
        - Length (longer queries tend to be more complex)
        - Presence of complex keywords
        - Presence of simple keywords (reduces score)
        - Number of clauses/sentences
        """
        if not query.strip():
            return 0.0

        query_lower = query.lower()
        words = query_lower.split()
        word_count = len(words)

        # Length factor: 0-1 based on word count (0-5 words = 0, 50+ = 1)
        length_score = min(1.0, max(0.0, (word_count - 5) / 45))

        # Complex keyword factor
        complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in query_lower)
        complex_score = min(1.0, complex_hits / 3)

        # Simple keyword factor (reduces complexity)
        simple_hits = sum(1 for kw in _SIMPLE_KEYWORDS if kw in query_lower)
        simple_penalty = min(0.3, simple_hits * 0.15)

        # Clause count (commas, semicolons, "and", "or")
        clause_markers = len(re.findall(r"[,;]|\band\b|\bor\b", query_lower))
        clause_score = min(1.0, clause_markers / 5)

        # Weighted combination
        raw_score = (
            length_score * 0.25 + complex_score * 0.40 + clause_score * 0.20 - simple_penalty
        )

        # Add small boost for very long queries
        if word_count > 30:
            raw_score += 0.15

        return max(0.0, min(1.0, raw_score))

    def route(self, query: str, force_tier: str | None = None) -> RoutingDecision:
        """Route a query to the appropriate model tier.

        Args:
            query: The user query to route.
            force_tier: If set, override the automatic tier selection.
        """
        score = self.score_complexity(query)

        if force_tier:
            tier = force_tier
            reason = f"forced to {force_tier}"
        elif score <= self._fast_max:
            tier = TIER_FAST
            reason = "low complexity — simple lookup or definition"
        elif score <= self._standard_max:
            tier = TIER_STANDARD
            reason = "moderate complexity — analysis or comparison"
        else:
            tier = TIER_COMPLEX
            reason = "high complexity — design, architecture, or multi-step reasoning"

        model = TIER_DEFAULT_MODEL.get(tier, TIER_DEFAULT_MODEL[TIER_STANDARD])

        return RoutingDecision(
            query=query,
            tier=tier,
            model=model,
            complexity_score=score,
            reason=reason,
        )
