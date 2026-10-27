"""Fallback cascade — 7 strategies when delta = 0 or confidence is low.

Analogy for a C dev:
- This is the error-handling chain: try strategy 1, if it fails, try 2, etc.
- Like a switch-case with fallthrough: each case is progressively more aggressive
- Strategy 7 is the "abort and report" — the final catch-all

The 7 strategies (in order):
1. BROADEN_QUERY    — Expand query terms, add synonyms
2. DEEPEN_GRAPH     — Increase graph search depth
3. BOOST_CREATIVITY — Shift tension toward exploration
4. BOOST_LOGIC      — Shift tension toward convergence
5. SWAP_RETRIEVER   — Switch from hybrid to pure vector or pure graph
6. RELAX_THRESHOLD  — Lower confidence threshold temporarily
7. REPORT_FAILURE   — Give up and report why
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FallbackStrategy(str, Enum):
    """The 7 fallback strategies, in cascade order."""

    BROADEN_QUERY = "BROADEN_QUERY"
    DEEPEN_GRAPH = "DEEPEN_GRAPH"
    BOOST_CREATIVITY = "BOOST_CREATIVITY"
    BOOST_LOGIC = "BOOST_LOGIC"
    SWAP_RETRIEVER = "SWAP_RETRIEVER"
    RELAX_THRESHOLD = "RELAX_THRESHOLD"
    REPORT_FAILURE = "REPORT_FAILURE"


# Ordered cascade
FALLBACK_CASCADE: list[FallbackStrategy] = list(FallbackStrategy)


@dataclass
class FallbackAction:
    """A specific fallback action to apply."""

    strategy: FallbackStrategy
    description: str
    adjustments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "strategy": self.strategy.value,
            "description": self.description,
            "adjustments": self.adjustments,
        }


def get_fallback_action(strategy: FallbackStrategy) -> FallbackAction:
    """Get the concrete action for a fallback strategy.

    Args:
        strategy: Which fallback to apply.

    Returns:
        FallbackAction with description and parameter adjustments.
    """
    actions: dict[FallbackStrategy, FallbackAction] = {
        FallbackStrategy.BROADEN_QUERY: FallbackAction(
            strategy=FallbackStrategy.BROADEN_QUERY,
            description="Expand query with related terms to cast a wider net.",
            adjustments={"top_k_multiplier": 2.0, "threshold_reduction": 0.1},
        ),
        FallbackStrategy.DEEPEN_GRAPH: FallbackAction(
            strategy=FallbackStrategy.DEEPEN_GRAPH,
            description="Increase graph traversal depth to find deeper connections.",
            adjustments={"graph_depth_increase": 2, "graph_top_k_increase": 3},
        ),
        FallbackStrategy.BOOST_CREATIVITY: FallbackAction(
            strategy=FallbackStrategy.BOOST_CREATIVITY,
            description="Shift tension toward exploration — allow more diverse results.",
            adjustments={"creativity_boost": 0.2, "logic_reduction": 0.1},
        ),
        FallbackStrategy.BOOST_LOGIC: FallbackAction(
            strategy=FallbackStrategy.BOOST_LOGIC,
            description="Shift tension toward convergence — focus on precise matches.",
            adjustments={"logic_boost": 0.2, "creativity_reduction": 0.1},
        ),
        FallbackStrategy.SWAP_RETRIEVER: FallbackAction(
            strategy=FallbackStrategy.SWAP_RETRIEVER,
            description="Switch retriever mode: if hybrid failed, try pure vector or graph.",
            adjustments={"retriever_mode": "vector_only", "fallback_mode": "graph_only"},
        ),
        FallbackStrategy.RELAX_THRESHOLD: FallbackAction(
            strategy=FallbackStrategy.RELAX_THRESHOLD,
            description="Temporarily lower confidence threshold to allow marginal results.",
            adjustments={"threshold_multiplier": 0.5},
        ),
        FallbackStrategy.REPORT_FAILURE: FallbackAction(
            strategy=FallbackStrategy.REPORT_FAILURE,
            description="All strategies exhausted. Report failure with diagnostics.",
            adjustments={"abort": True},
        ),
    }

    return actions[strategy]


class FallbackCascade:
    """Manages the fallback cascade — tries strategies in order until one works."""

    def __init__(self):
        self._current_index: int = 0
        self._attempted: list[FallbackStrategy] = []

    @property
    def current_strategy(self) -> FallbackStrategy | None:
        """Current strategy in the cascade, or None if exhausted."""
        if self._current_index >= len(FALLBACK_CASCADE):
            return None
        return FALLBACK_CASCADE[self._current_index]

    @property
    def is_exhausted(self) -> bool:
        """True if all strategies have been tried."""
        return self._current_index >= len(FALLBACK_CASCADE)

    @property
    def attempted(self) -> list[FallbackStrategy]:
        """List of strategies already attempted."""
        return list(self._attempted)

    def next(self) -> FallbackAction | None:
        """Get the next fallback action in the cascade.

        Returns:
            FallbackAction if strategies remain, None if exhausted.
        """
        strategy = self.current_strategy
        if strategy is None:
            return None

        action = get_fallback_action(strategy)
        self._attempted.append(strategy)
        self._current_index += 1
        return action

    def reset(self) -> None:
        """Reset the cascade to try from the beginning."""
        self._current_index = 0
        self._attempted.clear()
