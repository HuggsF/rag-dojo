"""Proposer — analyzes scores and proposes improvements.

Analogy for a C dev:
    Like a profiler that doesn't just show hot spots — it suggests fixes.
    "Function X is 40% of runtime → inline it" = "chunk_size is too big → reduce it".
"""

from __future__ import annotations

from dataclasses import dataclass

from spiral_06_state_of_art.config import TUNABLE_PARAMS
from spiral_06_state_of_art.scorer import ScoreCard


@dataclass
class Proposal:
    """A proposed improvement to the RAG system."""

    param: str  # which parameter to change
    current_value: object
    proposed_value: object
    reason: str
    expected_impact: str  # which dimension(s) should improve
    confidence: float  # 0.0 to 1.0 — how confident in the improvement
    priority: int = 0  # lower = higher priority

    def __repr__(self) -> str:
        return (
            f"Proposal({self.param}: {self.current_value} → {self.proposed_value}, "
            f"reason={self.reason!r}, confidence={self.confidence:.2f})"
        )


# Analysis rules: dimension → (threshold, param, direction, reason)
_RULES: list[dict[str, object]] = [
    {
        "dimension": "precision",
        "below": 60.0,
        "param": "top_k",
        "direction": "decrease",
        "reason": "low precision — retrieving too many irrelevant chunks",
        "impact": "precision",
        "confidence": 0.7,
    },
    {
        "dimension": "recall",
        "below": 60.0,
        "param": "chunk_size",
        "direction": "decrease",
        "reason": "low recall — chunks may be too large, splitting semantic units",
        "impact": "recall",
        "confidence": 0.6,
    },
    {
        "dimension": "recall",
        "below": 50.0,
        "param": "chunk_overlap",
        "direction": "increase",
        "reason": "very low recall — increasing overlap to preserve context between chunks",
        "impact": "recall",
        "confidence": 0.5,
    },
    {
        "dimension": "recall",
        "below": 40.0,
        "param": "top_k",
        "direction": "increase",
        "reason": "very low recall — retrieving too few chunks",
        "impact": "recall",
        "confidence": 0.65,
    },
    {
        "dimension": "precision",
        "below": 40.0,
        "param": "similarity_threshold",
        "direction": "increase",
        "reason": "very low precision — raising similarity threshold to filter noise",
        "impact": "precision",
        "confidence": 0.6,
    },
    {
        "dimension": "latency",
        "below": 50.0,
        "param": "top_k",
        "direction": "decrease",
        "reason": "high latency — reducing top_k to process fewer chunks",
        "impact": "latency",
        "confidence": 0.5,
    },
]


class Proposer:
    """Analyzes scores and proposes parameter changes."""

    def __init__(self, current_config: dict[str, object] | None = None) -> None:
        self._current_config = current_config or {}

    def analyze(self, scorecard: ScoreCard) -> list[Proposal]:
        """Analyze a scorecard and return proposals sorted by priority."""
        proposals: list[Proposal] = []

        for rule in _RULES:
            dim = scorecard.get_dimension(str(rule["dimension"]))
            if dim is None:
                continue

            threshold = float(str(rule["below"]))
            if dim.value >= threshold:
                continue

            param = str(rule["param"])
            if param not in TUNABLE_PARAMS:
                continue

            tunable = TUNABLE_PARAMS[param]
            current = self._current_config.get(param, tunable.get("min", 0))
            direction = str(rule["direction"])

            step = tunable.get("step", 1)
            if direction == "increase":
                proposed = min(  # type: ignore[call-overload]
                    current + step,  # type: ignore[operator]
                    tunable["max"],
                )
            else:
                proposed = max(  # type: ignore[call-overload]
                    current - step,  # type: ignore[operator]
                    tunable["min"],
                )

            if proposed == current:
                continue

            proposals.append(
                Proposal(
                    param=param,
                    current_value=current,
                    proposed_value=proposed,
                    reason=str(rule["reason"]),
                    expected_impact=str(rule["impact"]),
                    confidence=float(str(rule["confidence"])),
                    priority=len(proposals),
                )
            )

        proposals.sort(key=lambda p: (-p.confidence, p.priority))
        return proposals

    def best_proposal(self, scorecard: ScoreCard) -> Proposal | None:
        """Return the highest-confidence proposal, or None if no improvements found."""
        proposals = self.analyze(scorecard)
        return proposals[0] if proposals else None
