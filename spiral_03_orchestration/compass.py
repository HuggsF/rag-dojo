"""MendaCompass — pre-action reasoning block.

Analogy for a C dev:
- Before executing a function, you check preconditions (assert != NULL)
- The compass is that assert block for RAG orchestration
- It evaluates tension, calculates confidence, and decides: PROCEED or FALLBACK
- Think of it as the "should I even call this function?" guard

The compass answers: given the current tension field and profile,
what action should the orchestrator take?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from spiral_03_orchestration.confidence import ConfidenceResult, calculate_confidence
from spiral_03_orchestration.tension import TensionMetrics
from spiral_03_orchestration.tension_profile import TensionProfile


class Action(str, Enum):
    """Possible orchestrator actions."""

    RETRIEVE = "RETRIEVE"
    FALLBACK = "FALLBACK"
    SKIP = "SKIP"


@dataclass
class CompassReading:
    """Result of compass evaluation — the decision + reasoning."""

    action: Action
    confidence: ConfidenceResult
    metrics: TensionMetrics
    profile: TensionProfile
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "action": self.action.value,
            "confidence": self.confidence.to_dict(),
            "metrics": self.metrics.to_dict(),
            "profile": self.profile.to_dict(),
            "reason": self.reason,
        }


class MendaCompass:
    """Pre-action reasoning: evaluate tension → decide action.

    The compass reads the tension field and the active profile,
    calculates confidence, and recommends an action.
    """

    def evaluate(
        self,
        metrics: TensionMetrics,
        profile: TensionProfile,
    ) -> CompassReading:
        """Evaluate the tension field and recommend an action.

        Decision logic:
        1. If zero-delta collapse → FALLBACK (state didn't change, need different strategy)
        2. If confidence >= threshold → RETRIEVE (proceed with hybrid retrieval)
        3. If confidence < threshold → FALLBACK (not confident enough)

        Args:
            metrics: Current tension measurements.
            profile: Active tension profile.

        Returns:
            CompassReading with action and reasoning.
        """
        confidence = calculate_confidence(metrics, profile)

        if confidence.zero_delta_collapse:
            return CompassReading(
                action=Action.FALLBACK,
                confidence=confidence,
                metrics=metrics,
                profile=profile,
                reason=(
                    f"Zero-delta collapse: delta_estado={metrics.delta_estado:.4f} "
                    f"< threshold. State did not change — action would be useless."
                ),
            )

        if confidence.passed:
            return CompassReading(
                action=Action.RETRIEVE,
                confidence=confidence,
                metrics=metrics,
                profile=profile,
                reason=(
                    f"Confidence {confidence.score:.4f} >= "
                    f"threshold {profile.confidence_threshold:.4f}. "
                    f"Proceeding with retrieval."
                ),
            )

        return CompassReading(
            action=Action.FALLBACK,
            confidence=confidence,
            metrics=metrics,
            profile=profile,
            reason=(
                f"Confidence {confidence.score:.4f} < "
                f"threshold {profile.confidence_threshold:.4f}. "
                f"Falling back to alternative strategy."
            ),
        )
