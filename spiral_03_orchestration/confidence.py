"""Confidence calculation — the product of tension forces.

Analogy for a C dev:
- Confidence = (Creativity × Logic) × DeltaEstado
- This is like a checksum: if ANY factor is zero, the whole thing collapses
- Zero-delta collapse: if the state didn't change, confidence is ZERO
  regardless of how good creativity and logic look
- It's the "null pointer check" of decision-making

The formula uses weighted values from the active tension profile:
  weighted_creativity = creativity * profile.creativity_weight
  weighted_logic = logic * profile.logic_weight
  confidence = (weighted_creativity * weighted_logic) * delta_estado
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spiral_03_orchestration.tension import TensionMetrics
from spiral_03_orchestration.tension_profile import TensionProfile


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""

    score: float
    weighted_creativity: float
    weighted_logic: float
    delta_estado: float
    threshold: float
    passed: bool
    zero_delta_collapse: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "score": round(self.score, 4),
            "weighted_creativity": round(self.weighted_creativity, 4),
            "weighted_logic": round(self.weighted_logic, 4),
            "delta_estado": round(self.delta_estado, 4),
            "threshold": round(self.threshold, 4),
            "passed": self.passed,
            "zero_delta_collapse": self.zero_delta_collapse,
        }


def calculate_confidence(
    metrics: TensionMetrics,
    profile: TensionProfile,
) -> ConfidenceResult:
    """Calculate confidence from tension metrics and active profile.

    Formula: Confidence = (Creativity × creativity_weight) × (Logic × logic_weight) × DeltaEstado

    If delta_estado < ZERO_DELTA_THRESHOLD → confidence collapses to 0.0
    (the "Inertia Zero Law": if nothing changed, the action was useless).

    Args:
        metrics: Current tension measurements.
        profile: Active tension profile with weights and threshold.

    Returns:
        ConfidenceResult with score, components, and pass/fail.
    """
    weighted_creativity = metrics.creativity * profile.creativity_weight
    weighted_logic = metrics.logic * profile.logic_weight

    # Zero-delta collapse: if state didn't change, confidence is zero
    if metrics.is_zero_delta:
        return ConfidenceResult(
            score=0.0,
            weighted_creativity=weighted_creativity,
            weighted_logic=weighted_logic,
            delta_estado=metrics.delta_estado,
            threshold=profile.confidence_threshold,
            passed=False,
            zero_delta_collapse=True,
        )

    score = weighted_creativity * weighted_logic * metrics.delta_estado

    return ConfidenceResult(
        score=score,
        weighted_creativity=weighted_creativity,
        weighted_logic=weighted_logic,
        delta_estado=metrics.delta_estado,
        threshold=profile.confidence_threshold,
        passed=score >= profile.confidence_threshold,
        zero_delta_collapse=False,
    )
