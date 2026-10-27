"""TensionMetrics — measures the opposing forces in every RAG decision.

Analogy for a C dev:
- creativity = malloc freedom — how much exploration is allowed
- logic = bounds checking — how strict the convergence is
- delta_estado = diff between two states — did the action actually change anything?
- If delta == 0, the action was a no-op. Like writing to /dev/null.

The tension field is NOT a binary choice. It's a continuous calibration
between exploration (creativity) and exploitation (logic), measured by
whether the state actually changed (delta_estado).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensionMetrics:
    """Snapshot of the tension field at a given moment.

    All values are in [0.0, 1.0].
    """

    creativity: float = 0.5
    logic: float = 0.5
    delta_estado: float = 0.0

    def __post_init__(self):
        self.creativity = self._clamp(self.creativity)
        self.logic = self._clamp(self.logic)
        self.delta_estado = self._clamp(self.delta_estado)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp value to [0.0, 1.0]."""
        return max(0.0, min(1.0, float(value)))

    @property
    def tension_ratio(self) -> float:
        """Ratio between creativity and logic.

        Returns value in [0.0, 1.0]:
        - 0.0 = pure logic (exploitation)
        - 0.5 = balanced
        - 1.0 = pure creativity (exploration)
        """
        total = self.creativity + self.logic
        if total == 0:
            return 0.5
        return self.creativity / total

    @property
    def is_zero_delta(self) -> bool:
        """True if delta_estado is effectively zero (no state change)."""
        from spiral_03_orchestration.config import ZERO_DELTA_THRESHOLD

        return self.delta_estado < ZERO_DELTA_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging/display."""
        return {
            "creativity": round(self.creativity, 4),
            "logic": round(self.logic, 4),
            "delta_estado": round(self.delta_estado, 4),
            "tension_ratio": round(self.tension_ratio, 4),
            "is_zero_delta": self.is_zero_delta,
        }


@dataclass
class TensionHistory:
    """Tracks tension over consecutive queries for streak detection."""

    records: list[TensionMetrics] = field(default_factory=list)

    def record(self, metrics: TensionMetrics) -> None:
        """Append a new tension snapshot."""
        self.records.append(metrics)

    @property
    def zero_delta_streak(self) -> int:
        """Count consecutive zero-delta entries from the end."""
        streak = 0
        for m in reversed(self.records):
            if m.is_zero_delta:
                streak += 1
            else:
                break
        return streak

    @property
    def should_swap_strategy(self) -> bool:
        """True if zero-delta streak exceeds the limit (Inertia Zero Law)."""
        from spiral_03_orchestration.config import ZERO_DELTA_STREAK_LIMIT

        return self.zero_delta_streak >= ZERO_DELTA_STREAK_LIMIT

    def avg_creativity(self, last_n: int = 5) -> float:
        """Average creativity over last N entries."""
        recent = self.records[-last_n:] if self.records else []
        if not recent:
            return 0.5
        return sum(r.creativity for r in recent) / len(recent)

    def avg_logic(self, last_n: int = 5) -> float:
        """Average logic over last N entries."""
        recent = self.records[-last_n:] if self.records else []
        if not recent:
            return 0.5
        return sum(r.logic for r in recent) / len(recent)

    def clear(self) -> None:
        """Reset history."""
        self.records.clear()
