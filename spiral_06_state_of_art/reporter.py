"""Reporter — generates cycle reports showing what changed and score deltas.

Analogy for a C dev:
    Like `diff --stat` + `perf diff` combined — shows what changed in the code
    AND what changed in performance. One report per optimization run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from spiral_06_state_of_art.scorer import ScoreCard


@dataclass
class CycleReport:
    """Report for a complete optimization run."""

    timestamp: str
    cycles_run: int
    total_improvement: float
    initial_scores: dict[str, object]
    final_scores: dict[str, object]
    cycle_details: list[dict[str, object]] = field(default_factory=list)

    def to_text(self) -> str:
        """Generate human-readable text report."""
        lines = [
            f"=== Optimization Report ({self.timestamp}) ===",
            f"Cycles run: {self.cycles_run}",
            f"Total improvement: {self.total_improvement:+.2f} points",
            "",
            "--- Initial Scores ---",
        ]

        if isinstance(self.initial_scores, dict) and "dimensions" in self.initial_scores:
            dims = self.initial_scores["dimensions"]
            if isinstance(dims, dict):
                for name, data in dims.items():
                    val = data.get("value", 0) if isinstance(data, dict) else 0
                    lines.append(f"  {name}: {val:.1f}")
            lines.append(f"  TOTAL: {self.initial_scores.get('total', 0):.1f}")

        lines.append("")
        lines.append("--- Final Scores ---")

        if isinstance(self.final_scores, dict) and "dimensions" in self.final_scores:
            dims = self.final_scores["dimensions"]
            if isinstance(dims, dict):
                for name, data in dims.items():
                    val = data.get("value", 0) if isinstance(data, dict) else 0
                    lines.append(f"  {name}: {val:.1f}")
            lines.append(f"  TOTAL: {self.final_scores.get('total', 0):.1f}")

        if self.cycle_details:
            lines.append("")
            lines.append("--- Cycle Details ---")
            for detail in self.cycle_details:
                cycle_num = detail.get("cycle", "?")
                outcome = detail.get("outcome", "unknown")
                param = detail.get("param", "—")
                delta = detail.get("delta_total", 0)
                lines.append(f"  Cycle {cycle_num}: {outcome} (param={param}, delta={delta:+.2f})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CycleReport(cycles={self.cycles_run}, improvement={self.total_improvement:+.2f})"


class Reporter:
    """Generates reports from cycle results."""

    def generate(
        self,
        cycles: list[object],
        initial_scorecard: ScoreCard,
        final_scorecard: ScoreCard,
    ) -> CycleReport:
        """Generate a report from cycle results."""
        cycle_details: list[dict[str, object]] = []

        for cycle in cycles:
            detail: dict[str, object] = {
                "cycle": getattr(cycle, "cycle_number", 0),
                "outcome": getattr(cycle, "outcome", "unknown"),
            }

            proposal = getattr(cycle, "proposal", None)
            if proposal is not None:
                detail["param"] = getattr(proposal, "param", "—")
                detail["old_value"] = getattr(proposal, "current_value", None)
                detail["new_value"] = getattr(proposal, "proposed_value", None)
                detail["reason"] = getattr(proposal, "reason", "")

            verification = getattr(cycle, "verification", None)
            if verification is not None:
                deltas = getattr(verification, "deltas", {})
                detail["delta_total"] = deltas.get("total", 0.0)
            else:
                detail["delta_total"] = 0.0

            cycle_details.append(detail)

        total_improvement = final_scorecard.total - initial_scorecard.total

        return CycleReport(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            cycles_run=len(cycles),
            total_improvement=round(total_improvement, 2),
            initial_scores=initial_scorecard.to_dict(),
            final_scores=final_scorecard.to_dict(),
            cycle_details=cycle_details,
        )
