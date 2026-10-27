"""Scorer — evaluates RAG system dimensions with numeric scores.

Analogy for a C dev:
    Like running `perf stat` on your binary — you get measurable metrics
    (cache hits, branch misses, IPC) instead of "it feels fast".
    Scorer gives you numbers: precision, recall, latency, cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spiral_06_state_of_art.benchmark_suite import BenchmarkReport


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    name: str
    value: float  # 0.0 to 100.0
    weight: float  # contribution to total
    details: dict[str, object] = field(default_factory=dict)

    def weighted(self) -> float:
        """Return weighted score."""
        return self.value * self.weight

    def __repr__(self) -> str:
        return f"DimensionScore({self.name}={self.value:.1f}, weight={self.weight})"


@dataclass
class ScoreCard:
    """Complete evaluation scorecard across all dimensions."""

    dimensions: list[DimensionScore]
    total: float
    benchmark_report: BenchmarkReport | None = None

    def get_dimension(self, name: str) -> DimensionScore | None:
        """Get a specific dimension by name."""
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict."""
        return {
            "total": round(self.total, 2),
            "dimensions": {
                d.name: {"value": round(d.value, 2), "weight": d.weight, "details": d.details}
                for d in self.dimensions
            },
        }

    def __repr__(self) -> str:
        dims = ", ".join(f"{d.name}={d.value:.1f}" for d in self.dimensions)
        return f"ScoreCard(total={self.total:.1f}, {dims})"


# Latency tiers (seconds) — maps to scores
_LATENCY_TIERS: list[tuple[float, float]] = [
    (0.5, 100.0),  # < 0.5s = perfect
    (1.0, 85.0),  # < 1s = great
    (2.0, 70.0),  # < 2s = good
    (5.0, 50.0),  # < 5s = acceptable
    (10.0, 30.0),  # < 10s = slow
    (float("inf"), 10.0),  # > 10s = very slow
]

# Cost tiers (USD per query) — maps to scores
_COST_TIERS: list[tuple[float, float]] = [
    (0.0, 100.0),  # free = perfect
    (0.001, 90.0),  # < $0.001 = excellent
    (0.01, 70.0),  # < $0.01 = good
    (0.05, 50.0),  # < $0.05 = acceptable
    (0.10, 30.0),  # < $0.10 = expensive
    (float("inf"), 10.0),  # > $0.10 = very expensive
]


def _tier_score(value: float, tiers: list[tuple[float, float]]) -> float:
    """Map a value to a score based on tier thresholds."""
    for threshold, score in tiers:
        if value <= threshold:
            return score
    return tiers[-1][1]


class Scorer:
    """Evaluates RAG quality across multiple dimensions."""

    def __init__(
        self,
        precision_weight: float = 0.35,
        recall_weight: float = 0.30,
        latency_weight: float = 0.15,
        cost_weight: float = 0.20,
    ) -> None:
        self._weights = {
            "precision": precision_weight,
            "recall": recall_weight,
            "latency": latency_weight,
            "cost": cost_weight,
        }
        total_w = sum(self._weights.values())
        if abs(total_w - 1.0) > 0.01:
            # Normalize weights
            self._weights = {k: v / total_w for k, v in self._weights.items()}

    def score_benchmark(self, report: BenchmarkReport) -> ScoreCard:
        """Score a benchmark report across all dimensions."""
        precision_score = report.precision_at_k * 100
        recall_score = report.mean_keyword_recall * 100
        latency_score = _tier_score(report.mean_latency_s, _LATENCY_TIERS)

        avg_cost = report.total_cost / max(report.total_questions, 1)
        cost_score = _tier_score(avg_cost, _COST_TIERS)

        dimensions = [
            DimensionScore(
                name="precision",
                value=precision_score,
                weight=self._weights["precision"],
                details={
                    "precision_at_k": report.precision_at_k,
                    "hits": sum(1 for r in report.results if r.hit),
                    "total": report.total_questions,
                },
            ),
            DimensionScore(
                name="recall",
                value=recall_score,
                weight=self._weights["recall"],
                details={"mean_keyword_recall": report.mean_keyword_recall},
            ),
            DimensionScore(
                name="latency",
                value=latency_score,
                weight=self._weights["latency"],
                details={"mean_latency_s": report.mean_latency_s},
            ),
            DimensionScore(
                name="cost",
                value=cost_score,
                weight=self._weights["cost"],
                details={"total_cost": report.total_cost, "avg_cost_per_query": avg_cost},
            ),
        ]

        total = sum(d.weighted() for d in dimensions)

        return ScoreCard(
            dimensions=dimensions,
            total=total,
            benchmark_report=report,
        )

    def compare(self, before: ScoreCard, after: ScoreCard) -> dict[str, float]:
        """Compare two scorecards and return deltas per dimension."""
        deltas: dict[str, float] = {}
        for dim_after in after.dimensions:
            dim_before = before.get_dimension(dim_after.name)
            before_val = dim_before.value if dim_before else 0.0
            deltas[dim_after.name] = round(dim_after.value - before_val, 2)
        deltas["total"] = round(after.total - before.total, 2)
        return deltas
