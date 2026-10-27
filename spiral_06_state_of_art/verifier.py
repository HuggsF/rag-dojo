"""Verifier — runs benchmarks before/after changes and compares scores.

Analogy for a C dev:
    Like running `make test` before and after a patch, then diffing results.
    If tests regressed → revert. If improved → keep. Simple as that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from spiral_06_state_of_art.benchmark_suite import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSuite,
)
from spiral_06_state_of_art.config import MIN_IMPROVEMENT_THRESHOLD, SCORE_REGRESSION_THRESHOLD
from spiral_06_state_of_art.scorer import ScoreCard, Scorer


@dataclass
class VerificationResult:
    """Result of verifying a change by comparing before/after benchmarks."""

    before: ScoreCard
    after: ScoreCard
    deltas: dict[str, float]
    improved: bool
    regressed: bool
    recommendation: str  # "keep", "revert", "neutral"
    details: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"VerificationResult({self.recommendation}, "
            f"delta_total={self.deltas.get('total', 0):+.2f}, "
            f"improved={self.improved}, regressed={self.regressed})"
        )


# Type for an answer function: question → (answer, latency_s, cost)
AnswerFn = Callable[[str], tuple[str, float, float]]


class Verifier:
    """Runs before/after benchmarks and decides whether to keep or revert."""

    def __init__(
        self,
        suite: BenchmarkSuite | None = None,
        scorer: Scorer | None = None,
        regression_threshold: float = SCORE_REGRESSION_THRESHOLD,
        improvement_threshold: float = MIN_IMPROVEMENT_THRESHOLD,
    ) -> None:
        self._suite = suite or BenchmarkSuite()
        self._scorer = scorer or Scorer()
        self._regression_threshold = regression_threshold
        self._improvement_threshold = improvement_threshold

    def run_benchmark(self, answer_fn: AnswerFn) -> tuple[BenchmarkReport, ScoreCard]:
        """Run the full benchmark suite using the provided answer function."""
        results: list[BenchmarkResult] = []

        for q in self._suite.questions:
            answer, latency, cost = answer_fn(q.question)
            result = self._suite.evaluate_answer(q, answer, latency, cost)
            results.append(result)

        report = self._suite.compile_report(results)
        scorecard = self._scorer.score_benchmark(report)
        return report, scorecard

    def verify(
        self,
        before_scorecard: ScoreCard,
        after_answer_fn: AnswerFn,
    ) -> VerificationResult:
        """Run post-change benchmark and compare with before scores."""
        _, after_scorecard = self.run_benchmark(after_answer_fn)
        deltas = self._scorer.compare(before_scorecard, after_scorecard)

        total_delta = deltas.get("total", 0.0)
        regressed = total_delta < -self._regression_threshold
        improved = total_delta > self._improvement_threshold

        if regressed:
            recommendation = "revert"
        elif improved:
            recommendation = "keep"
        else:
            recommendation = "neutral"

        return VerificationResult(
            before=before_scorecard,
            after=after_scorecard,
            deltas=deltas,
            improved=improved,
            regressed=regressed,
            recommendation=recommendation,
            details={
                "regression_threshold": self._regression_threshold,
                "improvement_threshold": self._improvement_threshold,
            },
        )
