"""Cycle manager — orchestrates PROPOSE → SOLVE → VERIFY loop.

Analogy for a C dev:
    Like an optimization pass in a compiler: analyze → transform → verify.
    Repeat up to N times. If verification fails, revert the transform.
    The human checkpoint is like -Werror: you MUST approve before continuing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from spiral_06_state_of_art.config import MAX_CYCLES, SCORE_REGRESSION_THRESHOLD
from spiral_06_state_of_art.guardrails import Guardrails
from spiral_06_state_of_art.proposer import Proposal, Proposer
from spiral_06_state_of_art.reporter import CycleReport, Reporter
from spiral_06_state_of_art.scorer import ScoreCard, Scorer
from spiral_06_state_of_art.solver import Solver, SolverResult
from spiral_06_state_of_art.verifier import AnswerFn, VerificationResult, Verifier


@dataclass
class CycleResult:
    """Result of a single PROPOSE → SOLVE → VERIFY cycle."""

    cycle_number: int
    proposal: Proposal | None
    solver_result: SolverResult | None
    verification: VerificationResult | None
    outcome: str  # "improved", "reverted", "no_proposal", "guardrail_blocked"
    reverted: bool = False

    def __repr__(self) -> str:
        return f"CycleResult(cycle={self.cycle_number}, outcome={self.outcome})"


@dataclass
class RunResult:
    """Result of running the full cycle manager."""

    cycles: list[CycleResult] = field(default_factory=list)
    total_improvement: float = 0.0
    final_scorecard: ScoreCard | None = None
    stopped_reason: str = ""
    report: CycleReport | None = None

    @property
    def cycles_run(self) -> int:
        return len(self.cycles)

    def __repr__(self) -> str:
        return (
            f"RunResult(cycles={self.cycles_run}, "
            f"improvement={self.total_improvement:+.2f}, "
            f"stopped={self.stopped_reason!r})"
        )


class CycleManager:
    """Orchestrates the PROPOSE → SOLVE → VERIFY loop."""

    def __init__(
        self,
        config: dict[str, object],
        scorer: Scorer | None = None,
        proposer: Proposer | None = None,
        guardrails: Guardrails | None = None,
        verifier: Verifier | None = None,
        reporter: Reporter | None = None,
        max_cycles: int = MAX_CYCLES,
        regression_threshold: float = SCORE_REGRESSION_THRESHOLD,
    ) -> None:
        self._config = dict(config)
        self._scorer = scorer or Scorer()
        self._proposer = proposer or Proposer(current_config=config)
        self._guardrails = guardrails or Guardrails()
        self._verifier = verifier or Verifier()
        self._reporter = reporter or Reporter()
        self._max_cycles = max_cycles
        self._regression_threshold = regression_threshold

    def run(
        self,
        initial_scorecard: ScoreCard,
        answer_fn: AnswerFn,
        checkpoint_fn: Callable[[int, CycleResult], bool] | None = None,
    ) -> RunResult:
        """Run up to max_cycles of PROPOSE → SOLVE → VERIFY.

        Args:
            initial_scorecard: Baseline scores before any changes.
            answer_fn: Function that takes a question and returns (answer, latency, cost).
            checkpoint_fn: Optional human checkpoint — called after each cycle.
                Returns True to continue, False to stop.
                If None, all cycles run automatically.
        """
        result = RunResult()
        current_scorecard = initial_scorecard
        consecutive_failures = 0

        for cycle_num in range(1, self._max_cycles + 1):
            # Check guardrails before starting cycle
            guard_check = self._guardrails.check_circuit_breakers(
                consecutive_failures=consecutive_failures,
                cycle_depth=cycle_num - 1,
            )
            if not guard_check.passed:
                result.cycles.append(
                    CycleResult(
                        cycle_number=cycle_num,
                        proposal=None,
                        solver_result=None,
                        verification=None,
                        outcome="guardrail_blocked",
                    )
                )
                result.stopped_reason = f"guardrail: {guard_check.violations[0].message}"
                break

            # PROPOSE
            proposal = self._proposer.best_proposal(current_scorecard)
            if proposal is None:
                result.cycles.append(
                    CycleResult(
                        cycle_number=cycle_num,
                        proposal=None,
                        solver_result=None,
                        verification=None,
                        outcome="no_proposal",
                    )
                )
                result.stopped_reason = "no improvement proposals found"
                break

            # SOLVE
            solver = Solver(self._config)
            solver_result = solver.apply_proposals([proposal])

            if not solver_result.success:
                consecutive_failures += 1
                result.cycles.append(
                    CycleResult(
                        cycle_number=cycle_num,
                        proposal=proposal,
                        solver_result=solver_result,
                        verification=None,
                        outcome="solve_failed",
                    )
                )
                continue

            # VERIFY
            verification = self._verifier.verify(current_scorecard, answer_fn)

            if verification.regressed:
                # Auto-revert on regression
                solver.revert()
                consecutive_failures += 1
                result.cycles.append(
                    CycleResult(
                        cycle_number=cycle_num,
                        proposal=proposal,
                        solver_result=solver_result,
                        verification=verification,
                        outcome="reverted",
                        reverted=True,
                    )
                )
                continue

            # Success — apply config changes
            self._config = solver.config
            self._proposer = Proposer(current_config=self._config)
            consecutive_failures = 0

            if verification.improved:
                result.total_improvement += verification.deltas.get("total", 0.0)
                current_scorecard = verification.after

            cycle_result = CycleResult(
                cycle_number=cycle_num,
                proposal=proposal,
                solver_result=solver_result,
                verification=verification,
                outcome="improved" if verification.improved else "neutral",
            )
            result.cycles.append(cycle_result)

            # Human checkpoint
            if checkpoint_fn is not None and not checkpoint_fn(cycle_num, cycle_result):
                result.stopped_reason = f"human checkpoint stopped at cycle {cycle_num}"
                break

        if not result.stopped_reason:
            result.stopped_reason = f"completed {len(result.cycles)} cycles"

        result.final_scorecard = current_scorecard
        result.report = self._reporter.generate(result.cycles, initial_scorecard, current_scorecard)  # type: ignore[arg-type]
        return result
