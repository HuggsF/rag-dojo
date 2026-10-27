"""Tests for spiral_06_state_of_art.cycle_manager."""

from unittest.mock import MagicMock, patch

from spiral_06_state_of_art.cycle_manager import CycleManager, CycleResult, RunResult
from spiral_06_state_of_art.guardrails import GuardrailCheck, Guardrails, GuardrailViolation
from spiral_06_state_of_art.solver import SolverAction, SolverResult
from spiral_06_state_of_art.verifier import VerificationResult


class TestCycleManager:
    def test_run_with_no_proposals(
        self, sample_config, good_scorecard, scorer, guardrails, verifier, reporter
    ):
        """Good scores = no proposals = early stop."""
        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=guardrails,
            verifier=verifier,
            reporter=reporter,
        )

        def answer_fn(q):
            return ("good answer", 0.1, 0.0)

        result = manager.run(good_scorecard, answer_fn)
        assert isinstance(result, RunResult)
        assert result.stopped_reason == "no improvement proposals found"

    def test_run_with_bad_scores_proposes(
        self, sample_config, bad_scorecard, scorer, guardrails, reporter, benchmark_suite
    ):
        """Bad scores should trigger proposals."""
        verifier = __import__("spiral_06_state_of_art.verifier", fromlist=["Verifier"]).Verifier(
            suite=benchmark_suite, scorer=scorer
        )

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=guardrails,
            verifier=verifier,
            reporter=reporter,
            max_cycles=1,
        )

        def answer_fn(q):
            return (
                "FastAPI is a fast python web framework with middleware and depends injection for request processing",
                0.2,
                0.001,
            )

        result = manager.run(bad_scorecard, answer_fn)
        assert result.cycles_run >= 1
        assert result.cycles[0].proposal is not None

    def test_run_respects_max_cycles(
        self, sample_config, bad_scorecard, scorer, guardrails, reporter, benchmark_suite
    ):
        verifier = __import__("spiral_06_state_of_art.verifier", fromlist=["Verifier"]).Verifier(
            suite=benchmark_suite, scorer=scorer
        )

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=guardrails,
            verifier=verifier,
            reporter=reporter,
            max_cycles=2,
        )

        def answer_fn(q):
            return ("FastAPI is a fast python web framework", 0.2, 0.001)

        result = manager.run(bad_scorecard, answer_fn)
        assert result.cycles_run <= 2

    def test_checkpoint_can_stop(
        self, sample_config, bad_scorecard, scorer, guardrails, reporter, benchmark_suite
    ):
        verifier = __import__("spiral_06_state_of_art.verifier", fromlist=["Verifier"]).Verifier(
            suite=benchmark_suite, scorer=scorer
        )

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=guardrails,
            verifier=verifier,
            reporter=reporter,
            max_cycles=3,
        )

        def answer_fn(q):
            return ("FastAPI is a fast python web framework", 0.2, 0.001)

        def checkpoint(cycle_num, cycle_result):
            return False  # always stop

        result = manager.run(bad_scorecard, answer_fn, checkpoint_fn=checkpoint)
        assert "human checkpoint" in result.stopped_reason

    def test_generates_report(
        self, sample_config, bad_scorecard, scorer, guardrails, reporter, benchmark_suite
    ):
        verifier = __import__("spiral_06_state_of_art.verifier", fromlist=["Verifier"]).Verifier(
            suite=benchmark_suite, scorer=scorer
        )

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=guardrails,
            verifier=verifier,
            reporter=reporter,
            max_cycles=1,
        )

        def answer_fn(q):
            return ("FastAPI is a fast python web framework", 0.2, 0.001)

        result = manager.run(bad_scorecard, answer_fn)
        assert result.report is not None
        assert result.report.cycles_run >= 1


class TestCycleResult:
    def test_repr(self):
        cr = CycleResult(
            cycle_number=1,
            proposal=None,
            solver_result=None,
            verification=None,
            outcome="no_proposal",
        )
        assert "cycle=1" in repr(cr)


class TestCycleManagerEdgePaths:
    """Tests covering guardrail_blocked, solve_failed, and reverted paths."""

    def test_guardrail_blocked_stops_cycle(
        self, sample_config, bad_scorecard, scorer, reporter, benchmark_suite
    ):
        """When guardrails block, cycle stops with guardrail_blocked outcome."""
        # Guardrails that always block (consecutive_failures threshold = 0)
        blocked_guardrails = MagicMock()
        blocked_guardrails.check_circuit_breakers.return_value = GuardrailCheck(
            passed=False,
            violations=[
                GuardrailViolation(
                    rule="budget_exceeded",
                    message="Budget has been exceeded",
                )
            ],
        )

        verifier = (__import__("spiral_06_state_of_art.verifier", fromlist=["Verifier"])).Verifier(
            suite=benchmark_suite, scorer=scorer
        )

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=blocked_guardrails,
            verifier=verifier,
            reporter=reporter,
            max_cycles=3,
        )

        def answer_fn(q):
            return ("answer", 0.2, 0.001)

        result = manager.run(bad_scorecard, answer_fn)
        assert result.cycles_run >= 1
        assert result.cycles[0].outcome == "guardrail_blocked"
        assert "guardrail" in result.stopped_reason

    def test_solve_failed_continues_cycle(
        self, sample_config, bad_scorecard, scorer, reporter, benchmark_suite
    ):
        """When solver fails, cycle records solve_failed and continues."""
        verifier = (__import__("spiral_06_state_of_art.verifier", fromlist=["Verifier"])).Verifier(
            suite=benchmark_suite, scorer=scorer
        )

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=Guardrails(),
            verifier=verifier,
            reporter=reporter,
            max_cycles=2,
        )

        # Patch Solver.apply_proposals to return a failed result
        failed_result = SolverResult(
            actions=[
                SolverAction(
                    param="chunk_size",
                    old_value=512,
                    new_value=99999,
                    applied=False,
                    error="Value outside bounds",
                )
            ],
            config_snapshot_before=sample_config,
            config_snapshot_after=sample_config,
        )

        def answer_fn(q):
            return ("FastAPI is a web framework", 0.2, 0.001)

        with patch("spiral_06_state_of_art.cycle_manager.Solver") as MockSolver:
            mock_instance = MockSolver.return_value
            mock_instance.apply_proposals.return_value = failed_result
            result = manager.run(bad_scorecard, answer_fn)

        solve_failed_cycles = [c for c in result.cycles if c.outcome == "solve_failed"]
        assert len(solve_failed_cycles) >= 1

    def test_reverted_on_regression(
        self, sample_config, bad_scorecard, scorer, reporter, benchmark_suite
    ):
        """When verification detects regression, cycle reverts."""
        regressed_verification = VerificationResult(
            before=bad_scorecard,
            after=bad_scorecard,
            deltas={"total": -10.0},
            improved=False,
            regressed=True,
            recommendation="revert",
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = regressed_verification

        manager = CycleManager(
            config=sample_config,
            scorer=scorer,
            guardrails=Guardrails(),
            verifier=mock_verifier,
            reporter=reporter,
            max_cycles=2,
        )

        def answer_fn(q):
            return ("FastAPI is a web framework", 0.2, 0.001)

        result = manager.run(bad_scorecard, answer_fn)

        reverted_cycles = [c for c in result.cycles if c.outcome == "reverted"]
        assert len(reverted_cycles) >= 1
        assert reverted_cycles[0].reverted is True


class TestRunResult:
    def test_repr(self):
        rr = RunResult()
        assert "cycles=0" in repr(rr)
