"""Tests for spiral_06_state_of_art.guardrails."""

from spiral_06_state_of_art.guardrails import GuardrailViolation


class TestGuardrailViolation:
    def test_repr(self):
        v = GuardrailViolation(rule="test", message="test msg")
        assert "test" in repr(v)


class TestGuardrails:
    def test_blast_radius_within_limits(self, guardrails):
        check = guardrails.check_blast_radius(files_modified=3)
        assert check.passed is True

    def test_blast_radius_exceeded_files(self, guardrails):
        check = guardrails.check_blast_radius(files_modified=10)
        assert check.passed is False
        assert any(v.rule == "max_files" for v in check.violations)

    def test_blast_radius_exceeded_prs(self, guardrails):
        check = guardrails.check_blast_radius(prs_created=5)
        assert check.passed is False
        assert any(v.rule == "max_prs" for v in check.violations)

    def test_blast_radius_exceeded_ecosystems(self, guardrails):
        check = guardrails.check_blast_radius(ecosystems_affected=3)
        assert check.passed is False

    def test_blast_radius_at_limit_warns(self, guardrails):
        check = guardrails.check_blast_radius(files_modified=5)
        assert check.passed is True
        assert len(check.warnings) > 0

    def test_immutable_check_pass(self, guardrails):
        check = guardrails.check_immutable(["some_file.py"])
        assert check.passed is True

    def test_immutable_check_fail(self, guardrails):
        check = guardrails.check_immutable(["CLAUDE.md", "some_file.py"])
        assert check.passed is False
        assert len(check.violations) == 1

    def test_circuit_breaker_budget(self, guardrails):
        check = guardrails.check_circuit_breakers(budget_exceeded=True)
        assert check.passed is False

    def test_circuit_breaker_failures(self, guardrails):
        check = guardrails.check_circuit_breakers(consecutive_failures=3)
        assert check.passed is False

    def test_circuit_breaker_regression(self, guardrails):
        check = guardrails.check_circuit_breakers(score_regression=6.0)
        assert check.passed is False

    def test_circuit_breaker_depth(self, guardrails):
        check = guardrails.check_circuit_breakers(cycle_depth=3)
        assert check.passed is False

    def test_circuit_breaker_cost_spike(self, guardrails):
        check = guardrails.check_circuit_breakers(cost_spike_pct=300)
        assert check.passed is False

    def test_circuit_breaker_all_clear(self, guardrails):
        check = guardrails.check_circuit_breakers()
        assert check.passed is True

    def test_full_check(self, guardrails):
        check = guardrails.full_check(
            files_modified=2,
            files_list=["ok.py"],
        )
        assert check.passed is True

    def test_full_check_fails_on_immutable(self, guardrails):
        check = guardrails.full_check(
            files_modified=1,
            files_list=["CLAUDE.md"],
        )
        assert check.passed is False
