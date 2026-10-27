"""Tests for spiral_05_economics.circuit_breaker — budget hard stops."""

import pytest

from spiral_05_economics.circuit_breaker import BudgetExceededError, CircuitBreaker


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_allows_when_under_budget(self, circuit_breaker):
        assert circuit_breaker.is_open() is False
        circuit_breaker.check_or_raise()

    def test_raises_when_daily_exceeded(self, tmp_db):
        from spiral_05_economics.budget_tracker import BudgetTracker

        tracker = BudgetTracker(db_path=tmp_db, daily_budget=0.01, monthly_budget=150.0)
        tracker.record_cost("claude-sonnet", 10000, 5000, 0.05)
        breaker = CircuitBreaker(tracker=tracker)
        assert breaker.is_open() is True
        with pytest.raises(BudgetExceededError, match="Daily"):
            breaker.check_or_raise()
        tracker.close()

    def test_raises_when_monthly_exceeded(self, tmp_db):
        from spiral_05_economics.budget_tracker import BudgetTracker

        tracker = BudgetTracker(db_path=tmp_db, daily_budget=1000.0, monthly_budget=0.01)
        tracker.record_cost("claude-sonnet", 10000, 5000, 0.05)
        breaker = CircuitBreaker(tracker=tracker)
        assert breaker.is_open() is True
        with pytest.raises(BudgetExceededError, match="Monthly"):
            breaker.check_or_raise()
        tracker.close()

    def test_estimate_remaining_daily(self, circuit_breaker):
        remaining = circuit_breaker.remaining_daily()
        assert remaining > 0

    def test_estimate_remaining_monthly(self, circuit_breaker):
        remaining = circuit_breaker.remaining_monthly()
        assert remaining > 0

    def test_status_report(self, circuit_breaker):
        report = circuit_breaker.status()
        assert "open" in report
        assert "daily_remaining" in report
        assert "monthly_remaining" in report
