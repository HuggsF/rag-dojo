"""Tests for spiral_05_economics.budget_tracker — SQLite budget tracking."""

from spiral_05_economics.budget_tracker import BudgetStatus, BudgetTracker


class TestBudgetTrackerInit:
    """Tests for BudgetTracker initialization."""

    def test_creates_db_on_first_access(self, tmp_db):
        tracker = BudgetTracker(db_path=tmp_db)
        _ = tracker.conn
        from pathlib import Path

        assert Path(tmp_db).exists()
        tracker.close()

    def test_close_and_reopen(self, tmp_db):
        tracker = BudgetTracker(db_path=tmp_db)
        tracker.record_cost("llama-3.3-70b-versatile", 100, 50, 0.0)
        tracker.close()

        tracker2 = BudgetTracker(db_path=tmp_db)
        summary = tracker2.daily_summary()
        assert summary.total_requests == 1
        tracker2.close()


class TestRecordCost:
    """Tests for recording costs."""

    def test_record_returns_row_id(self, budget_tracker):
        row_id = budget_tracker.record_cost("llama-3.3-70b-versatile", 100, 50, 0.0)
        assert row_id > 0

    def test_record_multiple(self, budget_tracker):
        id1 = budget_tracker.record_cost("llama-3.3-70b-versatile", 100, 50, 0.0)
        id2 = budget_tracker.record_cost("claude-sonnet", 200, 100, 0.006)
        assert id2 > id1

    def test_record_with_query(self, budget_tracker):
        budget_tracker.record_cost("claude-sonnet", 200, 100, 0.006, query="What is FastAPI?")
        summary = budget_tracker.daily_summary()
        assert summary.total_requests == 1


class TestDailySummary:
    """Tests for daily summary."""

    def test_empty_summary(self, budget_tracker):
        summary = budget_tracker.daily_summary()
        assert summary.total_cost == 0.0
        assert summary.total_requests == 0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0

    def test_summary_aggregates_costs(self, budget_tracker):
        budget_tracker.record_cost("claude-sonnet", 1000, 500, 0.01)
        budget_tracker.record_cost("claude-sonnet", 2000, 1000, 0.02)
        summary = budget_tracker.daily_summary()
        assert summary.total_cost == 0.03
        assert summary.total_requests == 2
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500

    def test_summary_per_model(self, budget_tracker):
        budget_tracker.record_cost("claude-sonnet", 1000, 500, 0.01)
        budget_tracker.record_cost("claude-haiku", 2000, 1000, 0.001)
        summary = budget_tracker.daily_summary()
        assert len(summary.by_model) == 2
        assert "claude-sonnet" in summary.by_model
        assert "claude-haiku" in summary.by_model


class TestMonthlySummary:
    """Tests for monthly summary."""

    def test_empty_monthly(self, budget_tracker):
        summary = budget_tracker.monthly_summary()
        assert summary.total_cost == 0.0
        assert summary.total_requests == 0

    def test_monthly_includes_today(self, budget_tracker):
        budget_tracker.record_cost("claude-sonnet", 1000, 500, 0.01)
        summary = budget_tracker.monthly_summary()
        assert summary.total_cost == 0.01
        assert summary.total_requests == 1


class TestBudgetStatus:
    """Tests for budget status checks."""

    def test_under_budget(self, budget_tracker):
        status = budget_tracker.check_budget()
        assert isinstance(status, BudgetStatus)
        assert status.daily_ok is True
        assert status.monthly_ok is True
        assert status.alert_level == "normal"

    def test_daily_exceeded(self, tmp_db):
        tracker = BudgetTracker(db_path=tmp_db, daily_budget=0.01, monthly_budget=150.0)
        tracker.record_cost("claude-sonnet", 10000, 5000, 0.05)
        status = tracker.check_budget()
        assert status.daily_ok is False
        assert status.alert_level == "exceeded"
        tracker.close()

    def test_alert_threshold(self, tmp_db):
        tracker = BudgetTracker(
            db_path=tmp_db,
            daily_budget=1.0,
            monthly_budget=150.0,
            alert_pct=80.0,
        )
        tracker.record_cost("claude-sonnet", 100000, 50000, 0.85)
        status = tracker.check_budget()
        assert status.daily_ok is True
        assert status.alert_level in ("warning", "critical")
        tracker.close()
