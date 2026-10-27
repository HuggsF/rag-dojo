"""Tests for spiral_05_economics.dashboard — CLI reporting."""

from spiral_05_economics.dashboard import Dashboard


class TestDashboard:
    """Tests for Dashboard."""

    def test_daily_report_string(self, budget_tracker):
        dash = Dashboard(tracker=budget_tracker)
        report = dash.daily_report()
        assert isinstance(report, str)
        assert "Today" in report or "today" in report.lower()

    def test_monthly_report_string(self, budget_tracker):
        dash = Dashboard(tracker=budget_tracker)
        report = dash.monthly_report()
        assert isinstance(report, str)
        assert "Month" in report or "month" in report.lower()

    def test_report_with_data(self, budget_tracker):
        budget_tracker.record_cost("claude-sonnet", 1000, 500, 0.01)
        budget_tracker.record_cost("claude-haiku", 2000, 1000, 0.001)
        dash = Dashboard(tracker=budget_tracker)
        report = dash.daily_report()
        assert "$" in report or "0.01" in report

    def test_top_models_report(self, budget_tracker):
        budget_tracker.record_cost("claude-sonnet", 1000, 500, 0.01)
        budget_tracker.record_cost("claude-sonnet", 2000, 1000, 0.02)
        budget_tracker.record_cost("claude-haiku", 3000, 1500, 0.001)
        dash = Dashboard(tracker=budget_tracker)
        top = dash.top_models()
        assert isinstance(top, list)
        assert len(top) == 2
        assert top[0]["model"] == "claude-sonnet"

    def test_savings_report(self, budget_tracker):
        budget_tracker.record_cost("llama-3.3-70b-versatile", 5000, 2000, 0.0)
        dash = Dashboard(tracker=budget_tracker)
        savings = dash.estimate_savings()
        assert isinstance(savings, dict)
        assert "saved_vs_sonnet" in savings
        assert savings["saved_vs_sonnet"] > 0

    def test_monthly_report_with_model_breakdown(self, budget_tracker):
        """Cover monthly_report by_model branch (lines 55-57)."""
        budget_tracker.record_cost("claude-sonnet", 1000, 500, 0.01)
        budget_tracker.record_cost("claude-haiku", 2000, 1000, 0.001)
        dash = Dashboard(tracker=budget_tracker)
        report = dash.monthly_report()
        assert "By model:" in report
        assert "claude-sonnet" in report
        assert "claude-haiku" in report

    def test_top_models_empty_when_no_data(self, budget_tracker):
        """Cover top_models empty return (line 64)."""
        dash = Dashboard(tracker=budget_tracker)
        top = dash.top_models()
        assert top == []

    def test_display_method(self, budget_tracker):
        """Cover display() method."""
        dash = Dashboard(tracker=budget_tracker)
        # Should not raise
        dash.display()

    def test_summary_dict(self, budget_tracker):
        dash = Dashboard(tracker=budget_tracker)
        summary = dash.summary()
        assert isinstance(summary, dict)
        assert "daily_cost" in summary
        assert "monthly_cost" in summary
        assert "daily_budget" in summary
        assert "monthly_budget" in summary
