"""Tests for cli.py — unified CLI entry point.

Tests verify CLI argument parsing and command dispatch.
All pipeline operations are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCLIParsing:
    """Tests for CLI argument parsing and dispatch."""

    def test_no_command_exits(self):
        """No command shows help and exits 1."""
        from cli import main

        with patch("sys.argv", ["rag-dojo"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("pipeline_factory.build_pipeline")
    def test_ask_command(self, mock_build):
        """ask command calls build_pipeline and pipeline.query()."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = MagicMock(
            answer="FastAPI is a web framework",
            layer="L3_LLM",
            cost=0.0,
            model="llama-3.3-70b-versatile",
            cached=False,
            metadata={},
        )
        mock_build.return_value = mock_pipeline

        from cli import cmd_ask

        args = MagicMock()
        args.question = "What is FastAPI?"
        args.no_graph = False
        args.no_cognitive = False
        args.verbose = False

        cmd_ask(args)

        mock_build.assert_called_once()
        mock_pipeline.query.assert_called_once_with("What is FastAPI?")

    @patch("pipeline_factory.build_pipeline")
    def test_ask_no_graph(self, mock_build):
        """--no-graph passes enable_graph=False to build_pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = MagicMock(
            answer="answer",
            layer="L3_LLM",
            cost=0.0,
            model="test",
            cached=False,
            metadata={},
        )
        mock_build.return_value = mock_pipeline

        from cli import cmd_ask

        args = MagicMock()
        args.question = "test"
        args.no_graph = True
        args.no_cognitive = False
        args.ollama = False
        args.verbose = False

        cmd_ask(args)

        mock_build.assert_called_once_with(
            enable_graph=False, enable_cognitive=True, use_ollama=False
        )

    @patch("pipeline_factory.build_pipeline")
    def test_ask_shows_cognitive_alerts(self, mock_build):
        """Cognitive alerts are displayed when present."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = MagicMock(
            answer="answer",
            layer="L3_LLM",
            cost=0.0,
            model="test",
            cached=False,
            metadata={
                "cognitive_analysis": {
                    "status": "fail",
                    "alerts": ["REGRESSION: pipeline went from ok → fail"],
                }
            },
        )
        mock_build.return_value = mock_pipeline

        from cli import cmd_ask

        args = MagicMock()
        args.question = "test"
        args.no_graph = False
        args.no_cognitive = False
        args.verbose = False

        # Should not raise — alerts are printed
        cmd_ask(args)


class TestDashboardCommand:
    """Tests for the dashboard command."""

    @patch("spiral_05_economics.dashboard.Dashboard")
    @patch("spiral_05_economics.budget_tracker.BudgetTracker")
    def test_dashboard_calls_display(self, mock_tracker_cls, mock_dashboard_cls):
        """dashboard command creates tracker and dashboard."""
        from cli import cmd_dashboard

        mock_dashboard_cls.return_value.summary.return_value = {
            "daily_cost": 0.0,
            "daily_budget": 5.0,
            "monthly_cost": 0.0,
            "monthly_budget": 150.0,
            "daily_requests": 0,
            "monthly_requests": 0,
            "alert_level": "normal",
        }
        mock_dashboard_cls.return_value.estimate_savings.return_value = {
            "saved_vs_sonnet": 0.0,
            "actual_cost": 0.0,
            "hypothetical_sonnet_cost": 0.0,
        }

        args = MagicMock()
        args.with_query = False
        args.ollama = False
        cmd_dashboard(args)

        mock_tracker_cls.assert_called_once()
        mock_dashboard_cls.assert_called_once()
