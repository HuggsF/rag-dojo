"""Tests for spiral_04_cognitive.sample_bot — docs_bot."""

from unittest.mock import patch

import pytest

from spiral_04_cognitive.sample_bot import DocsBot, main


class TestDocsBotSimulation:
    """Tests for DocsBot in simulation mode."""

    def test_first_check_is_fresh(self, cognitive_layer):
        """First simulated check should be fresh (ok)."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        analysis = bot.run("check", show_details=False)
        assert analysis.status == "ok"

    def test_second_check_still_fresh(self, cognitive_layer):
        """Second simulated check should still be fresh."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        bot.run("check", show_details=False)
        analysis = bot.run("check", show_details=False)
        assert analysis.status == "ok"

    def test_third_check_becomes_stale(self, cognitive_layer):
        """Third simulated check should be stale (fail)."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        bot.run("check", show_details=False)
        bot.run("check", show_details=False)
        analysis = bot.run("check", show_details=False)
        assert analysis.status == "fail"

    def test_regression_on_third_check(self, cognitive_layer):
        """Third check (ok → fail) should detect regression."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        bot.run("check", show_details=False)
        bot.run("check", show_details=False)
        analysis = bot.run("check", show_details=False)
        assert analysis.is_regression is True

    def test_streak_alert_after_consecutive_fails(self, cognitive_layer):
        """After 3+ consecutive fails → streak alert with recommendation."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        # Run all 5 simulated states
        results = []
        for _ in range(5):
            results.append(bot.run("check", show_details=False))
        # Last result (5th) should have fail_streak >= 3
        last = results[-1]
        assert last.fail_streak >= 3
        assert last.recommendation is not None
        assert "re-index" in last.recommendation

    def test_metrics_include_freshness_hours(self, cognitive_layer):
        """Bot output should include index_freshness_hours metric."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        output = bot.handle("check")
        assert output.metrics is not None
        assert "index_freshness_hours" in output.metrics

    def test_unknown_command(self, cognitive_layer):
        """Unknown command should return informative output."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        output = bot.handle("unknown_cmd")
        assert "Unknown command" in output.output


class TestDocsBotStats:
    """Tests for DocsBot stats command."""

    def test_stats_command(self, cognitive_layer):
        """stats command should return memory stats."""
        bot = DocsBot(cognitive=cognitive_layer, simulation_mode=True)
        bot.run("check", show_details=False)
        output = bot.handle("stats")
        assert "Total actions" in output.output


class TestDocsBotReal:
    """Tests for DocsBot in real mode."""

    def test_real_mode_no_docs_dir(self, cognitive_layer, tmp_path):
        """Real mode with nonexistent dir should report stale."""
        bot = DocsBot(
            cognitive=cognitive_layer,
            docs_dir=str(tmp_path / "nonexistent"),
            simulation_mode=False,
        )
        output = bot.handle("check")
        assert "not found" in output.output.lower() or "stale" in output.output.lower()

    def test_real_mode_empty_dir(self, cognitive_layer, tmp_path):
        """Real mode with empty dir should report stale."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        bot = DocsBot(
            cognitive=cognitive_layer,
            docs_dir=str(docs_dir),
            simulation_mode=False,
        )
        output = bot.handle("check")
        assert "no docs" in output.output.lower() or "stale" in output.output.lower()

    def test_real_mode_with_fresh_docs(self, cognitive_layer, tmp_path):
        """Real mode with recently modified docs should report fresh."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Fresh doc\nSome content.")
        bot = DocsBot(
            cognitive=cognitive_layer,
            docs_dir=str(docs_dir),
            simulation_mode=False,
        )
        output = bot.handle("check")
        assert "fresh" in output.output.lower()


class TestMain:
    """Tests for CLI main() function."""

    def test_main_no_args(self):
        """main() with no args should exit with code 1."""
        with patch("sys.argv", ["sample_bot"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_unknown_command(self):
        """main() with unknown command should exit with code 1."""
        with patch("sys.argv", ["sample_bot", "unknown"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_check_command(self):
        """main() with 'check' should run without error."""
        with patch("sys.argv", ["sample_bot", "check"]):
            # Should not raise
            main()

    def test_main_demo_command(self):
        """main() with 'demo' should run 5 cycles."""
        with patch("sys.argv", ["sample_bot", "demo"]):
            main()

    def test_main_stats_command(self):
        """main() with 'stats' should run without error."""
        with patch("sys.argv", ["sample_bot", "stats"]):
            main()
