"""Tests for spiral_04_cognitive.bot_base — Bot template."""

import pytest

from spiral_04_cognitive.bot_base import BotBase, BotOutput
from spiral_04_cognitive.cognitive_layer import EnrichedAnalysis


class ConcreteBot(BotBase):
    """Concrete implementation for testing."""

    def __init__(self, name="test_bot", cognitive=None, output="all good"):
        super().__init__(name=name, cognitive=cognitive)
        self._output = output

    def handle(self, command: str) -> BotOutput:
        return BotOutput(output=self._output)


class TestBotBase:
    """Tests for BotBase abstract class."""

    def test_cannot_instantiate_abstract(self):
        """BotBase should not be instantiable directly."""
        with pytest.raises(TypeError):
            BotBase(name="test")

    def test_concrete_bot_has_name(self, cognitive_layer):
        """Concrete bot should have its name."""
        bot = ConcreteBot(name="my_bot", cognitive=cognitive_layer)
        assert bot.name == "my_bot"

    def test_lazy_cognitive_layer(self):
        """Cognitive layer should be lazy-loaded."""
        bot = ConcreteBot()
        # Before access, _cognitive is None
        assert bot._cognitive is None

    def test_handle_returns_bot_output(self, cognitive_layer):
        """handle() should return BotOutput."""
        bot = ConcreteBot(cognitive=cognitive_layer)
        result = bot.handle("check")
        assert isinstance(result, BotOutput)
        assert result.output == "all good"


class TestBotRun:
    """Tests for BotBase.run method."""

    def test_run_returns_enriched_analysis(self, cognitive_layer):
        """run() should return EnrichedAnalysis."""
        bot = ConcreteBot(cognitive=cognitive_layer)
        analysis = bot.run("check", show_details=False)
        assert isinstance(analysis, EnrichedAnalysis)
        assert analysis.status == "ok"

    def test_run_saves_to_memory(self, cognitive_layer):
        """run() should save action to memory via cognitive layer."""
        bot = ConcreteBot(cognitive=cognitive_layer)
        bot.run("check", show_details=False)
        stats = cognitive_layer.memory.stats(bot_name="test_bot")
        assert stats["total_actions"] == 1

    def test_run_with_failing_output(self, cognitive_layer):
        """run() with failure output should detect patterns."""
        bot = ConcreteBot(cognitive=cognitive_layer, output="Error: FAILED assertion in test_main")
        analysis = bot.run("check", show_details=False)
        assert analysis.status == "fail"
        assert len(analysis.pattern_matches) >= 1

    def test_run_with_show_details(self, cognitive_layer):
        """run() with show_details=True should not raise."""
        bot = ConcreteBot(cognitive=cognitive_layer)
        # Should not raise even with show_details=True
        analysis = bot.run("check", show_details=True)
        assert analysis.status == "ok"

    def test_run_display_with_alerts(self, cognitive_layer):
        """Display should handle alerts without raising."""
        bot = ConcreteBot(cognitive=cognitive_layer, output="FAILED test")
        # Run multiple times to build streak
        for _ in range(3):
            bot.run("check", show_details=False)
        # This one should have alerts and show them
        analysis = bot.run("check", show_details=True)
        assert analysis.fail_streak >= 3


class TestBotOutput:
    """Tests for BotOutput dataclass."""

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        output = BotOutput(output="test output", metrics={"cpu": 50.0})
        d = output.to_dict()
        assert d["metrics"] == {"cpu": 50.0}

    def test_to_dict_truncates_output(self):
        """to_dict should truncate long output."""
        output = BotOutput(output="x" * 500)
        d = output.to_dict()
        assert len(d["output"]) == 200

    def test_to_dict_none_metrics(self):
        """to_dict with None metrics."""
        output = BotOutput(output="test")
        d = output.to_dict()
        assert d["metrics"] is None
