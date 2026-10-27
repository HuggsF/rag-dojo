"""Tests for spiral_03_orchestration.tension — TensionMetrics and TensionHistory."""

import pytest

from spiral_03_orchestration.tension import TensionHistory, TensionMetrics


class TestTensionMetrics:
    """Tests for TensionMetrics dataclass."""

    def test_default_values(self):
        """Default metrics should be neutral."""
        m = TensionMetrics()
        assert m.creativity == pytest.approx(0.5)
        assert m.logic == pytest.approx(0.5)
        assert m.delta_estado == pytest.approx(0.0)

    def test_custom_values(self):
        """Custom values should be preserved."""
        m = TensionMetrics(creativity=0.7, logic=0.8, delta_estado=0.65)
        assert m.creativity == pytest.approx(0.7)
        assert m.logic == pytest.approx(0.8)
        assert m.delta_estado == pytest.approx(0.65)

    def test_clamp_above_one(self):
        """Values above 1.0 should be clamped to 1.0."""
        m = TensionMetrics(creativity=1.5, logic=2.0, delta_estado=3.0)
        assert m.creativity == pytest.approx(1.0)
        assert m.logic == pytest.approx(1.0)
        assert m.delta_estado == pytest.approx(1.0)

    def test_clamp_below_zero(self):
        """Values below 0.0 should be clamped to 0.0."""
        m = TensionMetrics(creativity=-0.5, logic=-1.0, delta_estado=-0.1)
        assert m.creativity == pytest.approx(0.0)
        assert m.logic == pytest.approx(0.0)
        assert m.delta_estado == pytest.approx(0.0)

    def test_tension_ratio_balanced(self):
        """Equal creativity and logic → ratio = 0.5."""
        m = TensionMetrics(creativity=0.5, logic=0.5)
        assert m.tension_ratio == pytest.approx(0.5)

    def test_tension_ratio_pure_creativity(self):
        """Pure creativity → ratio = 1.0."""
        m = TensionMetrics(creativity=1.0, logic=0.0)
        assert m.tension_ratio == pytest.approx(1.0)

    def test_tension_ratio_pure_logic(self):
        """Pure logic → ratio = 0.0."""
        m = TensionMetrics(creativity=0.0, logic=1.0)
        assert m.tension_ratio == pytest.approx(0.0)

    def test_tension_ratio_both_zero(self):
        """Both zero → ratio defaults to 0.5."""
        m = TensionMetrics(creativity=0.0, logic=0.0)
        assert m.tension_ratio == pytest.approx(0.5)

    def test_is_zero_delta_true(self):
        """Delta below threshold → is_zero_delta = True."""
        m = TensionMetrics(delta_estado=0.01)
        assert m.is_zero_delta is True

    def test_is_zero_delta_false(self):
        """Delta above threshold → is_zero_delta = False."""
        m = TensionMetrics(delta_estado=0.5)
        assert m.is_zero_delta is False

    def test_is_zero_delta_at_threshold(self):
        """Delta exactly at threshold → is_zero_delta = False (< not <=)."""
        m = TensionMetrics(delta_estado=0.05)
        assert m.is_zero_delta is False

    def test_to_dict(self):
        """to_dict should contain all expected keys."""
        m = TensionMetrics(creativity=0.7, logic=0.8, delta_estado=0.65)
        d = m.to_dict()
        assert "creativity" in d
        assert "logic" in d
        assert "delta_estado" in d
        assert "tension_ratio" in d
        assert "is_zero_delta" in d
        assert d["creativity"] == pytest.approx(0.7)
        assert d["logic"] == pytest.approx(0.8)
        assert d["delta_estado"] == pytest.approx(0.65)

    def test_to_dict_values_rounded(self):
        """to_dict values should be rounded to 4 decimals."""
        m = TensionMetrics(creativity=0.33333333, logic=0.66666666, delta_estado=0.12345678)
        d = m.to_dict()
        assert d["creativity"] == pytest.approx(0.3333)
        assert d["logic"] == pytest.approx(0.6667)
        assert d["delta_estado"] == pytest.approx(0.1235)


class TestTensionHistory:
    """Tests for TensionHistory streak tracking."""

    def test_empty_history(self, empty_history: TensionHistory):
        """Empty history should have zero streak."""
        assert empty_history.zero_delta_streak == 0
        assert empty_history.should_swap_strategy is False

    def test_record_adds_entry(self, empty_history: TensionHistory):
        """Recording should add to history."""
        m = TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.7)
        empty_history.record(m)
        assert len(empty_history.records) == 1

    def test_zero_delta_streak_count(self, history_with_streak: TensionHistory):
        """Should count 3 consecutive zero-delta entries from the end."""
        assert history_with_streak.zero_delta_streak == 3

    def test_should_swap_strategy_on_streak(self, history_with_streak: TensionHistory):
        """Streak of 3 should trigger swap (default limit is 3)."""
        assert history_with_streak.should_swap_strategy is True

    def test_streak_broken_by_good_delta(self, history_with_streak: TensionHistory):
        """Adding a good delta should break the streak."""
        history_with_streak.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.8))
        assert history_with_streak.zero_delta_streak == 0
        assert history_with_streak.should_swap_strategy is False

    def test_avg_creativity(self, empty_history: TensionHistory):
        """Average creativity over last N entries."""
        empty_history.record(TensionMetrics(creativity=0.4))
        empty_history.record(TensionMetrics(creativity=0.6))
        assert empty_history.avg_creativity(last_n=2) == pytest.approx(0.5)

    def test_avg_creativity_empty(self, empty_history: TensionHistory):
        """Empty history defaults to 0.5."""
        assert empty_history.avg_creativity() == pytest.approx(0.5)

    def test_avg_logic(self, empty_history: TensionHistory):
        """Average logic over last N entries."""
        empty_history.record(TensionMetrics(logic=0.3))
        empty_history.record(TensionMetrics(logic=0.7))
        assert empty_history.avg_logic(last_n=2) == pytest.approx(0.5)

    def test_avg_logic_empty(self, empty_history: TensionHistory):
        """Empty history defaults to 0.5."""
        assert empty_history.avg_logic() == pytest.approx(0.5)

    def test_clear(self, history_with_streak: TensionHistory):
        """Clear should reset history."""
        history_with_streak.clear()
        assert len(history_with_streak.records) == 0
        assert history_with_streak.zero_delta_streak == 0
