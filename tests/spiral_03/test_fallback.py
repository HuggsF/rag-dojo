"""Tests for spiral_03_orchestration.fallback — 7-strategy cascade."""

from spiral_03_orchestration.fallback import (
    FALLBACK_CASCADE,
    FallbackAction,
    FallbackCascade,
    FallbackStrategy,
    get_fallback_action,
)


class TestFallbackStrategy:
    """Tests for FallbackStrategy enum."""

    def test_all_seven_strategies(self):
        """All 7 strategies should be defined."""
        assert len(FallbackStrategy) == 7

    def test_cascade_order(self):
        """Cascade should be in the correct order."""
        expected = [
            FallbackStrategy.BROADEN_QUERY,
            FallbackStrategy.DEEPEN_GRAPH,
            FallbackStrategy.BOOST_CREATIVITY,
            FallbackStrategy.BOOST_LOGIC,
            FallbackStrategy.SWAP_RETRIEVER,
            FallbackStrategy.RELAX_THRESHOLD,
            FallbackStrategy.REPORT_FAILURE,
        ]
        assert FALLBACK_CASCADE == expected

    def test_report_failure_is_last(self):
        """REPORT_FAILURE should always be the last strategy."""
        assert FALLBACK_CASCADE[-1] == FallbackStrategy.REPORT_FAILURE


class TestFallbackAction:
    """Tests for FallbackAction dataclass."""

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        action = FallbackAction(
            strategy=FallbackStrategy.BROADEN_QUERY,
            description="test",
            adjustments={"key": "value"},
        )
        d = action.to_dict()
        assert d["strategy"] == "BROADEN_QUERY"
        assert d["description"] == "test"
        assert d["adjustments"] == {"key": "value"}


class TestGetFallbackAction:
    """Tests for get_fallback_action function."""

    def test_every_strategy_has_action(self):
        """Every strategy should have a corresponding action."""
        for strategy in FallbackStrategy:
            action = get_fallback_action(strategy)
            assert action.strategy == strategy
            assert action.description
            assert isinstance(action.adjustments, dict)

    def test_broaden_query_adjustments(self):
        """BROADEN_QUERY should increase top_k and reduce threshold."""
        action = get_fallback_action(FallbackStrategy.BROADEN_QUERY)
        assert "top_k_multiplier" in action.adjustments
        assert action.adjustments["top_k_multiplier"] > 1.0

    def test_deepen_graph_adjustments(self):
        """DEEPEN_GRAPH should increase depth."""
        action = get_fallback_action(FallbackStrategy.DEEPEN_GRAPH)
        assert "graph_depth_increase" in action.adjustments
        assert action.adjustments["graph_depth_increase"] > 0

    def test_report_failure_aborts(self):
        """REPORT_FAILURE should set abort=True."""
        action = get_fallback_action(FallbackStrategy.REPORT_FAILURE)
        assert action.adjustments.get("abort") is True


class TestFallbackCascade:
    """Tests for FallbackCascade manager."""

    def test_initial_state(self):
        """Fresh cascade should not be exhausted."""
        cascade = FallbackCascade()
        assert cascade.is_exhausted is False
        assert cascade.current_strategy == FallbackStrategy.BROADEN_QUERY
        assert cascade.attempted == []

    def test_next_returns_actions_in_order(self):
        """next() should return strategies in cascade order."""
        cascade = FallbackCascade()
        strategies = []
        while not cascade.is_exhausted:
            action = cascade.next()
            assert action is not None
            strategies.append(action.strategy)
        assert strategies == FALLBACK_CASCADE

    def test_exhausted_after_all_strategies(self):
        """Should be exhausted after iterating through all 7."""
        cascade = FallbackCascade()
        for _ in range(7):
            cascade.next()
        assert cascade.is_exhausted is True
        assert cascade.next() is None

    def test_attempted_tracks_history(self):
        """attempted should track which strategies were tried."""
        cascade = FallbackCascade()
        cascade.next()
        cascade.next()
        assert len(cascade.attempted) == 2
        assert cascade.attempted[0] == FallbackStrategy.BROADEN_QUERY
        assert cascade.attempted[1] == FallbackStrategy.DEEPEN_GRAPH

    def test_reset_clears_state(self):
        """reset() should restore cascade to initial state."""
        cascade = FallbackCascade()
        cascade.next()
        cascade.next()
        cascade.reset()
        assert cascade.is_exhausted is False
        assert cascade.current_strategy == FallbackStrategy.BROADEN_QUERY
        assert cascade.attempted == []

    def test_current_strategy_none_when_exhausted(self):
        """current_strategy should be None when exhausted."""
        cascade = FallbackCascade()
        for _ in range(7):
            cascade.next()
        assert cascade.current_strategy is None
