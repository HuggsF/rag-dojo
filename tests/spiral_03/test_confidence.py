"""Tests for spiral_03_orchestration.confidence — the confidence formula."""

import pytest

from spiral_03_orchestration.confidence import ConfidenceResult, calculate_confidence
from spiral_03_orchestration.tension import TensionMetrics
from spiral_03_orchestration.tension_profile import TensionProfile, get_profile


class TestConfidenceResult:
    """Tests for ConfidenceResult dataclass."""

    def test_to_dict(self):
        """to_dict should contain all fields."""
        r = ConfidenceResult(
            score=0.5,
            weighted_creativity=0.3,
            weighted_logic=0.4,
            delta_estado=0.7,
            threshold=0.3,
            passed=True,
            zero_delta_collapse=False,
        )
        d = r.to_dict()
        assert d["score"] == pytest.approx(0.5)
        assert d["passed"] is True
        assert d["zero_delta_collapse"] is False


class TestCalculateConfidence:
    """Tests for the calculate_confidence function."""

    def test_balanced_standard_passes(
        self,
        balanced_metrics: TensionMetrics,
        standard_profile: TensionProfile,
    ):
        """Balanced metrics with standard profile should pass."""
        result = calculate_confidence(balanced_metrics, standard_profile)
        expected = 0.35 * 0.40 * 0.65
        assert result.score == pytest.approx(expected, rel=1e-3)
        assert result.zero_delta_collapse is False

    def test_zero_delta_always_collapses(
        self,
        zero_delta_metrics: TensionMetrics,
        standard_profile: TensionProfile,
    ):
        """Zero delta should collapse confidence to 0 regardless of other values."""
        result = calculate_confidence(zero_delta_metrics, standard_profile)
        assert result.score == pytest.approx(0.0)
        assert result.zero_delta_collapse is True
        assert result.passed is False

    def test_zero_delta_collapses_even_with_exploratory(
        self,
        zero_delta_metrics: TensionMetrics,
        exploratory_profile: TensionProfile,
    ):
        """Zero delta collapses even with the most permissive profile."""
        result = calculate_confidence(zero_delta_metrics, exploratory_profile)
        assert result.score == pytest.approx(0.0)
        assert result.zero_delta_collapse is True

    def test_high_values_with_critical(self, critical_profile: TensionProfile):
        """High tension values with critical profile."""
        metrics = TensionMetrics(creativity=0.9, logic=0.9, delta_estado=0.9)
        result = calculate_confidence(metrics, critical_profile)
        expected = 0.18 * 0.72 * 0.9
        assert result.score == pytest.approx(expected, rel=1e-3)

    def test_destructive_requires_very_high_confidence(self, destructive_profile: TensionProfile):
        """Even good metrics might not pass destructive threshold (0.8)."""
        metrics = TensionMetrics(creativity=0.7, logic=0.8, delta_estado=0.65)
        result = calculate_confidence(metrics, destructive_profile)
        assert result.passed is False
        assert result.score < destructive_profile.confidence_threshold

    def test_exploratory_passes_easily(self, exploratory_profile: TensionProfile):
        """Exploratory profile with moderate values should not collapse."""
        metrics = TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.5)
        result = calculate_confidence(metrics, exploratory_profile)
        assert result.zero_delta_collapse is False

    def test_confidence_weighted_by_profile(self):
        """Same metrics should produce different scores with different profiles."""
        metrics = TensionMetrics(creativity=0.7, logic=0.8, delta_estado=0.65)
        standard = calculate_confidence(metrics, get_profile("STANDARD"))
        critical = calculate_confidence(metrics, get_profile("CRITICAL"))
        destructive = calculate_confidence(metrics, get_profile("DESTRUCTIVE"))
        scores = {standard.score, critical.score, destructive.score}
        assert len(scores) == 3

    def test_result_contains_threshold(self, standard_profile: TensionProfile):
        """Result should carry the threshold from the profile."""
        metrics = TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.5)
        result = calculate_confidence(metrics, standard_profile)
        assert result.threshold == pytest.approx(standard_profile.confidence_threshold)

    def test_passed_flag_matches_comparison(self, standard_profile: TensionProfile):
        """passed should be True iff score >= threshold."""
        metrics = TensionMetrics(creativity=0.99, logic=0.99, delta_estado=0.99)
        result = calculate_confidence(metrics, standard_profile)
        assert result.passed == (result.score >= result.threshold)
