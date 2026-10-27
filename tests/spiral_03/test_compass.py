"""Tests for spiral_03_orchestration.compass — MendaCompass pre-action reasoning."""

from spiral_03_orchestration.compass import Action, MendaCompass
from spiral_03_orchestration.tension import TensionMetrics
from spiral_03_orchestration.tension_profile import TensionProfile


class TestAction:
    """Tests for Action enum."""

    def test_action_values(self):
        """All 3 actions should exist."""
        assert Action.RETRIEVE.value == "RETRIEVE"
        assert Action.FALLBACK.value == "FALLBACK"
        assert Action.SKIP.value == "SKIP"


class TestCompassReading:
    """Tests for CompassReading dataclass."""

    def test_to_dict(self, balanced_metrics, standard_profile):
        """to_dict should serialize all fields."""
        compass = MendaCompass()
        reading = compass.evaluate(balanced_metrics, standard_profile)
        d = reading.to_dict()
        assert "action" in d
        assert "confidence" in d
        assert "metrics" in d
        assert "profile" in d
        assert "reason" in d


class TestMendaCompass:
    """Tests for MendaCompass evaluation logic."""

    def test_zero_delta_triggers_fallback(
        self,
        zero_delta_metrics: TensionMetrics,
        standard_profile: TensionProfile,
    ):
        """Zero delta should always result in FALLBACK."""
        compass = MendaCompass()
        reading = compass.evaluate(zero_delta_metrics, standard_profile)
        assert reading.action == Action.FALLBACK
        assert "Zero-delta collapse" in reading.reason

    def test_high_confidence_triggers_retrieve(self, exploratory_profile: TensionProfile):
        """High confidence should result in RETRIEVE with permissive profile."""
        metrics = TensionMetrics(creativity=0.99, logic=0.99, delta_estado=0.99)
        compass = MendaCompass()
        reading = compass.evaluate(metrics, exploratory_profile)
        # exploratory: 0.99*0.8 * 0.99*0.2 * 0.99 = 0.792 * 0.198 * 0.99 = 0.1552
        # threshold = 0.15, so 0.1552 >= 0.15 → RETRIEVE
        assert reading.action == Action.RETRIEVE
        assert "Proceeding with retrieval" in reading.reason

    def test_low_confidence_triggers_fallback(self, destructive_profile: TensionProfile):
        """Low confidence with restrictive profile should FALLBACK."""
        metrics = TensionMetrics(creativity=0.3, logic=0.3, delta_estado=0.3)
        compass = MendaCompass()
        reading = compass.evaluate(metrics, destructive_profile)
        assert reading.action == Action.FALLBACK
        assert "Falling back" in reading.reason

    def test_exploratory_more_permissive(
        self,
        exploratory_profile: TensionProfile,
        destructive_profile: TensionProfile,
    ):
        """Exploratory profile should have a lower threshold than destructive."""
        assert exploratory_profile.confidence_threshold < destructive_profile.confidence_threshold

    def test_reading_preserves_metrics(self, balanced_metrics, standard_profile):
        """Reading should contain the original metrics."""
        compass = MendaCompass()
        reading = compass.evaluate(balanced_metrics, standard_profile)
        assert reading.metrics is balanced_metrics
        assert reading.profile is standard_profile

    def test_reading_has_confidence_result(self, balanced_metrics, standard_profile):
        """Reading should contain a ConfidenceResult."""
        compass = MendaCompass()
        reading = compass.evaluate(balanced_metrics, standard_profile)
        assert reading.confidence is not None
        assert hasattr(reading.confidence, "score")
        assert hasattr(reading.confidence, "passed")
