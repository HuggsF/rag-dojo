"""Tests for spiral_04_cognitive.cognitive_layer — L0+L1+L2 enrichment."""

from spiral_04_cognitive.cognitive_layer import EnrichedAnalysis


class TestEnrich:
    """Tests for CognitiveLayer.enrich."""

    def test_enrich_ok_output(self, cognitive_layer):
        """Clean output → status ok, no alerts."""
        analysis = cognitive_layer.enrich(
            bot_name="test_bot",
            command="check",
            raw_output="All systems healthy. Status: OK",
        )
        assert analysis.status == "ok"
        assert analysis.is_regression is False
        assert analysis.fail_streak == 0

    def test_enrich_fail_output(self, cognitive_layer):
        """Failure output → status fail, pattern match."""
        analysis = cognitive_layer.enrich(
            bot_name="test_bot",
            command="check",
            raw_output="Error: No space left on device",
        )
        assert analysis.status == "fail"
        assert len(analysis.pattern_matches) >= 1
        assert analysis.pattern_matches[0].pattern.name == "disk_full"

    def test_enrich_with_threshold_violation(self, cognitive_layer):
        """Metric exceeding threshold → threshold violation in analysis."""
        analysis = cognitive_layer.enrich(
            bot_name="test_bot",
            command="check",
            raw_output="System running",
            metrics={"disk_usage_pct": 96.0},
        )
        assert analysis.status == "fail"  # critical threshold → fail
        assert len(analysis.threshold_violations) == 1
        assert analysis.threshold_violations[0].level == "critical"

    def test_enrich_with_warning_threshold(self, cognitive_layer):
        """Metric at warning level → partial status."""
        analysis = cognitive_layer.enrich(
            bot_name="test_bot",
            command="check",
            raw_output="System running",
            metrics={"disk_usage_pct": 85.0},
        )
        assert analysis.status == "partial"
        assert len(analysis.threshold_violations) == 1

    def test_enrich_saves_to_memory(self, cognitive_layer):
        """Each enrich call should save to memory store."""
        cognitive_layer.enrich("test_bot", "check", "all good")
        stats = cognitive_layer.memory.stats(bot_name="test_bot")
        assert stats["total_actions"] == 1


class TestRegressionDetection:
    """Tests for regression detection in enrich."""

    def test_regression_detected(self, cognitive_layer):
        """ok → fail should flag regression."""
        cognitive_layer.enrich("test_bot", "check", "All tests pass. OK")
        analysis = cognitive_layer.enrich("test_bot", "check", "Error: FAILED assertion")
        assert analysis.is_regression is True
        assert any("REGRESSION" in a for a in analysis.alerts)

    def test_no_regression_on_first_run(self, cognitive_layer):
        """First run should never be regression."""
        analysis = cognitive_layer.enrich("test_bot", "check", "FAILED test")
        assert analysis.is_regression is False


class TestRecoveryDetection:
    """Tests for recovery detection in enrich."""

    def test_recovery_detected(self, cognitive_layer):
        """fail → ok should flag recovery."""
        cognitive_layer.enrich("test_bot", "check", "FAILED test")
        analysis = cognitive_layer.enrich("test_bot", "check", "All tests pass. OK")
        assert analysis.is_recovery is True
        assert any("RECOVERY" in a for a in analysis.alerts)


class TestStreakAlert:
    """Tests for fail streak alerts."""

    def test_streak_alert_after_threshold(self, cognitive_layer):
        """3+ consecutive fails should trigger streak alert."""
        for _ in range(3):
            cognitive_layer.enrich("test_bot", "check", "FAILED: stale docs")

        analysis = cognitive_layer.enrich("test_bot", "check", "FAILED: stale again")
        # After 4th fail, streak should be 4
        assert analysis.fail_streak >= 3
        assert any("STREAK ALERT" in a for a in analysis.alerts)
        assert analysis.recommendation is not None

    def test_no_streak_alert_below_threshold(self, cognitive_layer):
        """Less than 3 consecutive fails → no streak alert."""
        cognitive_layer.enrich("test_bot", "check", "FAILED test")
        analysis = cognitive_layer.enrich("test_bot", "check", "FAILED again")
        streak_alerts = [a for a in analysis.alerts if "STREAK ALERT" in a]
        assert len(streak_alerts) == 0


class TestRecommendation:
    """Tests for recommendation building."""

    def test_stale_docs_recommendation(self, cognitive_layer):
        """Stale docs pattern → re-index recommendation."""
        for _ in range(3):
            cognitive_layer.enrich("test_bot", "check", "Docs are stale and not fresh")
        analysis = cognitive_layer.enrich("test_bot", "check", "Docs stale not fresh")
        assert analysis.recommendation is not None
        assert "re-index" in analysis.recommendation

    def test_generic_recommendation(self, cognitive_layer):
        """Unknown failure → investigate-manually recommendation."""
        for _ in range(3):
            cognitive_layer.enrich("test_bot", "check", "FAILED for unknown reason")
        analysis = cognitive_layer.enrich("test_bot", "check", "FAILED again somehow")
        assert analysis.recommendation is not None
        assert "investigate" in analysis.recommendation


class TestDetermineStatus:
    """Tests for _determine_status."""

    def test_fail_wins_over_partial(self, cognitive_layer):
        """fail status should win over partial."""
        # Pattern says fail + threshold says warning(partial)
        analysis = cognitive_layer.enrich(
            "test_bot",
            "check",
            "FAILED test",
            metrics={"disk_usage_pct": 85.0},  # warning
        )
        assert analysis.status == "fail"

    def test_partial_when_only_warning(self, cognitive_layer):
        """Only warning threshold → partial."""
        analysis = cognitive_layer.enrich(
            "test_bot",
            "check",
            "System running normally",
            metrics={"disk_usage_pct": 85.0},
        )
        assert analysis.status == "partial"

    def test_ok_when_clean(self, cognitive_layer):
        """No issues → ok."""
        analysis = cognitive_layer.enrich(
            "test_bot",
            "check",
            "System running normally",
            metrics={"disk_usage_pct": 50.0},
        )
        assert analysis.status == "ok"


class TestEnrichedAnalysis:
    """Tests for EnrichedAnalysis dataclass."""

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        analysis = EnrichedAnalysis(
            raw_output="test",
            status="ok",
            is_regression=False,
            is_recovery=True,
            fail_streak=0,
            recommendation=None,
            alerts=["RECOVERY: test_bot recovered"],
        )
        d = analysis.to_dict()
        assert d["status"] == "ok"
        assert d["is_recovery"] is True
        assert len(d["alerts"]) == 1

    def test_to_dict_with_matches(self, pattern_detector):
        """to_dict should serialize pattern matches."""
        matches = pattern_detector.detect("Disk full error")
        analysis = EnrichedAnalysis(
            raw_output="test",
            status="fail",
            pattern_matches=matches,
        )
        d = analysis.to_dict()
        assert len(d["pattern_matches"]) >= 1
