"""Tests for spiral_04_cognitive.threshold_checker — L2 Thresholds."""

from spiral_04_cognitive.threshold_checker import ThresholdChecker, ThresholdResult, ThresholdSpec


class TestThresholdCheckerInit:
    """Tests for ThresholdChecker initialization."""

    def test_loads_thresholds_from_yaml(self, threshold_checker):
        """Should load thresholds from YAML file."""
        assert len(threshold_checker.thresholds) > 0

    def test_missing_yaml_returns_empty(self, tmp_path):
        """Missing YAML → empty thresholds."""
        checker = ThresholdChecker(yaml_path=str(tmp_path / "nonexistent.yaml"))
        assert checker.thresholds == {}

    def test_empty_yaml_returns_empty(self, tmp_path):
        """Empty YAML → empty thresholds."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        checker = ThresholdChecker(yaml_path=str(path))
        assert checker.thresholds == {}


class TestCheckMaxDirection:
    """Tests for max-direction thresholds (value must stay BELOW limit)."""

    def test_ok_below_warning(self, threshold_checker):
        """Value below warning → ok."""
        result = threshold_checker.check("disk_usage_pct", 50.0)
        assert result.level == "ok"

    def test_warning_at_boundary(self, threshold_checker):
        """Value at warning boundary → warning."""
        result = threshold_checker.check("disk_usage_pct", 80.0)
        assert result.level == "warning"

    def test_warning_between_limits(self, threshold_checker):
        """Value between warning and critical → warning."""
        result = threshold_checker.check("disk_usage_pct", 90.0)
        assert result.level == "warning"

    def test_critical_at_boundary(self, threshold_checker):
        """Value at critical boundary → critical."""
        result = threshold_checker.check("disk_usage_pct", 95.0)
        assert result.level == "critical"

    def test_critical_above(self, threshold_checker):
        """Value above critical → critical."""
        result = threshold_checker.check("disk_usage_pct", 99.0)
        assert result.level == "critical"


class TestCheckMinDirection:
    """Tests for min-direction thresholds (value must stay ABOVE limit)."""

    def test_ok_above_warning(self, threshold_checker):
        """Value above warning → ok."""
        result = threshold_checker.check("coverage_pct", 90.0)
        assert result.level == "ok"

    def test_warning_at_boundary(self, threshold_checker):
        """Value at warning boundary → warning."""
        result = threshold_checker.check("coverage_pct", 80.0)
        assert result.level == "warning"

    def test_warning_between_limits(self, threshold_checker):
        """Value between critical and warning → warning."""
        result = threshold_checker.check("coverage_pct", 70.0)
        assert result.level == "warning"

    def test_critical_at_boundary(self, threshold_checker):
        """Value at critical boundary → critical."""
        result = threshold_checker.check("coverage_pct", 60.0)
        assert result.level == "critical"

    def test_critical_below(self, threshold_checker):
        """Value below critical → critical."""
        result = threshold_checker.check("coverage_pct", 40.0)
        assert result.level == "critical"


class TestCheckUnknownMetric:
    """Tests for unknown metrics."""

    def test_unknown_metric_returns_ok(self, threshold_checker):
        """Unknown metric → ok with informative message."""
        result = threshold_checker.check("unknown_metric", 999.0)
        assert result.level == "ok"
        assert "No threshold configured" in result.message


class TestCheckAll:
    """Tests for check_all method."""

    def test_check_all_multiple_metrics(self, threshold_checker):
        """Should check all provided metrics."""
        results = threshold_checker.check_all(
            {
                "disk_usage_pct": 50.0,
                "coverage_pct": 90.0,
                "response_time_ms": 100.0,
            }
        )
        assert len(results) == 3
        assert all(r.level == "ok" for r in results)

    def test_check_all_mixed_results(self, threshold_checker):
        """Should return mixed levels when metrics vary."""
        results = threshold_checker.check_all(
            {
                "disk_usage_pct": 96.0,  # critical
                "coverage_pct": 90.0,  # ok
            }
        )
        levels = {r.metric_name: r.level for r in results}
        assert levels["disk_usage_pct"] == "critical"
        assert levels["coverage_pct"] == "ok"


class TestGetViolations:
    """Tests for get_violations method."""

    def test_no_violations(self, threshold_checker):
        """All ok → empty list."""
        violations = threshold_checker.get_violations(
            {
                "disk_usage_pct": 50.0,
                "coverage_pct": 90.0,
            }
        )
        assert violations == []

    def test_returns_only_violations(self, threshold_checker):
        """Should return only warning and critical results."""
        violations = threshold_checker.get_violations(
            {
                "disk_usage_pct": 96.0,  # critical
                "coverage_pct": 90.0,  # ok
                "response_time_ms": 3000.0,  # warning
            }
        )
        assert len(violations) == 2
        levels = {v.metric_name: v.level for v in violations}
        assert levels["disk_usage_pct"] == "critical"
        assert levels["response_time_ms"] == "warning"


class TestThresholdSpec:
    """Tests for ThresholdSpec dataclass."""

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        spec = ThresholdSpec(name="test", warning=80.0, critical=95.0, unit="%", direction="max")
        d = spec.to_dict()
        assert d["name"] == "test"
        assert d["warning"] == 80.0
        assert d["direction"] == "max"


class TestThresholdResult:
    """Tests for ThresholdResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        spec = ThresholdSpec(name="test", warning=80.0, critical=95.0, unit="%", direction="max")
        result = ThresholdResult(
            metric_name="test",
            value=90.0,
            level="warning",
            threshold=spec,
            message="test=90% >= warning(80%)",
        )
        d = result.to_dict()
        assert d["metric"] == "test"
        assert d["level"] == "warning"
        assert d["value"] == 90.0
