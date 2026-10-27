"""Tests for spiral_04_cognitive.pattern_detector — L1 Patterns."""

from spiral_04_cognitive.pattern_detector import Pattern, PatternDetector, PatternMatch


class TestPatternDetectorInit:
    """Tests for PatternDetector initialization."""

    def test_loads_patterns_from_yaml(self, pattern_detector):
        """Should load patterns from YAML file."""
        assert len(pattern_detector.patterns) > 0

    def test_missing_yaml_returns_empty(self, tmp_path):
        """Missing YAML file should return empty patterns."""
        detector = PatternDetector(yaml_path=str(tmp_path / "nonexistent.yaml"))
        assert detector.patterns == []

    def test_empty_yaml_returns_empty(self, tmp_path):
        """Empty YAML file should return empty patterns."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        detector = PatternDetector(yaml_path=str(path))
        assert detector.patterns == []

    def test_yaml_without_patterns_key(self, tmp_path):
        """YAML without 'patterns' key should return empty."""
        path = tmp_path / "bad.yaml"
        path.write_text("other_key: value\n")
        detector = PatternDetector(yaml_path=str(path))
        assert detector.patterns == []


class TestPatternDetect:
    """Tests for detect method."""

    def test_detect_disk_full(self, pattern_detector):
        """Should detect 'disk full' pattern."""
        matches = pattern_detector.detect("Error: No space left on device")
        assert len(matches) >= 1
        names = [m.pattern.name for m in matches]
        assert "disk_full" in names

    def test_detect_test_failure(self, pattern_detector):
        """Should detect test failure pattern."""
        matches = pattern_detector.detect("3 tests FAILED, 10 passed")
        assert len(matches) >= 1
        names = [m.pattern.name for m in matches]
        assert "test_failure" in names

    def test_detect_stale_docs(self, pattern_detector):
        """Should detect stale docs pattern."""
        matches = pattern_detector.detect("Docs are stale and outdated")
        assert len(matches) >= 1
        names = [m.pattern.name for m in matches]
        assert "stale_docs" in names

    def test_detect_multiple_patterns(self, pattern_detector):
        """Should detect multiple patterns in same output."""
        output = "FAILED: disk full, deprecated API used"
        matches = pattern_detector.detect(output)
        names = {m.pattern.name for m in matches}
        assert len(names) >= 2

    def test_detect_no_match(self, pattern_detector):
        """Clean output should return no matches."""
        matches = pattern_detector.detect("Everything is running smoothly")
        # "success" pattern might match, but no failure patterns
        fail_matches = [m for m in matches if m.pattern.status == "fail"]
        assert len(fail_matches) == 0

    def test_detect_case_insensitive(self, pattern_detector):
        """Patterns should be case-insensitive."""
        matches = pattern_detector.detect("DISK FULL error occurred")
        names = [m.pattern.name for m in matches]
        assert "disk_full" in names


class TestPatternClassify:
    """Tests for classify method."""

    def test_classify_fail(self, pattern_detector):
        """Critical pattern → fail status."""
        status = pattern_detector.classify("No space left on device")
        assert status == "fail"

    def test_classify_partial(self, pattern_detector):
        """Low severity pattern only → partial status."""
        status = pattern_detector.classify("This API is deprecated")
        assert status == "partial"

    def test_classify_ok(self, pattern_detector):
        """No matching patterns → ok status."""
        status = pattern_detector.classify("just some random text with no issues")
        assert status == "ok"

    def test_classify_highest_severity_wins(self, pattern_detector):
        """When multiple patterns match, highest severity determines status."""
        # "FAILED" (high) + "deprecated" (low) → fail wins
        status = pattern_detector.classify("FAILED test using deprecated API")
        assert status == "fail"

    def test_classify_ok_output(self, pattern_detector):
        """Output matching success pattern → ok."""
        status = pattern_detector.classify("All tests pass. Status: OK")
        assert status == "ok"


class TestGetHighestSeverity:
    """Tests for get_highest_severity."""

    def test_empty_matches(self, pattern_detector):
        """Empty list → None."""
        assert pattern_detector.get_highest_severity([]) is None

    def test_single_match(self, pattern_detector):
        """Single match → its severity."""
        matches = pattern_detector.detect("No space left on device")
        severity = pattern_detector.get_highest_severity(matches)
        assert severity == "critical"


class TestPattern:
    """Tests for Pattern dataclass."""

    def test_compiled_regex(self):
        """compiled property should return a usable regex."""
        pattern = Pattern(name="test", regex="(?i)hello", severity="low", status="ok")
        assert pattern.compiled.search("Hello World") is not None

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        pattern = Pattern(name="test", regex="hello", severity="high", status="fail")
        d = pattern.to_dict()
        assert d["name"] == "test"
        assert d["severity"] == "high"


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_to_dict(self):
        """to_dict should serialize and truncate matched_text."""
        pattern = Pattern(name="test", regex="hello", severity="high", status="fail")
        match = PatternMatch(pattern=pattern, matched_text="hello world" * 20, position=0)
        d = match.to_dict()
        assert len(d["matched_text"]) <= 100
        assert d["pattern"] == "test"
        assert d["position"] == 0
