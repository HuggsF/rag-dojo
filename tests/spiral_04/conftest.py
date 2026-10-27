"""Shared fixtures for Spiral 04 tests."""

import pytest
import yaml

from spiral_04_cognitive.cognitive_layer import CognitiveLayer
from spiral_04_cognitive.memory_store import MemoryStore
from spiral_04_cognitive.pattern_detector import PatternDetector
from spiral_04_cognitive.threshold_checker import ThresholdChecker


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def memory_store(tmp_db):
    """MemoryStore with temporary database."""
    store = MemoryStore(db_path=tmp_db)
    yield store
    store.close()


@pytest.fixture
def patterns_yaml(tmp_path):
    """Temporary patterns.yaml for testing."""
    data = {
        "patterns": [
            {
                "name": "disk_full",
                "regex": "(?i)(no space left on device|disk full)",
                "severity": "critical",
                "status": "fail",
            },
            {
                "name": "test_failure",
                "regex": "(?i)(FAILED|AssertionError)",
                "severity": "high",
                "status": "fail",
            },
            {
                "name": "stale_docs",
                "regex": "(?i)(stale|outdated|not.*fresh)",
                "severity": "medium",
                "status": "fail",
            },
            {
                "name": "deprecation",
                "regex": "(?i)(deprecated)",
                "severity": "low",
                "status": "partial",
            },
            {
                "name": "success",
                "regex": "(?i)(all.*pass|success|OK\\b|healthy)",
                "severity": "low",
                "status": "ok",
            },
        ]
    }
    path = tmp_path / "patterns.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


@pytest.fixture
def thresholds_yaml(tmp_path):
    """Temporary thresholds.yaml for testing."""
    data = {
        "thresholds": {
            "disk_usage_pct": {
                "warning": 80,
                "critical": 95,
                "unit": "%",
                "direction": "max",
            },
            "coverage_pct": {
                "warning": 80,
                "critical": 60,
                "unit": "%",
                "direction": "min",
            },
            "response_time_ms": {
                "warning": 2000,
                "critical": 5000,
                "unit": "ms",
                "direction": "max",
            },
        }
    }
    path = tmp_path / "thresholds.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


@pytest.fixture
def pattern_detector(patterns_yaml):
    """PatternDetector with test patterns."""
    return PatternDetector(yaml_path=patterns_yaml)


@pytest.fixture
def threshold_checker(thresholds_yaml):
    """ThresholdChecker with test thresholds."""
    return ThresholdChecker(yaml_path=thresholds_yaml)


@pytest.fixture
def cognitive_layer(memory_store, pattern_detector, threshold_checker):
    """CognitiveLayer with all test dependencies."""
    return CognitiveLayer(
        memory=memory_store,
        detector=pattern_detector,
        checker=threshold_checker,
    )
