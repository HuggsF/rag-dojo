"""Shared fixtures for Spiral 03 tests."""

import pytest

from spiral_03_orchestration.tension import TensionHistory, TensionMetrics
from spiral_03_orchestration.tension_profile import ProfileType, TensionProfile, get_profile


@pytest.fixture
def standard_profile() -> TensionProfile:
    """Standard tension profile."""
    return get_profile(ProfileType.STANDARD)


@pytest.fixture
def exploratory_profile() -> TensionProfile:
    """Exploratory tension profile."""
    return get_profile(ProfileType.EXPLORATORY)


@pytest.fixture
def critical_profile() -> TensionProfile:
    """Critical tension profile."""
    return get_profile(ProfileType.CRITICAL)


@pytest.fixture
def destructive_profile() -> TensionProfile:
    """Destructive tension profile."""
    return get_profile(ProfileType.DESTRUCTIVE)


@pytest.fixture
def balanced_metrics() -> TensionMetrics:
    """Balanced tension metrics (creativity = logic, good delta)."""
    return TensionMetrics(creativity=0.7, logic=0.8, delta_estado=0.65)


@pytest.fixture
def zero_delta_metrics() -> TensionMetrics:
    """Zero-delta metrics (state didn't change)."""
    return TensionMetrics(creativity=0.7, logic=0.8, delta_estado=0.0)


@pytest.fixture
def high_creativity_metrics() -> TensionMetrics:
    """High creativity, low logic metrics."""
    return TensionMetrics(creativity=0.9, logic=0.3, delta_estado=0.6)


@pytest.fixture
def empty_history() -> TensionHistory:
    """Empty tension history."""
    return TensionHistory()


@pytest.fixture
def history_with_streak() -> TensionHistory:
    """History with a zero-delta streak of 3."""
    history = TensionHistory()
    history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.7))
    history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.02))
    history.record(TensionMetrics(creativity=0.6, logic=0.4, delta_estado=0.01))
    history.record(TensionMetrics(creativity=0.4, logic=0.6, delta_estado=0.03))
    return history
