"""Shared fixtures for Spiral 05 tests."""

import pytest

from spiral_05_economics.budget_tracker import BudgetTracker
from spiral_05_economics.circuit_breaker import CircuitBreaker
from spiral_05_economics.cost_calculator import CostCalculator
from spiral_05_economics.layer_pipeline import LayerPipeline
from spiral_05_economics.model_router import ModelRouter
from spiral_05_economics.token_counter import TokenCounter


@pytest.fixture
def token_counter():
    """TokenCounter instance."""
    return TokenCounter()


@pytest.fixture
def cost_calculator():
    """CostCalculator with default pricing."""
    return CostCalculator()


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database path."""
    return str(tmp_path / "test_budget.db")


@pytest.fixture
def budget_tracker(tmp_db):
    """BudgetTracker with temporary database."""
    tracker = BudgetTracker(db_path=tmp_db)
    yield tracker
    tracker.close()


@pytest.fixture
def model_router(cost_calculator):
    """ModelRouter with default config."""
    return ModelRouter(calculator=cost_calculator)


@pytest.fixture
def circuit_breaker(budget_tracker):
    """CircuitBreaker with test budget tracker."""
    return CircuitBreaker(tracker=budget_tracker)


@pytest.fixture
def layer_pipeline(budget_tracker, cost_calculator, model_router, circuit_breaker):
    """LayerPipeline with all test dependencies."""
    return LayerPipeline(
        tracker=budget_tracker,
        calculator=cost_calculator,
        router=model_router,
        breaker=circuit_breaker,
    )
