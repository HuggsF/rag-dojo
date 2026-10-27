"""Tests for spiral_05_economics.cost_calculator — pricing and cost computation."""

import pytest

from spiral_05_economics.cost_calculator import CostCalculator, CostRecord
from spiral_05_economics.token_counter import TokenCount


class TestCostRecord:
    """Tests for CostRecord dataclass."""

    def test_has_required_fields(self):
        rec = CostRecord(
            model="llama-3.3-70b-versatile",
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
        )
        assert rec.model == "llama-3.3-70b-versatile"
        assert rec.total_cost == 0.0

    def test_repr(self):
        rec = CostRecord(
            model="claude-sonnet",
            input_tokens=1000,
            output_tokens=200,
            input_cost=0.003,
            output_cost=0.003,
            total_cost=0.006,
        )
        assert "$" in repr(rec)


class TestCostCalculator:
    """Tests for CostCalculator."""

    def test_free_model_costs_zero(self, cost_calculator):
        tc = TokenCount(input_tokens=1000, output_tokens=500, model="llama-3.3-70b-versatile")
        rec = cost_calculator.calculate(tc)
        assert rec.total_cost == 0.0

    def test_paid_model_costs_positive(self, cost_calculator):
        tc = TokenCount(input_tokens=1000, output_tokens=500, model="claude-sonnet")
        rec = cost_calculator.calculate(tc)
        assert rec.total_cost > 0.0

    def test_input_output_cost_breakdown(self, cost_calculator):
        tc = TokenCount(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet")
        rec = cost_calculator.calculate(tc)
        assert rec.input_cost == pytest.approx(3.00, rel=1e-2)
        assert rec.output_cost == pytest.approx(15.00, rel=1e-2)
        assert rec.total_cost == pytest.approx(18.00, rel=1e-2)

    def test_zero_tokens_zero_cost(self, cost_calculator):
        tc = TokenCount(input_tokens=0, output_tokens=0, model="claude-sonnet")
        rec = cost_calculator.calculate(tc)
        assert rec.total_cost == 0.0

    def test_unknown_model_raises(self, cost_calculator):
        tc = TokenCount(input_tokens=100, output_tokens=50, model="nonexistent-model")
        with pytest.raises(ValueError, match="Unknown model"):
            cost_calculator.calculate(tc)

    def test_get_pricing_returns_tuple(self, cost_calculator):
        price = cost_calculator.get_pricing("claude-haiku")
        assert isinstance(price, tuple)
        assert len(price) == 2
        assert price[0] == 0.25
        assert price[1] == 1.25

    def test_list_models(self, cost_calculator):
        models = cost_calculator.list_models()
        assert "llama-3.3-70b-versatile" in models
        assert "claude-sonnet" in models

    def test_custom_pricing(self):
        custom = {"my-model": (1.00, 2.00)}
        calc = CostCalculator(pricing=custom)
        tc = TokenCount(input_tokens=1_000_000, output_tokens=1_000_000, model="my-model")
        rec = calc.calculate(tc)
        assert rec.input_cost == pytest.approx(1.00)
        assert rec.output_cost == pytest.approx(2.00)
