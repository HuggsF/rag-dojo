"""Tests for spiral_06_state_of_art.scorer."""

import pytest

from spiral_06_state_of_art.scorer import DimensionScore, Scorer


class TestDimensionScore:
    def test_weighted(self):
        d = DimensionScore(name="precision", value=80.0, weight=0.35)
        assert d.weighted() == pytest.approx(28.0)

    def test_repr(self):
        d = DimensionScore(name="cost", value=90.0, weight=0.2)
        assert "cost" in repr(d)


class TestScoreCard:
    def test_get_dimension(self, good_scorecard):
        dim = good_scorecard.get_dimension("precision")
        assert dim is not None
        assert dim.name == "precision"

    def test_get_missing_dimension(self, good_scorecard):
        assert good_scorecard.get_dimension("nonexistent") is None

    def test_to_dict(self, good_scorecard):
        d = good_scorecard.to_dict()
        assert "total" in d
        assert "dimensions" in d
        assert "precision" in d["dimensions"]

    def test_repr(self, good_scorecard):
        assert "total=" in repr(good_scorecard)


class TestScorer:
    def test_good_report_high_scores(self, scorer, good_report):
        card = scorer.score_benchmark(good_report)
        assert card.total > 50.0
        prec = card.get_dimension("precision")
        assert prec is not None
        assert prec.value == 100.0

    def test_bad_report_low_scores(self, scorer, bad_report):
        card = scorer.score_benchmark(bad_report)
        assert card.total < 50.0

    def test_compare(self, scorer, good_report, bad_report):
        good_card = scorer.score_benchmark(good_report)
        bad_card = scorer.score_benchmark(bad_report)
        deltas = scorer.compare(bad_card, good_card)
        assert deltas["total"] > 0
        assert deltas["precision"] > 0

    def test_weights_normalized(self):
        scorer = Scorer(precision_weight=1, recall_weight=1, latency_weight=1, cost_weight=1)
        # Should still work — weights get normalized internally
        from spiral_06_state_of_art.benchmark_suite import BenchmarkReport

        report = BenchmarkReport(
            results=[],
            total_questions=0,
            precision_at_k=0,
            mean_keyword_recall=0,
            mean_latency_s=0,
            total_cost=0,
            pass_rate=0,
        )
        card = scorer.score_benchmark(report)
        # Empty report: precision=0, recall=0, but latency=100 (0s=fast) and cost=100 ($0=free)
        assert card.total > 0  # latency + cost tiers give non-zero scores

    def test_latency_score_tiers(self, scorer, good_report):
        card = scorer.score_benchmark(good_report)
        latency_dim = card.get_dimension("latency")
        assert latency_dim is not None
        assert latency_dim.value >= 85.0  # < 0.5s should be high

    def test_cost_score_tiers(self, scorer, good_report):
        card = scorer.score_benchmark(good_report)
        cost_dim = card.get_dimension("cost")
        assert cost_dim is not None
        assert cost_dim.value >= 70.0  # low cost
