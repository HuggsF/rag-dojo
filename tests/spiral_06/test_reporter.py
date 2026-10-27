"""Tests for spiral_06_state_of_art.reporter."""

from spiral_06_state_of_art.cycle_manager import CycleResult
from spiral_06_state_of_art.proposer import Proposal
from spiral_06_state_of_art.reporter import CycleReport


class TestReporter:
    def test_generate_empty(self, reporter, good_scorecard):
        report = reporter.generate([], good_scorecard, good_scorecard)
        assert isinstance(report, CycleReport)
        assert report.cycles_run == 0
        assert report.total_improvement == 0.0

    def test_generate_with_cycles(self, reporter, bad_scorecard, good_scorecard):
        cycle = CycleResult(
            cycle_number=1,
            proposal=Proposal(
                param="chunk_size",
                current_value=512,
                proposed_value=256,
                reason="too big",
                expected_impact="recall",
                confidence=0.6,
            ),
            solver_result=None,
            verification=None,
            outcome="improved",
        )
        report = reporter.generate([cycle], bad_scorecard, good_scorecard)
        assert report.cycles_run == 1
        assert report.total_improvement > 0

    def test_to_text(self, reporter, good_scorecard):
        report = reporter.generate([], good_scorecard, good_scorecard)
        text = report.to_text()
        assert "Optimization Report" in text
        assert "Cycles run: 0" in text

    def test_to_text_with_details(self, reporter, bad_scorecard, good_scorecard):
        cycle = CycleResult(
            cycle_number=1,
            proposal=Proposal(
                param="top_k",
                current_value=5,
                proposed_value=4,
                reason="low precision",
                expected_impact="precision",
                confidence=0.7,
            ),
            solver_result=None,
            verification=None,
            outcome="improved",
        )
        report = reporter.generate([cycle], bad_scorecard, good_scorecard)
        text = report.to_text()
        assert "Cycle 1" in text

    def test_repr(self, reporter, good_scorecard):
        report = reporter.generate([], good_scorecard, good_scorecard)
        assert "cycles=" in repr(report)
