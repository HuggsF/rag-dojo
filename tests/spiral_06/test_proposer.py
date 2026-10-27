"""Tests for spiral_06_state_of_art.proposer."""

from spiral_06_state_of_art.proposer import Proposal, Proposer


class TestProposal:
    def test_has_required_fields(self):
        p = Proposal(
            param="chunk_size",
            current_value=512,
            proposed_value=448,
            reason="too big",
            expected_impact="recall",
            confidence=0.6,
        )
        assert p.param == "chunk_size"
        assert p.confidence == 0.6

    def test_repr(self):
        p = Proposal(
            param="top_k",
            current_value=5,
            proposed_value=4,
            reason="low precision",
            expected_impact="precision",
            confidence=0.7,
        )
        assert "top_k" in repr(p)


class TestProposer:
    def test_no_proposals_for_good_scores(self, proposer, good_scorecard):
        proposals = proposer.analyze(good_scorecard)
        assert len(proposals) == 0

    def test_proposals_for_bad_scores(self, proposer, bad_scorecard):
        proposals = proposer.analyze(bad_scorecard)
        assert len(proposals) > 0
        assert all(isinstance(p, Proposal) for p in proposals)

    def test_best_proposal_returns_highest_confidence(self, proposer, bad_scorecard):
        best = proposer.best_proposal(bad_scorecard)
        assert best is not None
        all_proposals = proposer.analyze(bad_scorecard)
        if len(all_proposals) > 1:
            assert best.confidence >= all_proposals[-1].confidence

    def test_best_proposal_none_for_good(self, proposer, good_scorecard):
        assert proposer.best_proposal(good_scorecard) is None

    def test_proposals_sorted_by_confidence(self, proposer, bad_scorecard):
        proposals = proposer.analyze(bad_scorecard)
        if len(proposals) >= 2:
            for i in range(len(proposals) - 1):
                assert proposals[i].confidence >= proposals[i + 1].confidence

    def test_proposals_respect_tunable_bounds(self, bad_scorecard):
        proposer = Proposer(current_config={"chunk_size": 128})
        proposals = proposer.analyze(bad_scorecard)
        for p in proposals:
            if p.param == "chunk_size":
                assert p.proposed_value >= 128  # can't go below min
