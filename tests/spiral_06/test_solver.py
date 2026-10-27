"""Tests for spiral_06_state_of_art.solver."""

from spiral_06_state_of_art.proposer import Proposal
from spiral_06_state_of_art.solver import SolverAction, SolverResult


class TestSolverAction:
    def test_repr_success(self):
        a = SolverAction(param="top_k", old_value=5, new_value=4, applied=True)
        assert "OK" in repr(a)

    def test_repr_failure(self):
        a = SolverAction(param="top_k", old_value=5, new_value=4, applied=False, error="oops")
        assert "FAILED" in repr(a)


class TestSolver:
    def test_apply_valid_proposal(self, solver):
        proposal = Proposal(
            param="chunk_size",
            current_value=512,
            proposed_value=448,
            reason="test",
            expected_impact="recall",
            confidence=0.6,
        )
        action = solver.apply(proposal)
        assert action.applied is True
        assert solver.config["chunk_size"] == 448

    def test_apply_rejects_non_tunable(self, solver):
        proposal = Proposal(
            param="not_a_param",
            current_value=1,
            proposed_value=2,
            reason="test",
            expected_impact="test",
            confidence=0.5,
        )
        action = solver.apply(proposal)
        assert action.applied is False
        assert "not tunable" in action.error

    def test_apply_rejects_out_of_bounds(self, solver):
        proposal = Proposal(
            param="chunk_size",
            current_value=512,
            proposed_value=9999,
            reason="test",
            expected_impact="test",
            confidence=0.5,
        )
        action = solver.apply(proposal)
        assert action.applied is False
        assert "outside bounds" in action.error

    def test_apply_proposals_batch(self, solver):
        proposals = [
            Proposal(
                param="chunk_size",
                current_value=512,
                proposed_value=448,
                reason="test",
                expected_impact="recall",
                confidence=0.6,
            ),
            Proposal(
                param="top_k",
                current_value=5,
                proposed_value=4,
                reason="test",
                expected_impact="precision",
                confidence=0.7,
            ),
        ]
        result = solver.apply_proposals(proposals)
        assert isinstance(result, SolverResult)
        assert result.success is True
        assert result.files_modified == 2

    def test_revert(self, solver):
        solver.apply(
            Proposal(
                param="chunk_size",
                current_value=512,
                proposed_value=256,
                reason="test",
                expected_impact="test",
                confidence=0.5,
            )
        )
        assert solver.config["chunk_size"] == 256
        reverted = solver.revert()
        assert reverted["chunk_size"] == 512

    def test_validate_immutable(self, solver):
        violations = solver.validate_immutable(["CLAUDE.md", "some_file.py"])
        assert "CLAUDE.md" in violations
        assert "some_file.py" not in violations

    def test_config_version_starts_at_zero(self, solver):
        assert solver.config_version() == 0

    def test_config_version_increments_on_apply(self, solver):
        proposal = Proposal(
            param="chunk_size",
            current_value=512,
            proposed_value=448,
            reason="test",
            expected_impact="recall",
            confidence=0.6,
        )
        solver.apply(proposal)
        assert solver.config_version() == 1
        solver.apply(proposal)
        assert solver.config_version() == 2

    def test_config_version_not_incremented_on_failed_apply(self, solver):
        proposal = Proposal(
            param="not_a_param",
            current_value=1,
            proposed_value=2,
            reason="test",
            expected_impact="test",
            confidence=0.5,
        )
        solver.apply(proposal)
        assert solver.config_version() == 0

    def test_config_history_empty_initially(self, solver):
        assert solver.config_history() == []

    def test_config_history_records_changes(self, solver):
        p1 = Proposal(
            param="chunk_size",
            current_value=512,
            proposed_value=448,
            reason="test",
            expected_impact="recall",
            confidence=0.6,
        )
        p2 = Proposal(
            param="top_k",
            current_value=5,
            proposed_value=4,
            reason="test",
            expected_impact="precision",
            confidence=0.7,
        )
        solver.apply(p1)
        solver.apply(p2)
        history = solver.config_history()
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[0]["param"] == "chunk_size"
        assert history[0]["old_value"] == 512
        assert history[0]["new_value"] == 448
        assert "timestamp" in history[0]
        assert history[1]["version"] == 2
        assert history[1]["param"] == "top_k"
        assert history[1]["old_value"] == 5
        assert history[1]["new_value"] == 4

    def test_config_history_not_mutated_by_caller(self, solver):
        proposal = Proposal(
            param="chunk_size",
            current_value=512,
            proposed_value=448,
            reason="test",
            expected_impact="recall",
            confidence=0.6,
        )
        solver.apply(proposal)
        history = solver.config_history()
        history.clear()
        assert len(solver.config_history()) == 1  # original not affected

    def test_config_snapshot(self, solver):
        proposal = Proposal(
            param="top_k",
            current_value=5,
            proposed_value=3,
            reason="test",
            expected_impact="test",
            confidence=0.5,
        )
        result = solver.apply_proposals([proposal])
        assert result.config_snapshot_before["top_k"] == 5
        assert result.config_snapshot_after["top_k"] == 3
