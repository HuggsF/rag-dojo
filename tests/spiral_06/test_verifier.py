"""Tests for spiral_06_state_of_art.verifier."""

from spiral_06_state_of_art.verifier import VerificationResult


class TestVerifier:
    def test_run_benchmark_returns_scorecard(self, verifier):
        def good_answer(q):
            return (
                "FastAPI is a fast python web framework with middleware and depends injection for request processing",
                0.1,
                0.0,
            )

        report, scorecard = verifier.run_benchmark(good_answer)
        assert report.total_questions == 3
        assert scorecard.total > 0

    def test_verify_improved(self, verifier, bad_scorecard):
        def good_answer(q):
            return (
                "FastAPI is a fast python web framework with middleware and depends injection for request processing",
                0.2,
                0.001,
            )

        result = verifier.verify(bad_scorecard, good_answer)
        assert isinstance(result, VerificationResult)
        assert result.improved is True
        assert result.recommendation == "keep"

    def test_verify_regressed(self, verifier, good_scorecard):
        def bad_answer(q):
            return ("I don't know", 10.0, 0.1)

        result = verifier.verify(good_scorecard, bad_answer)
        assert result.regressed is True
        assert result.recommendation == "revert"

    def test_verify_neutral(self, verifier, good_scorecard):
        def same_answer(q):
            return (
                "FastAPI is a fast python web framework with middleware and depends injection for request processing",
                0.3,
                0.001,
            )

        result = verifier.verify(good_scorecard, same_answer)
        assert result.recommendation in ("keep", "neutral")

    def test_deltas_computed(self, verifier, bad_scorecard):
        def good_answer(q):
            return (
                "FastAPI is a fast python web framework with middleware and depends injection for request processing",
                0.1,
                0.0,
            )

        result = verifier.verify(bad_scorecard, good_answer)
        assert "total" in result.deltas
        assert "precision" in result.deltas

    def test_repr(self, verifier, bad_scorecard):
        def fn(q):
            return ("test", 0.1, 0.0)

        result = verifier.verify(bad_scorecard, fn)
        assert "delta_total" in repr(result)
