"""Tests for spiral_06_state_of_art.benchmark_suite."""

from spiral_06_state_of_art.benchmark_suite import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkSuite,
)


class TestBenchmarkQuestion:
    def test_keyword_recall_full_match(self):
        q = BenchmarkQuestion(
            question="test",
            ground_truth="test",
            keywords=["fast", "python"],
        )
        assert q.keyword_recall("FastAPI is a fast Python framework") == 1.0

    def test_keyword_recall_partial_match(self):
        q = BenchmarkQuestion(
            question="test",
            ground_truth="test",
            keywords=["fast", "python", "rust"],
        )
        recall = q.keyword_recall("FastAPI is fast")
        assert recall == pytest.approx(1 / 3)

    def test_keyword_recall_no_match(self):
        q = BenchmarkQuestion(
            question="test",
            ground_truth="test",
            keywords=["quantum", "physics"],
        )
        assert q.keyword_recall("FastAPI web framework") == 0.0

    def test_keyword_recall_empty_keywords(self):
        q = BenchmarkQuestion(question="test", ground_truth="test", keywords=[])
        assert q.keyword_recall("anything") == 1.0


class TestBenchmarkSuite:
    def test_default_has_20_questions(self):
        suite = BenchmarkSuite()
        assert suite.size == 20

    def test_custom_questions(self, sample_questions):
        suite = BenchmarkSuite(questions=sample_questions)
        assert suite.size == 3

    def test_evaluate_answer_hit(self, benchmark_suite, sample_questions):
        q = sample_questions[0]
        result = benchmark_suite.evaluate_answer(q, "FastAPI is a fast python web framework")
        assert isinstance(result, BenchmarkResult)
        assert result.hit is True
        assert result.keyword_recall == 1.0

    def test_evaluate_answer_miss(self, benchmark_suite, sample_questions):
        q = sample_questions[0]
        result = benchmark_suite.evaluate_answer(q, "I have no idea")
        assert result.hit is False

    def test_compile_report_empty(self, benchmark_suite):
        report = benchmark_suite.compile_report([])
        assert report.total_questions == 0
        assert report.precision_at_k == 0.0

    def test_compile_report_with_results(self, benchmark_suite, sample_questions):
        results = [
            benchmark_suite.evaluate_answer(q, q.ground_truth, 0.1, 0.001) for q in sample_questions
        ]
        report = benchmark_suite.compile_report(results)
        assert report.total_questions == 3
        assert report.precision_at_k == 1.0
        assert report.pass_rate == 1.0
        assert report.total_cost == pytest.approx(0.003)

    def test_save_and_load(self, benchmark_suite, tmp_path):
        path = str(tmp_path / "bench.json")
        benchmark_suite.save_to_file(path)

        loaded = BenchmarkSuite()
        loaded.load_from_file(path)
        assert loaded.size == benchmark_suite.size
        assert loaded.questions[0].question == benchmark_suite.questions[0].question

    def test_repr(self, benchmark_suite, sample_questions):
        result = benchmark_suite.evaluate_answer(sample_questions[0], "test", 0.1, 0.0)
        assert "HIT" in repr(result) or "MISS" in repr(result)


import pytest  # noqa: E402
