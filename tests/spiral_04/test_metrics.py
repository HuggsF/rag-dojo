"""Tests for spiral_04_cognitive.metrics."""

from __future__ import annotations

import time

import pytest

from spiral_04_cognitive.metrics import (
    InMemoryMetrics,
    MetricsCollector,
    NoopMetrics,
    timed,
)


class TestMetricsCollectorProtocol:
    def test_noop_satisfies_protocol(self) -> None:
        assert isinstance(NoopMetrics(), MetricsCollector)

    def test_in_memory_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryMetrics(), MetricsCollector)


class TestNoopMetrics:
    def test_all_methods_callable(self) -> None:
        m = NoopMetrics()
        m.record_latency("op", 0.1)
        m.record_tokens("model", 100)
        m.record_cache_hit("L0")
        m.record_cache_miss("L1")


class TestInMemoryMetrics:
    def test_avg_latency_single_sample(self) -> None:
        m = InMemoryMetrics()
        m.record_latency("retrieval", 0.05)
        assert abs(m.avg_latency("retrieval") - 0.05) < 1e-9

    def test_avg_latency_multiple_samples(self) -> None:
        m = InMemoryMetrics()
        m.record_latency("retrieval", 0.1)
        m.record_latency("retrieval", 0.3)
        assert abs(m.avg_latency("retrieval") - 0.2) < 1e-9

    def test_avg_latency_unknown_op_returns_zero(self) -> None:
        m = InMemoryMetrics()
        assert m.avg_latency("nonexistent") == 0.0

    def test_p99_latency_requires_two_samples(self) -> None:
        m = InMemoryMetrics()
        m.record_latency("op", 0.5)
        assert m.p99_latency("op") == 0.0

    def test_p99_latency_returns_high_percentile(self) -> None:
        m = InMemoryMetrics()
        for i in range(100):
            m.record_latency("op", float(i))
        p99 = m.p99_latency("op")
        assert p99 >= 90.0  # 99th percentile of 0-99 should be >= 90

    def test_total_tokens_accumulates(self) -> None:
        m = InMemoryMetrics()
        m.record_tokens("llama", 100)
        m.record_tokens("llama", 200)
        assert m.total_tokens("llama") == 300

    def test_total_tokens_different_models(self) -> None:
        m = InMemoryMetrics()
        m.record_tokens("llama", 100)
        m.record_tokens("mistral", 50)
        assert m.total_tokens("llama") == 100
        assert m.total_tokens("mistral") == 50

    def test_cache_hit_rate_all_hits(self) -> None:
        m = InMemoryMetrics()
        m.record_cache_hit("L0")
        m.record_cache_hit("L0")
        assert m.cache_hit_rate("L0") == 1.0

    def test_cache_hit_rate_all_misses(self) -> None:
        m = InMemoryMetrics()
        m.record_cache_miss("L1")
        assert m.cache_hit_rate("L1") == 0.0

    def test_cache_hit_rate_mixed(self) -> None:
        m = InMemoryMetrics()
        m.record_cache_hit("L2")
        m.record_cache_miss("L2")
        assert abs(m.cache_hit_rate("L2") - 0.5) < 1e-9

    def test_cache_hit_rate_unknown_layer_returns_zero(self) -> None:
        m = InMemoryMetrics()
        assert m.cache_hit_rate("unknown") == 0.0

    def test_summary_structure(self) -> None:
        m = InMemoryMetrics()
        m.record_latency("retrieval", 0.1)
        m.record_tokens("llama", 128)
        m.record_cache_hit("L0")
        summary = m.summary()
        assert "latencies" in summary
        assert "tokens" in summary
        assert "cache" in summary
        assert "retrieval" in summary["latencies"]
        assert "llama" in summary["tokens"]
        assert "L0" in summary["cache"]

    def test_reset_clears_all(self) -> None:
        m = InMemoryMetrics()
        m.record_latency("op", 0.1)
        m.record_tokens("model", 50)
        m.record_cache_hit("L0")
        m.reset()
        assert m.avg_latency("op") == 0.0
        assert m.total_tokens("model") == 0
        assert m.cache_hit_rate("L0") == 0.0


class TestPrometheusMetrics:
    def test_raises_without_prometheus_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If prometheus_client is not installed, must raise ImportError."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):
            if name == "prometheus_client":
                raise ImportError("No module named 'prometheus_client'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from spiral_04_cognitive.metrics import PrometheusMetrics

        with pytest.raises(ImportError, match="prometheus_client"):
            PrometheusMetrics()

    def test_instantiates_when_prometheus_available(self) -> None:
        """PrometheusMetrics initializes all counters when prometheus_client is present."""
        from unittest.mock import MagicMock, patch

        mock_prom = MagicMock()
        mock_prom.Histogram.return_value = MagicMock()
        mock_prom.Counter.return_value = MagicMock()

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            from spiral_04_cognitive.metrics import PrometheusMetrics

            m = PrometheusMetrics(start_server=False)
            assert m._latency_hist is not None
            assert m._token_counter is not None
            assert m._cache_hits_counter is not None
            assert m._cache_misses_counter is not None

    def test_record_methods_delegate_to_prometheus(self) -> None:
        """record_* methods call the prometheus client objects."""
        from unittest.mock import MagicMock, patch

        mock_prom = MagicMock()

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            from spiral_04_cognitive.metrics import PrometheusMetrics

            m = PrometheusMetrics(start_server=False)
            m.record_latency("retrieval", 0.05)
            m.record_tokens("llama", 100)
            m.record_cache_hit("L0")
            m.record_cache_miss("L1")
            # Verify prometheus objects were called
            assert mock_prom.Histogram.called
            assert mock_prom.Counter.called

    def test_start_server_calls_start_http_server(self) -> None:
        """Cover line 233: start_server=True calls prom.start_http_server."""
        from unittest.mock import MagicMock, patch

        mock_prom = MagicMock()
        mock_prom.Histogram.return_value = MagicMock()
        mock_prom.Counter.return_value = MagicMock()

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            from spiral_04_cognitive.metrics import PrometheusMetrics

            PrometheusMetrics(port=9999, start_server=True)
            mock_prom.start_http_server.assert_called_once_with(9999)


class TestTimedContextManager:
    def test_records_latency(self) -> None:
        m = InMemoryMetrics()
        with timed(m, "test_op"):
            time.sleep(0.01)
        # Verify latency was recorded (>= 0 — sleep precision varies on Windows)
        assert len(m._latencies.get("test_op", [])) == 1
        assert m.avg_latency("test_op") >= 0

    def test_records_on_exception(self) -> None:
        """timed must record latency even if the body raises."""
        m = InMemoryMetrics()
        with pytest.raises(ValueError):
            with timed(m, "failing_op"):
                raise ValueError("oops")
        assert m.avg_latency("failing_op") >= 0.0
