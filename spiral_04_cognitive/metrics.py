"""Observability metrics — Protocol-based metrics collection for the RAG pipeline.

Analogy for a C dev:
    Like perf_event_open() — instrument the kernel without modifying it.
    The MetricsCollector Protocol is the interface; implementations decide
    where the data goes (memory, Prometheus, stdout, etc.).

Three implementations:

1. NoopMetrics     — /dev/null. Zero overhead. Default in tests.
2. InMemoryMetrics — in-process counters + histograms. Used by the CLI dashboard.
3. PrometheusMetrics — exports to prometheus_client. Optional dependency.
                       Import is conditional — prometheus_client is NOT required.

Integration pattern:
    The factory (pipeline_factory.py) injects a MetricsCollector into pipeline
    components. Components call record_* methods without knowing the implementation.
    This is the classic Strategy pattern over a Protocol.

Usage:
    from spiral_04_cognitive.metrics import InMemoryMetrics

    metrics = InMemoryMetrics()
    metrics.record_latency("retrieval", 0.042)
    metrics.record_tokens("llama-3.3-70b", 512)
    print(metrics.summary())
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

# ─── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class MetricsCollector(Protocol):
    """Interface for pipeline metrics collection.

    Implementations must be thread-safe — multiple threads may call
    record_* concurrently in production pod environments.
    """

    def record_latency(self, operation: str, seconds: float) -> None:
        """Record how long an operation took.

        Args:
            operation: Name of the operation (e.g. "retrieval", "generation").
            seconds: Elapsed time in seconds.
        """
        ...

    def record_tokens(self, model: str, count: int) -> None:
        """Record token usage for an LLM call.

        Args:
            model: Model identifier (e.g. "llama-3.3-70b-versatile").
            count: Total tokens consumed (input + output).
        """
        ...

    def record_cache_hit(self, layer: str) -> None:
        """Record a cache hit at a specific pipeline layer.

        Args:
            layer: Layer name (e.g. "L0_semantic", "L1_exact", "L2_llm").
        """
        ...

    def record_cache_miss(self, layer: str) -> None:
        """Record a cache miss at a specific pipeline layer.

        Args:
            layer: Layer name.
        """
        ...


# ─── Noop implementation ──────────────────────────────────────────────────────


class NoopMetrics:
    """Metrics collector that discards everything. Zero overhead.

    Use in unit tests or when observability is not needed.
    """

    def record_latency(self, operation: str, seconds: float) -> None:
        pass

    def record_tokens(self, model: str, count: int) -> None:
        pass

    def record_cache_hit(self, layer: str) -> None:
        pass

    def record_cache_miss(self, layer: str) -> None:
        pass


# ─── In-memory implementation ─────────────────────────────────────────────────


class InMemoryMetrics:
    """Stores metrics in memory for dashboard display.

    Not persistent across restarts. Use for the CLI dashboard and
    local development visibility.

    Thread safety: uses defaultdict which is GIL-safe for simple increments.
    For production multi-threaded use, prefer PrometheusMetrics.
    """

    def __init__(self) -> None:
        # latencies[operation] = list of seconds
        self._latencies: dict[str, list[float]] = defaultdict(list)
        # tokens[model] = total token count
        self._tokens: dict[str, int] = defaultdict(int)
        # cache_hits[layer] = count
        self._cache_hits: dict[str, int] = defaultdict(int)
        # cache_misses[layer] = count
        self._cache_misses: dict[str, int] = defaultdict(int)

    def record_latency(self, operation: str, seconds: float) -> None:
        self._latencies[operation].append(seconds)

    def record_tokens(self, model: str, count: int) -> None:
        self._tokens[model] += count

    def record_cache_hit(self, layer: str) -> None:
        self._cache_hits[layer] += 1

    def record_cache_miss(self, layer: str) -> None:
        self._cache_misses[layer] += 1

    def avg_latency(self, operation: str) -> float:
        """Average latency in seconds for an operation. Returns 0.0 if none recorded."""
        samples = self._latencies.get(operation, [])
        return sum(samples) / len(samples) if samples else 0.0

    def p99_latency(self, operation: str) -> float:
        """99th percentile latency in seconds. Returns 0.0 if fewer than 2 samples."""
        samples = sorted(self._latencies.get(operation, []))
        if len(samples) < 2:
            return 0.0
        idx = int(len(samples) * 0.99)
        return samples[min(idx, len(samples) - 1)]

    def cache_hit_rate(self, layer: str) -> float:
        """Cache hit rate [0.0, 1.0] for a layer. Returns 0.0 if no events recorded."""
        hits = self._cache_hits.get(layer, 0)
        misses = self._cache_misses.get(layer, 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def total_tokens(self, model: str) -> int:
        """Total tokens consumed for a model."""
        return self._tokens.get(model, 0)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for dashboard display."""
        return {
            "latencies": {
                op: {"avg": self.avg_latency(op), "p99": self.p99_latency(op), "n": len(samples)}
                for op, samples in self._latencies.items()
            },
            "tokens": dict(self._tokens),
            "cache": {
                layer: {
                    "hits": self._cache_hits.get(layer, 0),
                    "misses": self._cache_misses.get(layer, 0),
                    "hit_rate": self.cache_hit_rate(layer),
                }
                for layer in set(self._cache_hits) | set(self._cache_misses)
            },
        }

    def reset(self) -> None:
        """Clear all recorded metrics. Useful between test runs."""
        self._latencies.clear()
        self._tokens.clear()
        self._cache_hits.clear()
        self._cache_misses.clear()


# ─── Prometheus implementation ────────────────────────────────────────────────


class PrometheusMetrics:
    """Exports metrics to Prometheus format.

    Requires: pip install prometheus_client
    This dependency is NOT in requirements.txt — it is optional.
    Import fails gracefully at instantiation time (not at module import).

    Usage:
        metrics = PrometheusMetrics(port=8000)
        # Visit http://localhost:8000/metrics for Prometheus scrape endpoint
    """

    def __init__(self, port: int = 8000, start_server: bool = False) -> None:
        try:
            import prometheus_client as prom
        except ImportError as exc:
            raise ImportError(
                "prometheus_client is not installed. Run: pip install prometheus_client"
            ) from exc

        self._latency_hist = prom.Histogram(
            "rag_operation_latency_seconds",
            "RAG pipeline operation latency",
            ["operation"],
        )
        self._token_counter = prom.Counter(
            "rag_tokens_total",
            "Total tokens consumed by LLM calls",
            ["model"],
        )
        self._cache_hits_counter = prom.Counter(
            "rag_cache_hits_total",
            "Cache hits per layer",
            ["layer"],
        )
        self._cache_misses_counter = prom.Counter(
            "rag_cache_misses_total",
            "Cache misses per layer",
            ["layer"],
        )

        if start_server:
            prom.start_http_server(port)

    def record_latency(self, operation: str, seconds: float) -> None:
        self._latency_hist.labels(operation=operation).observe(seconds)

    def record_tokens(self, model: str, count: int) -> None:
        self._token_counter.labels(model=model).inc(count)

    def record_cache_hit(self, layer: str) -> None:
        self._cache_hits_counter.labels(layer=layer).inc()

    def record_cache_miss(self, layer: str) -> None:
        self._cache_misses_counter.labels(layer=layer).inc()


# ─── Context manager helper ───────────────────────────────────────────────────


class timed:  # noqa: N801 — lowercase intentional (context manager style)
    """Context manager to record operation latency.

    Usage:
        with timed(metrics, "retrieval"):
            results = retriever.retrieve(query)
    """

    def __init__(self, collector: MetricsCollector, operation: str) -> None:
        self._collector = collector
        self._operation = operation
        self._start: float = 0.0

    def __enter__(self) -> "timed":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.monotonic() - self._start
        self._collector.record_latency(self._operation, elapsed)
