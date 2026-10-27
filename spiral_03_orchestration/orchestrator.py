"""Orchestrator — the main loop that ties tension, confidence, and retrieval together.

Analogy for a C dev:
- This is the event loop / main() for Spiral 03
- Flow: query → evaluate tension → compass decides → execute → measure post-delta
- Like a game loop: read input → process → render → check state → repeat

Usage:
    python -m spiral_03_orchestration.orchestrator --profile STANDARD "your question"
    python -m spiral_03_orchestration.orchestrator --profile CRITICAL "your question"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel

from spiral_03_orchestration.compass import Action, CompassReading, MendaCompass
from spiral_03_orchestration.confidence import ConfidenceResult
from spiral_03_orchestration.fallback import FallbackCascade
from spiral_03_orchestration.tension import TensionHistory, TensionMetrics
from spiral_03_orchestration.tension_profile import (
    ProfileType,
    TensionProfile,
    get_profile,
)

console = Console()


@dataclass
class OrchestrationResult:
    """Full result of an orchestrated query."""

    query: str
    profile: TensionProfile
    pre_tension: TensionMetrics
    confidence: ConfidenceResult
    action: Action
    answer: str | None
    post_tension: TensionMetrics | None
    fallback_attempts: list[dict[str, Any]] = field(default_factory=list)
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "query": self.query,
            "profile": self.profile.to_dict(),
            "pre_tension": self.pre_tension.to_dict(),
            "confidence": self.confidence.to_dict(),
            "action": self.action.value,
            "answer": self.answer,
            "post_tension": self.post_tension.to_dict() if self.post_tension else None,
            "fallback_attempts": self.fallback_attempts,
            "retrieval_stats": self.retrieval_stats,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class Orchestrator:
    """Main orchestration loop for tension-aware RAG.

    Flow:
    1. Evaluate pre-action tension (creativity, logic, delta from history)
    2. Compass decides: RETRIEVE or FALLBACK
    3. If RETRIEVE: execute hybrid retrieval via spiral_02 pipeline
    4. If FALLBACK: try cascade of 7 strategies
    5. Measure post-action delta (did the state improve?)
    6. Record tension history for streak detection
    """

    def __init__(self, profile: str | ProfileType = "STANDARD"):
        self._profile = get_profile(profile)
        self._compass = MendaCompass()
        self._history = TensionHistory()
        self._pipeline = None  # lazy-loaded

    @property
    def profile(self) -> TensionProfile:
        """Active tension profile."""
        return self._profile

    @property
    def history(self) -> TensionHistory:
        """Tension history for streak detection."""
        return self._history

    @property
    def pipeline(self):
        """Lazy-load the spiral_02 pipeline."""
        if self._pipeline is None:
            from spiral_02_graph.pipeline import GraphRAGPipeline

            self._pipeline = GraphRAGPipeline()
        return self._pipeline

    def set_profile(self, profile: str | ProfileType) -> None:
        """Change the active tension profile."""
        self._profile = get_profile(profile)

    def evaluate_tension(
        self,
        query: str,
        previous_delta: float = 0.5,
    ) -> TensionMetrics:
        """Evaluate tension metrics for a query.

        Heuristics for initial tension:
        - creativity: based on query complexity (longer/diverse = more creative)
        - logic: inverse of creativity (they're opposing forces)
        - delta_estado: from previous action result, or 0.5 for first query

        Args:
            query: The user's question.
            previous_delta: Delta from the last action (0.5 if first query).

        Returns:
            TensionMetrics snapshot.
        """
        # Simple heuristic: query length and word diversity drive creativity
        words = query.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)

        # Longer, more diverse queries → more exploration needed
        length_factor = min(len(words) / 20.0, 1.0)
        creativity = unique_ratio * 0.6 + length_factor * 0.4

        # Logic is the opposing force
        logic = 1.0 - (creativity * 0.5)

        return TensionMetrics(
            creativity=creativity,
            logic=logic,
            delta_estado=previous_delta,
        )

    def measure_post_delta(
        self,
        result: dict[str, Any] | None,
    ) -> float:
        """Measure how much the state changed after an action.

        Uses retrieval quality signals:
        - answer presence and length
        - number of sources found
        - average similarity scores

        Returns:
            Delta in [0.0, 1.0]. 0.0 = nothing changed. 1.0 = perfect state change.
        """
        if result is None:
            return 0.0

        signals: list[float] = []

        # Did we get an answer?
        answer = result.get("answer", "")
        if answer and answer != "No relevant context found.":
            answer_quality = min(len(answer) / 200.0, 1.0)
            signals.append(answer_quality)
        else:
            signals.append(0.0)

        # How many sources?
        sources = result.get("sources", [])
        source_signal = min(len(sources) / 5.0, 1.0)
        signals.append(source_signal)

        # Average source score
        if sources:
            avg_score = sum(s.get("score", 0) for s in sources) / len(sources)
            signals.append(avg_score)

        return sum(signals) / len(signals)

    def ask(
        self,
        query: str,
        show_details: bool = True,
    ) -> OrchestrationResult:
        """Execute an orchestrated query with tension evaluation.

        Args:
            query: The user's question.
            show_details: Whether to display Rich output.

        Returns:
            OrchestrationResult with full diagnostics.
        """
        start = time.time()

        # Determine previous delta from history
        if self._history.records:
            previous_delta = self._history.records[-1].delta_estado
        else:
            previous_delta = 0.5  # first query gets neutral delta

        # Step 1: Evaluate pre-action tension
        pre_metrics = self.evaluate_tension(query, previous_delta)

        # Step 2: Compass decides
        reading = self._compass.evaluate(pre_metrics, self._profile)

        if show_details:
            self._display_pre_action(reading)

        # Check for zero-delta streak (Inertia Zero Law)
        if self._history.should_swap_strategy:
            if show_details:
                console.print(
                    f"[yellow]Inertia Zero Law: {self._history.zero_delta_streak} "
                    f"consecutive zero-delta queries. Auto-swapping strategy.[/]"
                )
            reading = CompassReading(
                action=Action.FALLBACK,
                confidence=reading.confidence,
                metrics=pre_metrics,
                profile=self._profile,
                reason="Inertia Zero Law triggered — forcing fallback cascade.",
            )

        # Step 3: Execute action
        answer = None
        retrieval_stats: dict[str, Any] = {}
        fallback_attempts: list[dict[str, Any]] = []

        if reading.action == Action.RETRIEVE:
            result = self.pipeline.ask_hybrid(
                query,
                top_k=self._profile.max_chunks,
                show_sources=show_details,
            )
            answer = result.get("answer")
            retrieval_stats = result

        elif reading.action == Action.FALLBACK:
            cascade = FallbackCascade()
            while not cascade.is_exhausted:
                fallback = cascade.next()
                fallback_attempts.append(fallback.to_dict())  # type: ignore[union-attr]

                if fallback.adjustments.get("abort"):  # type: ignore[union-attr]
                    if show_details:
                        console.print(
                            "[red]All fallback strategies exhausted. Reporting failure.[/]"
                        )
                    break

                if show_details:
                    console.print(
                        f"[yellow]Fallback: {fallback.strategy.value} — {fallback.description}[/]"  # type: ignore[union-attr]
                    )

                # Try retrieval with adjusted parameters
                result = self.pipeline.ask_hybrid(
                    query,
                    top_k=self._profile.max_chunks,
                    show_sources=False,
                )
                answer = result.get("answer")
                retrieval_stats = result

                if answer and answer != "No relevant context found.":
                    break

        # Step 4: Measure post-action delta
        post_delta = self.measure_post_delta(retrieval_stats)
        post_metrics = TensionMetrics(
            creativity=pre_metrics.creativity,
            logic=pre_metrics.logic,
            delta_estado=post_delta,
        )

        # Step 5: Record in history
        self._history.record(post_metrics)

        elapsed = time.time() - start

        orchestration_result = OrchestrationResult(
            query=query,
            profile=self._profile,
            pre_tension=pre_metrics,
            confidence=reading.confidence,
            action=reading.action,
            answer=answer,
            post_tension=post_metrics,
            fallback_attempts=fallback_attempts,
            retrieval_stats=retrieval_stats,
            elapsed_s=elapsed,
        )

        if show_details:
            self._display_post_action(orchestration_result)

        return orchestration_result

    def _display_pre_action(self, reading: CompassReading) -> None:
        """Display pre-action tension evaluation."""
        m = reading.metrics
        c = reading.confidence

        console.print(
            Panel(
                f"Creativity: {m.creativity:.4f}  |  Logic: {m.logic:.4f}  |  "
                f"Delta: {m.delta_estado:.4f}\n"
                f"Confidence: {c.score:.4f}  |  Threshold: {c.threshold:.4f}  |  "
                f"Passed: {c.passed}\n"
                f"Action: [bold]{reading.action.value}[/]  |  "
                f"Profile: {reading.profile.name.value}\n"
                f"Reason: {reading.reason}",
                title=f"[bold cyan]Tension Field — {reading.profile.name.value}",
            )
        )

    def _display_post_action(self, result: OrchestrationResult) -> None:
        """Display post-action summary."""
        if result.post_tension is None:
            return

        delta_change = result.post_tension.delta_estado - result.pre_tension.delta_estado
        delta_emoji = "+" if delta_change > 0 else ""

        streak = self._history.zero_delta_streak

        console.print(
            Panel(
                f"Post-delta: {result.post_tension.delta_estado:.4f}  "
                f"({delta_emoji}{delta_change:.4f})\n"
                f"Zero-delta streak: {streak}\n"
                f"Fallback attempts: {len(result.fallback_attempts)}\n"
                f"Total time: {result.elapsed_s:.3f}s",
                title="[bold green]Post-Action Summary",
            )
        )


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/]")
        console.print(
            '  python -m spiral_03_orchestration.orchestrator --profile STANDARD "your question"'
        )
        sys.exit(1)

    # Parse args
    profile_name = "STANDARD"
    query_parts: list[str] = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--profile" and i + 1 < len(args):
            profile_name = args[i + 1]
            i += 2
        else:
            query_parts.append(args[i])
            i += 1

    if not query_parts:
        console.print("[red]Error: provide a question[/]")
        sys.exit(1)

    query = " ".join(query_parts)

    orchestrator = Orchestrator(profile=profile_name)
    orchestrator.ask(query)


if __name__ == "__main__":  # pragma: no cover
    main()
