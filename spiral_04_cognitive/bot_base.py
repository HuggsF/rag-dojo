"""Bot Base — Template for cognitive-enabled bots.

Analogy for a C dev:
- BotBase = abstract base class (like a vtable) — defines the interface
- handle() = pure virtual — each bot implements its own check logic
- run() = the main loop: handle() → cognitive.enrich() → display
- Every bot gets cognitive enrichment for free by inheriting BotBase.
- Think of it as a daemon template: each bot is a specialized daemon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spiral_04_cognitive.cognitive_layer import CognitiveLayer, EnrichedAnalysis

console = Console()


@dataclass
class BotOutput:
    """Raw output from a bot's handle() method.

    Like the return struct from a daemon's check function.
    """

    output: str
    metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "output": self.output[:200],
            "metrics": self.metrics,
        }


class BotBase(ABC):
    """Abstract base for cognitive-enabled bots.

    Subclass and implement handle() — cognitive enrichment is automatic.
    """

    def __init__(
        self,
        name: str,
        cognitive: CognitiveLayer | None = None,
    ):
        self.name = name
        self._cognitive = cognitive

    @property
    def cognitive(self) -> CognitiveLayer:
        """Lazy-load cognitive layer."""
        if self._cognitive is None:
            self._cognitive = CognitiveLayer()
        return self._cognitive

    @abstractmethod
    def handle(self, command: str) -> BotOutput:
        """Execute the bot's core logic for a given command.

        Must return BotOutput with raw output string and optional metrics.
        """

    def run(self, command: str, show_details: bool = True) -> EnrichedAnalysis:
        """Execute handle() + cognitive enrichment pipeline.

        This is the main entry point for running a bot action.
        """
        # Step 1: Execute bot logic
        bot_output = self.handle(command)

        # Step 2: Cognitive enrichment (L0 + L1 + L2)
        analysis = self.cognitive.enrich(
            bot_name=self.name,
            command=command,
            raw_output=bot_output.output,
            metrics=bot_output.metrics,
        )

        # Step 3: Display results
        if show_details:
            self._display_analysis(command, analysis)

        return analysis

    def _display_analysis(self, command: str, analysis: EnrichedAnalysis) -> None:
        """Rich console output for the cognitive analysis."""
        # Status color
        status_color = {
            "ok": "green",
            "fail": "red",
            "partial": "yellow",
        }.get(analysis.status, "white")

        # Header panel
        console.print(
            Panel(
                f"[bold]Bot:[/] {self.name}\n"
                f"[bold]Command:[/] {command}\n"
                f"[bold]Status:[/] [{status_color}]{analysis.status}[/{status_color}]",
                title=f"[bold blue]Cognitive Analysis — {self.name}",
            )
        )

        # Alerts
        if analysis.alerts:
            alert_table = Table(title="Alerts", show_lines=True)
            alert_table.add_column("Alert", style="red bold")
            for alert in analysis.alerts:
                alert_table.add_row(alert)
            console.print(alert_table)

        # Pattern matches
        if analysis.pattern_matches:
            pat_table = Table(title="Pattern Matches")
            pat_table.add_column("Pattern", style="cyan")
            pat_table.add_column("Severity", style="yellow")
            pat_table.add_column("Matched", style="dim")
            for m in analysis.pattern_matches:
                pat_table.add_row(m.pattern.name, m.pattern.severity, m.matched_text[:60])
            console.print(pat_table)

        # Threshold violations
        if analysis.threshold_violations:
            thresh_table = Table(title="Threshold Violations")
            thresh_table.add_column("Metric", style="cyan")
            thresh_table.add_column("Level", style="red")
            thresh_table.add_column("Message", style="dim")
            for v in analysis.threshold_violations:
                thresh_table.add_row(v.metric_name, v.level, v.message)
            console.print(thresh_table)

        # Recommendation
        if analysis.recommendation:
            console.print(
                Panel(
                    f"[bold yellow]ALERT:[/] {analysis.recommendation}",
                    title="[bold red]Recommendation",
                )
            )

        # Temporal context
        if analysis.is_regression:
            console.print("[bold red]⚠ REGRESSION detected: was ok, now fail[/]")
        if analysis.is_recovery:
            console.print("[bold green]✓ RECOVERY detected: was fail, now ok[/]")
        if analysis.fail_streak > 0:
            console.print(f"[dim]Fail streak: {analysis.fail_streak}[/]")
