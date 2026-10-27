"""Sample Bot — docs_bot monitoring index freshness.

Analogy for a C dev:
- docs_bot = a cron job that checks if your man pages are up to date
- check command = stat() the files, compare mtime with last index time
- Simulates variable freshness to demonstrate cognitive layer alerts
- On 3+ consecutive failures: ALERT with recommendation to re-index

Usage:
    python -m spiral_04_cognitive.sample_bot check
    python -m spiral_04_cognitive.sample_bot stats
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spiral_04_cognitive.bot_base import BotBase, BotOutput
from spiral_04_cognitive.cognitive_layer import CognitiveLayer

console = Console()

# Simulated doc states for demonstration
_SIMULATED_STATES = [
    {"fresh": True, "hours_since_index": 2, "doc_count": 150, "hash_match": True},
    {"fresh": True, "hours_since_index": 12, "doc_count": 150, "hash_match": True},
    {"fresh": False, "hours_since_index": 96, "doc_count": 148, "hash_match": False},
    {"fresh": False, "hours_since_index": 120, "doc_count": 145, "hash_match": False},
    {"fresh": False, "hours_since_index": 168, "doc_count": 140, "hash_match": False},
]


class DocsBot(BotBase):
    """Bot that monitors index freshness for ingested documentation.

    In production, this would check ChromaDB metadata vs filesystem state.
    For training, it uses simulated states to demonstrate the cognitive layer.
    """

    def __init__(
        self,
        cognitive: CognitiveLayer | None = None,
        docs_dir: str | None = None,
        simulation_mode: bool = True,
    ):
        super().__init__(name="docs_bot", cognitive=cognitive)
        self._docs_dir = docs_dir or "./data/fastapi_docs/"
        self._simulation_mode = simulation_mode
        self._sim_index = 0

    def handle(self, command: str) -> BotOutput:
        """Execute docs_bot commands."""
        if command == "check":
            return self._check_freshness()
        if command == "stats":
            return self._get_stats()
        return BotOutput(output=f"Unknown command: {command}")

    def _check_freshness(self) -> BotOutput:
        """Check if indexed docs are fresh.

        In simulation mode: cycles through predefined states.
        In production mode: compares file hashes with ChromaDB metadata.
        """
        if self._simulation_mode:
            return self._check_freshness_simulated()
        return self._check_freshness_real()

    def _check_freshness_simulated(self) -> BotOutput:
        """Simulated freshness check — cycles through states."""
        state = _SIMULATED_STATES[self._sim_index % len(_SIMULATED_STATES)]
        self._sim_index += 1

        if state["fresh"]:
            output = (
                f"Docs fresh: {state['doc_count']} docs indexed, "
                f"last index {state['hours_since_index']}h ago. "
                f"All hashes match. Status: OK, healthy."
            )
        else:
            output = (
                f"Docs stale: {state['doc_count']} docs found but "
                f"last update {state['hours_since_index']} hours ago. "
                f"Hash mismatch detected — docs are outdated and not fresh."
            )

        metrics = {
            "index_freshness_hours": float(state["hours_since_index"]),
        }

        return BotOutput(output=output, metrics=metrics)

    def _check_freshness_real(self) -> BotOutput:
        """Real freshness check against filesystem.

        Checks if doc files exist and computes a hash fingerprint.
        """
        docs_path = Path(self._docs_dir)
        if not docs_path.exists():
            return BotOutput(
                output=f"Docs directory not found: {self._docs_dir}. "
                "Stale — last index unknown days ago.",
                metrics={"index_freshness_hours": 9999.0},
            )

        # Count files and compute aggregate hash
        files = sorted(docs_path.rglob("*.md"))
        if not files:
            files = sorted(docs_path.rglob("*.html"))
        if not files:
            files = sorted(docs_path.rglob("*.*"))

        doc_count = len(files)
        if doc_count == 0:
            return BotOutput(
                output="No docs found in directory. Stale — not fresh.",
                metrics={"index_freshness_hours": 9999.0},
            )

        # Check most recent modification
        newest_mtime = max(f.stat().st_mtime for f in files)
        hours_since = (time.time() - newest_mtime) / 3600

        # Compute simple hash of file listing
        hasher = hashlib.md5(usedforsecurity=False)  # nosec B324
        for f in files[:50]:  # Sample first 50 for speed
            hasher.update(f.name.encode())
        fingerprint = hasher.hexdigest()[:8]

        if hours_since < 48:
            output = (
                f"Docs fresh: {doc_count} docs, last modified {hours_since:.1f}h ago. "
                f"Fingerprint: {fingerprint}. Status: OK, healthy."
            )
        else:
            output = (
                f"Docs stale: {doc_count} docs but last update "
                f"{hours_since:.1f} hours ago. "
                f"Fingerprint: {fingerprint}. Outdated — not fresh."
            )

        return BotOutput(
            output=output,
            metrics={"index_freshness_hours": hours_since},
        )

    def _get_stats(self) -> BotOutput:
        """Get historical stats from cognitive memory."""
        stats = self.cognitive.memory.stats(bot_name=self.name)
        output = (
            f"Total actions: {stats['total_actions']}\n"
            f"By status: {stats['by_status']}\n"
            f"Last action: {stats['last_action']}"
        )
        return BotOutput(output=output)


def _display_stats(stats: dict[str, Any]) -> None:
    """Display bot stats in a Rich table."""
    table = Table(title="docs_bot Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total actions", str(stats.get("total_actions", 0)))
    for status, count in stats.get("by_status", {}).items():
        table.add_row(f"  {status}", str(count))
    table.add_row("Last action", str(stats.get("last_action", "never")))
    console.print(table)


def main() -> None:
    """CLI entry point for docs_bot."""
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/]")
        console.print("  python -m spiral_04_cognitive.sample_bot check   # Check index freshness")
        console.print("  python -m spiral_04_cognitive.sample_bot stats   # Show history stats")
        console.print("  python -m spiral_04_cognitive.sample_bot demo    # Run 5x demo cycle")
        sys.exit(1)

    command = sys.argv[1]

    if command == "demo":
        # Run the full demo: 5 checks cycling through simulated states
        bot = DocsBot(simulation_mode=True)
        console.print(
            Panel(
                "[bold]Running docs_bot demo — 5 cycles with variable freshness[/]",
                title="[bold blue]Spiral 04 — Cognitive Demo",
            )
        )
        for i in range(5):
            console.print(f"\n[bold]═══ Run {i + 1}/5 ═══[/]")
            bot.run("check", show_details=True)
            console.print()
        return

    if command in ("check", "stats"):
        # Determine mode from args
        simulation = "--real" not in sys.argv
        bot = DocsBot(simulation_mode=simulation)

        if command == "stats":
            stats = bot.cognitive.memory.stats(bot_name=bot.name)
            _display_stats(stats)
        else:
            bot.run(command, show_details=True)
    else:
        console.print(f"[red]Unknown command: {command}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
