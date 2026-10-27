"""Dashboard — CLI reporting for cost monitoring.

Analogy for a C dev:
    Like `top` or `htop` but for your LLM spend.
    Shows where your tokens are going, which models cost the most,
    and how much you saved by routing smart.
"""

from __future__ import annotations

from spiral_05_economics.budget_tracker import BudgetTracker
from spiral_05_economics.config import DAILY_BUDGET, MODEL_PRICING, MONTHLY_BUDGET


class Dashboard:
    """CLI dashboard for cost monitoring and reporting."""

    def __init__(
        self,
        tracker: BudgetTracker,
        daily_budget: float = DAILY_BUDGET,
        monthly_budget: float = MONTHLY_BUDGET,
        metrics: object | None = None,
    ) -> None:
        self._tracker = tracker
        self._daily_budget = daily_budget
        self._monthly_budget = monthly_budget
        self._metrics = metrics

    def daily_report(self) -> str:
        """Generate a daily cost report string."""
        summary = self._tracker.daily_summary()
        pct = (summary.total_cost / self._daily_budget * 100) if self._daily_budget > 0 else 0
        lines = [
            f"Today ({summary.date}):",
            f"  Cost: ${summary.total_cost:.4f} / ${self._daily_budget:.2f} ({pct:.1f}%)",
            f"  Requests: {summary.total_requests}",
            f"  Tokens: {summary.total_input_tokens} in / {summary.total_output_tokens} out",
        ]
        if summary.by_model:
            lines.append("  By model:")
            for model, cost in sorted(summary.by_model.items(), key=lambda x: -x[1]):
                lines.append(f"    {model}: ${cost:.4f}")
        return "\n".join(lines)

    def monthly_report(self) -> str:
        """Generate a monthly cost report string."""
        summary = self._tracker.monthly_summary()
        pct = (summary.total_cost / self._monthly_budget * 100) if self._monthly_budget > 0 else 0
        lines = [
            f"Month ({summary.date}):",
            f"  Cost: ${summary.total_cost:.4f} / ${self._monthly_budget:.2f} ({pct:.1f}%)",
            f"  Requests: {summary.total_requests}",
            f"  Tokens: {summary.total_input_tokens} in / {summary.total_output_tokens} out",
        ]
        if summary.by_model:
            lines.append("  By model:")
            for model, cost in sorted(summary.by_model.items(), key=lambda x: -x[1]):
                lines.append(f"    {model}: ${cost:.4f}")
        return "\n".join(lines)

    def top_models(self) -> list[dict[str, object]]:
        """Return models ranked by total cost (daily)."""
        summary = self._tracker.daily_summary()
        if not summary.by_model:
            return []
        return sorted(
            [{"model": m, "cost": c} for m, c in summary.by_model.items()],
            key=lambda x: -x["cost"],  # type: ignore[operator]
        )

    def estimate_savings(self) -> dict[str, float]:
        """Estimate how much was saved by using free/cheap models.

        Compares actual cost vs hypothetical cost if everything used claude-sonnet.
        """
        summary = self._tracker.daily_summary()
        sonnet_input_price, sonnet_output_price = MODEL_PRICING.get("claude-sonnet", (3.00, 15.00))
        hypothetical_cost = (summary.total_input_tokens / 1_000_000) * sonnet_input_price + (
            summary.total_output_tokens / 1_000_000
        ) * sonnet_output_price
        return {
            "actual_cost": summary.total_cost,
            "hypothetical_sonnet_cost": hypothetical_cost,
            "saved_vs_sonnet": max(0.0, hypothetical_cost - summary.total_cost),
        }

    def display(self) -> None:
        """Print the dashboard to stdout using rich."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Cost Dashboard")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        s = self.summary()
        table.add_row("Daily Cost", f"${s['daily_cost']:.4f}")
        table.add_row("Daily Budget", f"${s['daily_budget']:.2f}")
        table.add_row("Monthly Cost", f"${s['monthly_cost']:.4f}")
        table.add_row("Monthly Budget", f"${s['monthly_budget']:.2f}")
        table.add_row("Daily Requests", str(s["daily_requests"]))
        table.add_row("Monthly Requests", str(s["monthly_requests"]))
        table.add_row("Alert Level", str(s["alert_level"]))
        console.print(table)

    def summary(self) -> dict[str, object]:
        """Return a summary dict suitable for CLI or API output."""
        daily = self._tracker.daily_summary()
        monthly = self._tracker.monthly_summary()
        status = self._tracker.check_budget()
        result: dict[str, object] = {
            "daily_cost": daily.total_cost,
            "daily_budget": self._daily_budget,
            "daily_requests": daily.total_requests,
            "monthly_cost": monthly.total_cost,
            "monthly_budget": self._monthly_budget,
            "monthly_requests": monthly.total_requests,
            "alert_level": status.alert_level,
        }
        # Include live metrics if wired
        if self._metrics is not None and hasattr(self._metrics, "summary"):
            result["pipeline_metrics"] = self._metrics.summary()
        return result
