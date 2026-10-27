"""Circuit breaker — hard stop when budget is exceeded.

Analogy for a C dev:
    Like a stack guard page — if you hit it, execution STOPS.
    No "just one more malloc". Budget exceeded = circuit open = no LLM calls.
    This is Law 8 of AI-OS: financial limits are HARD STOP, no exceptions.
"""

from __future__ import annotations

from spiral_05_economics.budget_tracker import BudgetTracker
from spiral_05_economics.exceptions import BudgetExceededError

# Re-export for backwards compatibility — existing imports still work.
__all__ = ["BudgetExceededError", "CircuitBreaker"]


class CircuitBreaker:
    """Hard stop on LLM calls when budget is exceeded."""

    def __init__(self, tracker: BudgetTracker) -> None:
        self._tracker = tracker

    def is_open(self) -> bool:
        """Check if circuit is open (budget exceeded).

        Returns True if budget is exceeded and no more calls should be made.
        """
        budget_status = self._tracker.check_budget()
        return not budget_status.daily_ok or not budget_status.monthly_ok

    def check_or_raise(self) -> None:
        """Check budget and raise BudgetExceededError if exceeded."""
        budget_status = self._tracker.check_budget()

        if not budget_status.daily_ok:
            raise BudgetExceededError(
                f"Daily budget exceeded: ${budget_status.daily_spent:.4f} / "
                f"${budget_status.daily_budget:.2f}"
            )

        if not budget_status.monthly_ok:
            raise BudgetExceededError(
                f"Monthly budget exceeded: ${budget_status.monthly_spent:.4f} / "
                f"${budget_status.monthly_budget:.2f}"
            )

    def remaining_daily(self) -> float:
        """Remaining daily budget in USD."""
        budget_status = self._tracker.check_budget()
        return max(0.0, budget_status.daily_budget - budget_status.daily_spent)

    def remaining_monthly(self) -> float:
        """Remaining monthly budget in USD."""
        budget_status = self._tracker.check_budget()
        return max(0.0, budget_status.monthly_budget - budget_status.monthly_spent)

    def status(self) -> dict[str, object]:
        """Return a status report for the circuit breaker."""
        budget_status = self._tracker.check_budget()
        return {
            "open": self.is_open(),
            "daily_remaining": self.remaining_daily(),
            "monthly_remaining": self.remaining_monthly(),
            "alert_level": budget_status.alert_level,
        }
