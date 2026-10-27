"""Budget tracker — SQLite-backed daily/monthly cost tracking.

Analogy for a C dev:
    Like tracking heap allocations with a custom allocator —
    every malloc (LLM call) gets logged with size (tokens) and cost.
    Daily/monthly summaries = memory profiler reports.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from spiral_05_economics.config import (
    BUDGET_ALERT_PCT,
    BUDGET_CRITICAL_PCT,
    BUDGET_DB_PATH,
    DAILY_BUDGET,
    MONTHLY_BUDGET,
)


@dataclass
class DailySummary:
    """Aggregated cost data for a single day."""

    date: str
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    by_model: dict[str, float] = field(default_factory=dict)


@dataclass
class BudgetStatus:
    """Current budget status."""

    daily_spent: float
    daily_budget: float
    daily_ok: bool
    monthly_spent: float
    monthly_budget: float
    monthly_ok: bool
    alert_level: str  # "normal", "warning", "critical", "exceeded"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cost_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now')),
    date        TEXT    NOT NULL DEFAULT (date('now')),
    model       TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost        REAL    NOT NULL,
    query       TEXT
);

CREATE INDEX IF NOT EXISTS idx_cost_log_date ON cost_log(date);
CREATE INDEX IF NOT EXISTS idx_cost_log_model ON cost_log(model);
"""


class BudgetTracker:
    """Tracks LLM costs in SQLite with daily/monthly budgets."""

    def __init__(
        self,
        db_path: str | None = None,
        daily_budget: float = DAILY_BUDGET,
        monthly_budget: float = MONTHLY_BUDGET,
        alert_pct: float = BUDGET_ALERT_PCT,
        critical_pct: float = BUDGET_CRITICAL_PCT,
    ) -> None:
        self._db_path = db_path or BUDGET_DB_PATH
        self._daily_budget = daily_budget
        self._monthly_budget = monthly_budget
        self._alert_pct = alert_pct
        self._critical_pct = critical_pct
        self._conn: sqlite3.Connection | None = None

        # Ensure data directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-init SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def record_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        query: str | None = None,
    ) -> int:
        """Record a single LLM call cost. Returns the row ID."""
        cursor = self.conn.execute(
            "INSERT INTO cost_log (model, input_tokens, output_tokens, cost, query) "
            "VALUES (?, ?, ?, ?, ?)",
            (model, input_tokens, output_tokens, cost, query),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def daily_summary(self, target_date: str | None = None) -> DailySummary:
        """Get cost summary for a specific day (default: today)."""
        target = target_date or date.today().isoformat()

        row = self.conn.execute(
            "SELECT "
            "  COALESCE(SUM(cost), 0) as total_cost, "
            "  COUNT(*) as total_requests, "
            "  COALESCE(SUM(input_tokens), 0) as total_input, "
            "  COALESCE(SUM(output_tokens), 0) as total_output "
            "FROM cost_log WHERE date = ?",
            (target,),
        ).fetchone()

        # Per-model breakdown
        model_rows = self.conn.execute(
            "SELECT model, SUM(cost) as model_cost FROM cost_log WHERE date = ? GROUP BY model",
            (target,),
        ).fetchall()

        by_model = {r["model"]: r["model_cost"] for r in model_rows}

        return DailySummary(
            date=target,
            total_cost=row["total_cost"],
            total_requests=row["total_requests"],
            total_input_tokens=row["total_input"],
            total_output_tokens=row["total_output"],
            by_model=by_model,
        )

    def monthly_summary(self, year: int | None = None, month: int | None = None) -> DailySummary:
        """Get cost summary for a month (default: current month)."""
        now = datetime.now()
        y = year or now.year
        m = month or now.month
        month_prefix = f"{y:04d}-{m:02d}"

        row = self.conn.execute(
            "SELECT "
            "  COALESCE(SUM(cost), 0) as total_cost, "
            "  COUNT(*) as total_requests, "
            "  COALESCE(SUM(input_tokens), 0) as total_input, "
            "  COALESCE(SUM(output_tokens), 0) as total_output "
            "FROM cost_log WHERE date LIKE ?",
            (f"{month_prefix}%",),
        ).fetchone()

        model_rows = self.conn.execute(
            "SELECT model, SUM(cost) as model_cost FROM cost_log WHERE date LIKE ? GROUP BY model",
            (f"{month_prefix}%",),
        ).fetchall()

        by_model = {r["model"]: r["model_cost"] for r in model_rows}

        return DailySummary(
            date=month_prefix,
            total_cost=row["total_cost"],
            total_requests=row["total_requests"],
            total_input_tokens=row["total_input"],
            total_output_tokens=row["total_output"],
            by_model=by_model,
        )

    def check_budget(self) -> BudgetStatus:
        """Check current budget status against limits."""
        daily = self.daily_summary()
        monthly = self.monthly_summary()

        daily_ok = daily.total_cost <= self._daily_budget
        monthly_ok = monthly.total_cost <= self._monthly_budget

        # Determine alert level
        daily_pct = (daily.total_cost / self._daily_budget * 100) if self._daily_budget > 0 else 0
        monthly_pct = (
            (monthly.total_cost / self._monthly_budget * 100) if self._monthly_budget > 0 else 0
        )
        max_pct = max(daily_pct, monthly_pct)

        if not daily_ok or not monthly_ok:
            alert_level = "exceeded"
        elif max_pct >= self._critical_pct:
            alert_level = "critical"
        elif max_pct >= self._alert_pct:
            alert_level = "warning"
        else:
            alert_level = "normal"

        return BudgetStatus(
            daily_spent=daily.total_cost,
            daily_budget=self._daily_budget,
            daily_ok=daily_ok,
            monthly_spent=monthly.total_cost,
            monthly_budget=self._monthly_budget,
            monthly_ok=monthly_ok,
            alert_level=alert_level,
        )
