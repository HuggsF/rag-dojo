"""L0 Memory — Persistent action journal with temporal comparison.

Analogy for a C dev:
- MemoryStore = append-only log file, but with indexed random access (like SQLite WAL)
- save_action() = write() to the journal — timestamp, bot, command, status, output
- query_history() = SELECT with filters — like grepping /var/log/ but O(log n)
- stats() = aggregated counters — like wc -l piped through awk
- detect_regression() = diff between last two runs — OK→fail = regression
- detect_streak() = count consecutive same-status — like a watchdog timer
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spiral_04_cognitive.config import HISTORY_QUERY_LIMIT, MAX_OUTPUT_LENGTH, MEMORY_DB_PATH

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    bot_name TEXT NOT NULL,
    command TEXT NOT NULL,
    status TEXT NOT NULL,
    output TEXT,
    metadata TEXT
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_action_log_bot_ts
ON action_log (bot_name, timestamp DESC)
"""


@dataclass
class ActionRecord:
    """Single action record from the memory store.

    Like a struct in /var/log/audit.log — one line per event.
    """

    id: int
    timestamp: str
    bot_name: str
    command: str
    status: str
    output: str | None = None
    metadata: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for display/logging."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "bot_name": self.bot_name,
            "command": self.command,
            "status": self.status,
            "output": self.output[:200] if self.output else None,
            "metadata": self.metadata,
        }


class MemoryStore:
    """SQLite-backed persistent memory for bot actions.

    Thread-safe via SQLite's built-in locking. One DB per deployment.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or MEMORY_DB_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy-load SQLite connection."""
        if self._conn is None:
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)
            self._conn.commit()
        return self._conn

    def save_action(
        self,
        bot_name: str,
        command: str,
        status: str,
        output: str | None = None,
        metadata: str | None = None,
    ) -> int:
        """Persist an action record. Returns the row ID.

        Output is truncated to MAX_OUTPUT_LENGTH to prevent DB bloat.
        """
        truncated = output[:MAX_OUTPUT_LENGTH] if output else None
        ts = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO action_log (timestamp, bot_name, command, status, output, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, bot_name, command, status, truncated, metadata),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def query_history(
        self,
        bot_name: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[ActionRecord]:
        """Query action history with optional filters.

        Returns most recent first (DESC by timestamp).
        """
        query = "SELECT * FROM action_log WHERE 1=1"
        params: list[Any] = []

        if bot_name is not None:
            query += " AND bot_name = ?"
            params.append(bot_name)
        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit or HISTORY_QUERY_LIMIT)

        rows = self.conn.execute(query, params).fetchall()
        return [
            ActionRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                bot_name=row["bot_name"],
                command=row["command"],
                status=row["status"],
                output=row["output"],
                metadata=row["metadata"],
            )
            for row in rows
        ]

    def get_last_action(self, bot_name: str) -> ActionRecord | None:
        """Get the most recent action for a bot."""
        records = self.query_history(bot_name=bot_name, limit=1)
        return records[0] if records else None

    def detect_regression(self, bot_name: str) -> bool:
        """Detect regression: previous was ok, current is fail.

        Like comparing two snapshots — if the diff shows degradation, flag it.
        """
        records = self.query_history(bot_name=bot_name, limit=2)
        if len(records) < 2:
            return False
        # records[0] is newest, records[1] is previous
        return records[0].status == "fail" and records[1].status == "ok"

    def detect_recovery(self, bot_name: str) -> bool:
        """Detect recovery: previous was fail, current is ok."""
        records = self.query_history(bot_name=bot_name, limit=2)
        if len(records) < 2:
            return False
        return records[0].status == "ok" and records[1].status == "fail"

    def detect_streak(self, bot_name: str, status: str) -> int:
        """Count consecutive records with the same status (most recent first).

        Like a watchdog counter — how many times in a row has this status occurred?
        """
        records = self.query_history(bot_name=bot_name, limit=HISTORY_QUERY_LIMIT)
        count = 0
        for record in records:
            if record.status == status:
                count += 1
            else:
                break
        return count

    def stats(self, bot_name: str | None = None) -> dict[str, Any]:
        """Aggregated statistics for a bot (or all bots).

        Returns total actions, status distribution, and last action timestamp.
        """
        where = ""
        params: list[Any] = []
        if bot_name:
            where = "WHERE bot_name = ?"
            params.append(bot_name)

        # Total count
        total = self.conn.execute(
            f"SELECT COUNT(*) as cnt FROM action_log {where}",  # nosec B608
            params,
        ).fetchone()["cnt"]

        # Status distribution
        rows = self.conn.execute(
            f"SELECT status, COUNT(*) as cnt FROM action_log {where} GROUP BY status",  # nosec B608
            params,
        ).fetchall()
        by_status = {row["status"]: row["cnt"] for row in rows}

        # Last action
        last_row = self.conn.execute(
            f"SELECT MAX(timestamp) as last_ts FROM action_log {where}",  # nosec B608
            params,
        ).fetchone()

        return {
            "total_actions": total,
            "by_status": by_status,
            "last_action": last_row["last_ts"] if last_row else None,
        }

    def consolidate(self, bot_name: str, min_group_size: int = 3) -> list[dict[str, Any]]:
        """Consolidate repetitive action records into patterns.

        Groups similar (bot_name, command, status) tuples and replaces
        N individual records with 1 summary record. Returns the patterns found.

        Only consolidates groups with >= min_group_size records.
        This is memory compaction — reduces DB bloat while preserving insights.
        """
        # Find groups with enough records to consolidate
        rows = self.conn.execute(
            "SELECT command, status, COUNT(*) as cnt, "
            "MIN(timestamp) as first_ts, MAX(timestamp) as last_ts "
            "FROM action_log WHERE bot_name = ? "
            "GROUP BY command, status HAVING cnt >= ?",
            (bot_name, min_group_size),
        ).fetchall()

        patterns: list[dict[str, Any]] = []

        for row in rows:
            command = row["command"]
            status = row["status"]
            count = row["cnt"]
            first_ts = row["first_ts"]
            last_ts = row["last_ts"]

            # Delete individual records (keep most recent one)
            self.conn.execute(
                "DELETE FROM action_log WHERE bot_name = ? AND command = ? "
                "AND status = ? AND id NOT IN ("
                "  SELECT id FROM action_log WHERE bot_name = ? AND command = ? "
                "  AND status = ? ORDER BY id DESC LIMIT 1"
                ")",
                (bot_name, command, status, bot_name, command, status),
            )

            # Update the surviving record with consolidated metadata
            self.conn.execute(
                "UPDATE action_log SET metadata = ? "
                "WHERE id = ("
                "  SELECT id FROM action_log WHERE bot_name = ? AND command = ? "
                "  AND status = ? ORDER BY id DESC LIMIT 1"
                ")",
                (
                    f"consolidated:{count} records from {first_ts} to {last_ts}",
                    bot_name,
                    command,
                    status,
                ),
            )

            patterns.append(
                {
                    "bot_name": bot_name,
                    "command": command,
                    "status": status,
                    "consolidated_count": count,
                    "first_seen": first_ts,
                    "last_seen": last_ts,
                }
            )

        self.conn.commit()
        return patterns

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
