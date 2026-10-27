"""Tests for spiral_04_cognitive.memory_store — L0 Memory."""

from spiral_04_cognitive.memory_store import ActionRecord, MemoryStore


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization."""

    def test_creates_db_on_first_access(self, tmp_db):
        """DB file should be created on first connection."""
        store = MemoryStore(db_path=tmp_db)
        _ = store.conn
        from pathlib import Path

        assert Path(tmp_db).exists()
        store.close()

    def test_close_and_reopen(self, tmp_db):
        """Should survive close/reopen cycle."""
        store = MemoryStore(db_path=tmp_db)
        store.save_action("bot1", "check", "ok", "all good")
        store.close()

        store2 = MemoryStore(db_path=tmp_db)
        records = store2.query_history(bot_name="bot1")
        assert len(records) == 1
        store2.close()


class TestSaveAction:
    """Tests for save_action."""

    def test_save_returns_row_id(self, memory_store):
        """save_action should return a positive row ID."""
        row_id = memory_store.save_action("bot1", "check", "ok", "output text")
        assert row_id > 0

    def test_save_multiple(self, memory_store):
        """Multiple saves should produce incrementing IDs."""
        id1 = memory_store.save_action("bot1", "check", "ok")
        id2 = memory_store.save_action("bot1", "check", "fail")
        assert id2 > id1

    def test_save_truncates_long_output(self, memory_store):
        """Output longer than MAX_OUTPUT_LENGTH should be truncated."""
        long_output = "x" * 5000
        memory_store.save_action("bot1", "check", "ok", long_output)
        records = memory_store.query_history(bot_name="bot1")
        assert len(records[0].output) <= 2000

    def test_save_with_none_output(self, memory_store):
        """None output should be stored as None."""
        memory_store.save_action("bot1", "check", "ok", None)
        records = memory_store.query_history(bot_name="bot1")
        assert records[0].output is None

    def test_save_with_metadata(self, memory_store):
        """Metadata should be stored and retrievable."""
        memory_store.save_action("bot1", "check", "ok", "out", '{"key": "val"}')
        records = memory_store.query_history(bot_name="bot1")
        assert records[0].metadata == '{"key": "val"}'


class TestQueryHistory:
    """Tests for query_history."""

    def test_empty_history(self, memory_store):
        """Empty DB should return empty list."""
        records = memory_store.query_history(bot_name="bot1")
        assert records == []

    def test_filter_by_bot_name(self, memory_store):
        """Should filter by bot_name."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot2", "check", "fail")
        records = memory_store.query_history(bot_name="bot1")
        assert len(records) == 1
        assert records[0].bot_name == "bot1"

    def test_filter_by_status(self, memory_store):
        """Should filter by status."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "ok")
        records = memory_store.query_history(bot_name="bot1", status="fail")
        assert len(records) == 1

    def test_limit(self, memory_store):
        """Should respect limit parameter."""
        for i in range(10):
            memory_store.save_action("bot1", "check", "ok")
        records = memory_store.query_history(bot_name="bot1", limit=3)
        assert len(records) == 3

    def test_order_is_newest_first(self, memory_store):
        """Results should be ordered newest first."""
        memory_store.save_action("bot1", "cmd1", "ok")
        memory_store.save_action("bot1", "cmd2", "fail")
        records = memory_store.query_history(bot_name="bot1")
        assert records[0].command == "cmd2"
        assert records[1].command == "cmd1"


class TestGetLastAction:
    """Tests for get_last_action."""

    def test_no_history(self, memory_store):
        """No history should return None."""
        assert memory_store.get_last_action("bot1") is None

    def test_returns_latest(self, memory_store):
        """Should return the most recent action."""
        memory_store.save_action("bot1", "cmd1", "ok")
        memory_store.save_action("bot1", "cmd2", "fail")
        last = memory_store.get_last_action("bot1")
        assert last is not None
        assert last.command == "cmd2"


class TestDetectRegression:
    """Tests for regression detection."""

    def test_no_history(self, memory_store):
        """No history → no regression."""
        assert memory_store.detect_regression("bot1") is False

    def test_single_record(self, memory_store):
        """Single record → no regression."""
        memory_store.save_action("bot1", "check", "fail")
        assert memory_store.detect_regression("bot1") is False

    def test_ok_to_fail(self, memory_store):
        """ok → fail = regression."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "fail")
        assert memory_store.detect_regression("bot1") is True

    def test_fail_to_fail(self, memory_store):
        """fail → fail = NOT regression (already broken)."""
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "fail")
        assert memory_store.detect_regression("bot1") is False

    def test_fail_to_ok(self, memory_store):
        """fail → ok = NOT regression (recovery)."""
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "ok")
        assert memory_store.detect_regression("bot1") is False


class TestDetectRecovery:
    """Tests for recovery detection."""

    def test_fail_to_ok(self, memory_store):
        """fail → ok = recovery."""
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "ok")
        assert memory_store.detect_recovery("bot1") is True

    def test_ok_to_ok(self, memory_store):
        """ok → ok = NOT recovery."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "ok")
        assert memory_store.detect_recovery("bot1") is False


class TestDetectStreak:
    """Tests for streak detection."""

    def test_empty_streak(self, memory_store):
        """No history → streak = 0."""
        assert memory_store.detect_streak("bot1", "fail") == 0

    def test_single_fail(self, memory_store):
        """Single fail → streak = 1."""
        memory_store.save_action("bot1", "check", "fail")
        assert memory_store.detect_streak("bot1", "fail") == 1

    def test_consecutive_fails(self, memory_store):
        """3 consecutive fails → streak = 3."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "fail")
        assert memory_store.detect_streak("bot1", "fail") == 3

    def test_broken_streak(self, memory_store):
        """ok between fails breaks the streak."""
        memory_store.save_action("bot1", "check", "fail")
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "fail")
        assert memory_store.detect_streak("bot1", "fail") == 1


class TestStats:
    """Tests for stats aggregation."""

    def test_empty_stats(self, memory_store):
        """Empty DB → total_actions = 0."""
        stats = memory_store.stats(bot_name="bot1")
        assert stats["total_actions"] == 0
        assert stats["by_status"] == {}

    def test_stats_with_data(self, memory_store):
        """Stats should reflect actual data."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot1", "check", "fail")
        stats = memory_store.stats(bot_name="bot1")
        assert stats["total_actions"] == 3
        assert stats["by_status"]["ok"] == 2
        assert stats["by_status"]["fail"] == 1
        assert stats["last_action"] is not None

    def test_stats_all_bots(self, memory_store):
        """Stats without bot_name should include all bots."""
        memory_store.save_action("bot1", "check", "ok")
        memory_store.save_action("bot2", "check", "fail")
        stats = memory_store.stats()
        assert stats["total_actions"] == 2


class TestActionRecord:
    """Tests for ActionRecord dataclass."""

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        record = ActionRecord(
            id=1,
            timestamp="2026-03-22T10:00:00+00:00",
            bot_name="bot1",
            command="check",
            status="ok",
            output="all good",
        )
        d = record.to_dict()
        assert d["id"] == 1
        assert d["bot_name"] == "bot1"
        assert d["status"] == "ok"

    def test_to_dict_truncates_output(self):
        """to_dict should truncate long output."""
        record = ActionRecord(
            id=1,
            timestamp="2026-03-22T10:00:00+00:00",
            bot_name="bot1",
            command="check",
            status="ok",
            output="x" * 500,
        )
        d = record.to_dict()
        assert len(d["output"]) == 200


class TestConsolidate:
    """Tests for memory consolidation — grouping repetitive records."""

    def test_consolidates_large_group(self, memory_store):
        """Groups with >= min_group_size are consolidated."""
        for _ in range(5):
            memory_store.save_action("bot1", "query", "ok", output="result")
        patterns = memory_store.consolidate("bot1", min_group_size=3)
        assert len(patterns) == 1
        assert patterns[0]["consolidated_count"] == 5
        # After consolidation, only 1 record remains
        records = memory_store.query_history(bot_name="bot1")
        assert len(records) == 1

    def test_skips_small_group(self, memory_store):
        """Groups with < min_group_size are not consolidated."""
        memory_store.save_action("bot1", "query", "ok")
        memory_store.save_action("bot1", "query", "ok")
        patterns = memory_store.consolidate("bot1", min_group_size=3)
        assert len(patterns) == 0
        records = memory_store.query_history(bot_name="bot1")
        assert len(records) == 2

    def test_consolidates_multiple_groups(self, memory_store):
        """Different (command, status) groups are consolidated separately."""
        for _ in range(4):
            memory_store.save_action("bot1", "query", "ok")
        for _ in range(3):
            memory_store.save_action("bot1", "query", "fail")
        patterns = memory_store.consolidate("bot1", min_group_size=3)
        assert len(patterns) == 2

    def test_preserves_other_bots(self, memory_store):
        """Consolidation for bot1 doesn't affect bot2."""
        for _ in range(5):
            memory_store.save_action("bot1", "query", "ok")
        memory_store.save_action("bot2", "query", "ok")
        memory_store.consolidate("bot1", min_group_size=3)
        records = memory_store.query_history(bot_name="bot2")
        assert len(records) == 1

    def test_returns_pattern_metadata(self, memory_store):
        """Pattern dict contains all required fields."""
        for _ in range(3):
            memory_store.save_action("bot1", "check", "ok")
        patterns = memory_store.consolidate("bot1", min_group_size=3)
        p = patterns[0]
        assert p["bot_name"] == "bot1"
        assert p["command"] == "check"
        assert p["status"] == "ok"
        assert p["consolidated_count"] == 3
        assert "first_seen" in p
        assert "last_seen" in p

    def test_idempotent(self, memory_store):
        """Running consolidate twice doesn't break anything."""
        for _ in range(5):
            memory_store.save_action("bot1", "query", "ok")
        memory_store.consolidate("bot1", min_group_size=3)
        patterns = memory_store.consolidate("bot1", min_group_size=3)
        assert len(patterns) == 0  # nothing left to consolidate
