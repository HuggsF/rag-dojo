"""Tests for scripts/ingest_with_retry.py — retry logic and resume support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.ingest_with_retry import (
    INITIAL_BACKOFF,
    MAX_BACKOFF,
    ingest_all,
    load_progress,
    save_progress,
)


@dataclass
class FakeChunk:
    """Minimal chunk for testing."""

    text: str
    source: str
    chunk_index: int


@pytest.fixture
def tmp_progress(tmp_path):
    """Return a temp progress file path."""
    return str(tmp_path / "progress.json")


@pytest.fixture
def fake_chunks():
    """Return a list of fake chunks."""
    return [FakeChunk(text=f"chunk text {i}", source="test.md", chunk_index=i) for i in range(5)]


@pytest.fixture
def mock_extractor():
    """Mock entity extractor that returns empty results."""
    ext = MagicMock()
    result = MagicMock()
    result.entities = []
    result.relations = []
    ext.extract_from_chunk.return_value = result
    return ext


@pytest.fixture
def mock_graph_store():
    """Mock graph store."""
    return MagicMock()


class TestLoadProgress:
    def test_returns_empty_set_for_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_progress(path) == set()

    def test_returns_empty_set_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert load_progress(str(path)) == set()

    def test_loads_ingested_ids(self, tmp_path):
        path = tmp_path / "progress.json"
        data = {"ingested_ids": ["chunk_a", "chunk_b"], "count": 2}
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_progress(str(path))
        assert result == {"chunk_a", "chunk_b"}


class TestSaveProgress:
    def test_saves_and_loads_roundtrip(self, tmp_path):
        path = str(tmp_path / "progress.json")
        ids = {"a::chunk_0", "b::chunk_1"}
        save_progress(path, ids)
        loaded = load_progress(path)
        assert loaded == ids

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "progress.json")
        save_progress(path, {"x"})
        assert Path(path).exists()

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "progress.json")
        save_progress(path, {"a"})
        save_progress(path, {"a", "b"})
        loaded = load_progress(path)
        assert loaded == {"a", "b"}


class TestIngestAll:
    def test_ingests_all_chunks(self, fake_chunks, mock_extractor, mock_graph_store, tmp_progress):
        stats = ingest_all(
            chunks=fake_chunks,
            extractor=mock_extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        assert stats["ingested_count"] == 5
        assert stats["skipped_count"] == 0
        assert stats["failed_count"] == 0
        assert mock_extractor.extract_from_chunk.call_count == 5
        assert mock_graph_store.persist_extraction.call_count == 5

    def test_resume_skips_already_ingested(
        self, fake_chunks, mock_extractor, mock_graph_store, tmp_progress
    ):
        # Pre-populate progress with first 2 chunks
        existing = {"test.md::chunk_0", "test.md::chunk_1"}
        save_progress(tmp_progress, existing)

        stats = ingest_all(
            chunks=fake_chunks,
            extractor=mock_extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        assert stats["ingested_count"] == 3
        assert stats["skipped_count"] == 2
        assert stats["resumed_from"] == 2

    def test_no_resume_starts_fresh(
        self, fake_chunks, mock_extractor, mock_graph_store, tmp_progress
    ):
        # Pre-populate progress
        save_progress(tmp_progress, {"test.md::chunk_0"})

        stats = ingest_all(
            chunks=fake_chunks,
            extractor=mock_extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
            resume=False,
        )
        assert stats["ingested_count"] == 5
        assert stats["skipped_count"] == 0

    def test_limit_caps_chunks(self, fake_chunks, mock_extractor, mock_graph_store, tmp_progress):
        stats = ingest_all(
            chunks=fake_chunks,
            extractor=mock_extractor,
            graph_store=mock_graph_store,
            limit=2,
            progress_path=tmp_progress,
        )
        assert stats["ingested_count"] == 2

    def test_non_rate_limit_error_recorded(
        self, fake_chunks, mock_extractor, mock_graph_store, tmp_progress
    ):
        mock_extractor.extract_from_chunk.side_effect = [
            ValueError("connection refused"),
            MagicMock(entities=[], relations=[]),
            MagicMock(entities=[], relations=[]),
            MagicMock(entities=[], relations=[]),
            MagicMock(entities=[], relations=[]),
        ]
        stats = ingest_all(
            chunks=fake_chunks,
            extractor=mock_extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        assert stats["failed_count"] == 1
        assert stats["ingested_count"] == 4
        assert len(stats["errors"]) == 1
        assert "connection refused" in stats["errors"][0]["error"]

    @patch("scripts.ingest_with_retry.time.sleep")
    def test_rate_limit_retries_with_backoff(
        self, mock_sleep, fake_chunks, mock_graph_store, tmp_progress
    ):
        extractor = MagicMock()
        # First call: 429, second call: success, rest: success
        result_ok = MagicMock(entities=[], relations=[])
        extractor.extract_from_chunk.side_effect = [
            Exception("429 rate limit exceeded"),
            result_ok,
            result_ok,
            result_ok,
            result_ok,
            result_ok,
        ]
        stats = ingest_all(
            chunks=fake_chunks,
            extractor=extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        # Should have slept once with initial backoff
        mock_sleep.assert_called_once_with(INITIAL_BACKOFF)
        assert stats["ingested_count"] == 5

    @patch("scripts.ingest_with_retry.time.sleep")
    def test_backoff_doubles_on_consecutive_429(self, mock_sleep, mock_graph_store, tmp_progress):
        chunks = [FakeChunk(text="t", source="x.md", chunk_index=0)]
        extractor = MagicMock()
        result_ok = MagicMock(entities=[], relations=[])
        # Three 429s then success
        extractor.extract_from_chunk.side_effect = [
            Exception("429"),
            Exception("429"),
            Exception("429"),
            result_ok,
        ]
        stats = ingest_all(
            chunks=chunks,
            extractor=extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        assert stats["ingested_count"] == 1
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == INITIAL_BACKOFF
        assert calls[1] == INITIAL_BACKOFF * 2
        assert calls[2] == INITIAL_BACKOFF * 4

    @patch("scripts.ingest_with_retry.time.sleep")
    def test_backoff_caps_at_max(self, mock_sleep, mock_graph_store, tmp_progress):
        chunks = [FakeChunk(text="t", source="x.md", chunk_index=0)]
        extractor = MagicMock()
        result_ok = MagicMock(entities=[], relations=[])
        # Many 429s — backoff should cap at MAX_BACKOFF
        side_effects = [Exception("429")] * 10 + [result_ok]
        extractor.extract_from_chunk.side_effect = side_effects
        ingest_all(
            chunks=chunks,
            extractor=extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert all(c <= MAX_BACKOFF for c in calls)

    def test_progress_saved_after_each_chunk(
        self, fake_chunks, mock_extractor, mock_graph_store, tmp_progress
    ):
        ingest_all(
            chunks=fake_chunks,
            extractor=mock_extractor,
            graph_store=mock_graph_store,
            progress_path=tmp_progress,
        )
        progress = load_progress(tmp_progress)
        assert len(progress) == 5
        assert "test.md::chunk_0" in progress
        assert "test.md::chunk_4" in progress
