"""Tests for ConfigWatcher."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import pytest

from spiral_01_chunks.config_watcher import ConfigWatcher


class TestConfigWatcherGet:
    def test_reads_existing_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_RAG_KEY", "hello")
        watcher = ConfigWatcher()
        assert watcher.get("TEST_RAG_KEY") == "hello"

    def test_returns_default_when_missing(self) -> None:
        watcher = ConfigWatcher()
        assert watcher.get("__NONEXISTENT_KEY_XYZ__", "fallback") == "fallback"

    def test_returns_none_default_when_missing(self) -> None:
        watcher = ConfigWatcher()
        assert watcher.get("__NONEXISTENT_KEY_XYZ__") is None

    def test_reads_updated_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get() must reflect changes to os.environ without reload."""
        monkeypatch.setenv("TEST_LIVE_KEY", "v1")
        watcher = ConfigWatcher()
        assert watcher.get("TEST_LIVE_KEY") == "v1"

        monkeypatch.setenv("TEST_LIVE_KEY", "v2")
        assert watcher.get("TEST_LIVE_KEY") == "v2"


class TestConfigWatcherGetInt:
    def test_parses_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT_KEY", "42")
        watcher = ConfigWatcher()
        assert watcher.get_int("TEST_INT_KEY") == 42

    def test_returns_default_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT_KEY", "not_a_number")
        watcher = ConfigWatcher()
        assert watcher.get_int("TEST_INT_KEY", default=7) == 7

    def test_returns_default_when_missing(self) -> None:
        watcher = ConfigWatcher()
        assert watcher.get_int("__MISSING__", default=99) == 99


class TestConfigWatcherGetFloat:
    def test_parses_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT_KEY", "3.14")
        watcher = ConfigWatcher()
        assert abs(watcher.get_float("TEST_FLOAT_KEY") - 3.14) < 1e-9

    def test_returns_default_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT_KEY", "nan_value")
        watcher = ConfigWatcher()
        assert watcher.get_float("TEST_FLOAT_KEY", default=0.5) == 0.5


class TestConfigWatcherReload:
    def test_reload_picks_up_new_value(self) -> None:
        """reload() must update os.environ from the .env file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_RELOAD_KEY=initial\n")
            env_path = f.name

        try:
            watcher = ConfigWatcher(env_path=env_path)
            watcher.reload()
            assert os.environ.get("TEST_RELOAD_KEY") == "initial"

            # Update the file
            Path(env_path).write_text("TEST_RELOAD_KEY=updated\n")
            watcher.reload()
            assert os.environ.get("TEST_RELOAD_KEY") == "updated"
        finally:
            os.unlink(env_path)
            os.environ.pop("TEST_RELOAD_KEY", None)

    def test_reload_with_nonexistent_file_does_not_raise(self) -> None:
        """reload() on a missing .env file must not crash."""
        watcher = ConfigWatcher(env_path="/nonexistent/.env")
        watcher.reload()  # should not raise


class TestConfigWatcherThreadSafety:
    def test_concurrent_reads_do_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple threads reading simultaneously must not raise."""
        monkeypatch.setenv("TEST_THREAD_KEY", "shared_value")
        watcher = ConfigWatcher()
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    watcher.get("TEST_THREAD_KEY")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
