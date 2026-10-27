"""Config watcher — live environment variable reader with periodic reload.

Analogy for a C dev:
    The standard config.py reads env vars at import time, like global constants
    set at program startup. This module is like inotify(7) — it watches for
    changes and refreshes values without restarting the process.

Usage (opt-in — does NOT affect config.py):
    watcher = ConfigWatcher()
    chunk_size = watcher.get("CHUNK_SIZE", "512")

    # Force reload from disk (picks up .env changes):
    watcher.reload()

This module is thread-safe: concurrent reads and reloads are serialized
via a threading.Lock, matching the guarantee expected in pod environments
where multiple threads may be serving requests while a reload fires.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from dotenv import dotenv_values, find_dotenv


class ConfigWatcher:
    """Reads environment variables with optional dotenv reload.

    Values are always read from os.environ at call time, not cached
    at import time. Call reload() to re-parse the .env file into
    os.environ before the next read.
    """

    def __init__(self, env_path: str | None = None) -> None:
        """Initialize the watcher.

        Args:
            env_path: Path to .env file. If None, uses python-dotenv's
                      find_dotenv() to locate the nearest .env file.
        """
        self._env_path = env_path or find_dotenv(usecwd=True) or ".env"
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """Read an environment variable (always current, never cached).

        Args:
            key: Environment variable name.
            default: Value to return if key is not set.

        Returns:
            Current value from os.environ, or default.
        """
        with self._lock:
            return os.environ.get(key, default)

    def reload(self) -> None:
        """Re-read the .env file and update os.environ.

        Only updates keys that are present in the .env file. Keys that
        were set outside the .env file (e.g. via export in the shell)
        are not affected.
        """
        with self._lock:
            values = dotenv_values(self._env_path)
            for key, value in values.items():
                if value is not None:
                    os.environ[key] = value

    def get_int(self, key: str, default: int = 0) -> int:
        """Read an integer environment variable."""
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return int(raw)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Read a float environment variable."""
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except (ValueError, TypeError):
            return default
