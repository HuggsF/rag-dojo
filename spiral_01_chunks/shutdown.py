"""Shutdown hooks — graceful cleanup of resources on process exit.

Analogy for a C dev:
    Like atexit() handlers in C. When the process exits, we walk the list
    of registered resources in reverse order and call .close() on each.
    Reverse order ensures LIFO cleanup (last opened = first closed),
    just like unwinding a stack of file descriptors.

Usage:
    from spiral_01_chunks.shutdown import register_for_cleanup

    graph_store = GraphStore(...)
    register_for_cleanup(graph_store)
    # graph_store.close() will be called automatically at process exit
"""

from __future__ import annotations

import atexit
from typing import Any

_resources: list[Any] = []


def register_for_cleanup(resource: Any) -> None:
    """Register a resource for cleanup on shutdown."""
    _resources.append(resource)


def _cleanup() -> None:
    """Close all registered resources in reverse order (LIFO)."""
    for r in reversed(_resources):
        try:
            if hasattr(r, "close"):
                r.close()
        except Exception:
            pass
    _resources.clear()


atexit.register(_cleanup)
