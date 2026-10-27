"""Tests for spiral_01_chunks.shutdown — atexit cleanup hooks.

Verifies:
- register_for_cleanup adds resources to the internal list
- _cleanup calls .close() on each resource in reverse order
- _cleanup handles exceptions gracefully (no propagation)
- _cleanup clears the resource list after running
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spiral_01_chunks.shutdown import _cleanup, _resources, register_for_cleanup


@pytest.fixture(autouse=True)
def _clear_resources():
    """Ensure the resource list is empty before and after each test."""
    _resources.clear()
    yield
    _resources.clear()


class TestRegisterForCleanup:
    """register_for_cleanup adds resources to the internal list."""

    def test_adds_single_resource(self) -> None:
        obj = MagicMock()
        register_for_cleanup(obj)
        assert obj in _resources
        assert len(_resources) == 1

    def test_adds_multiple_resources(self) -> None:
        a, b, c = MagicMock(), MagicMock(), MagicMock()
        register_for_cleanup(a)
        register_for_cleanup(b)
        register_for_cleanup(c)
        assert _resources == [a, b, c]

    def test_preserves_insertion_order(self) -> None:
        objs = [MagicMock(name=f"r{i}") for i in range(5)]
        for o in objs:
            register_for_cleanup(o)
        assert _resources == objs


class TestCleanup:
    """_cleanup calls .close() in reverse order and handles errors."""

    def test_calls_close_on_resources(self) -> None:
        r = MagicMock()
        register_for_cleanup(r)
        _cleanup()
        r.close.assert_called_once()

    def test_reverse_order(self) -> None:
        """Resources must be closed in LIFO order."""
        order: list[str] = []
        a = MagicMock()
        a.close.side_effect = lambda: order.append("a")
        b = MagicMock()
        b.close.side_effect = lambda: order.append("b")
        c = MagicMock()
        c.close.side_effect = lambda: order.append("c")

        register_for_cleanup(a)
        register_for_cleanup(b)
        register_for_cleanup(c)
        _cleanup()

        assert order == ["c", "b", "a"]

    def test_clears_list_after_cleanup(self) -> None:
        register_for_cleanup(MagicMock())
        register_for_cleanup(MagicMock())
        assert len(_resources) == 2
        _cleanup()
        assert len(_resources) == 0

    def test_handles_exception_gracefully(self) -> None:
        """If .close() raises, cleanup continues for remaining resources."""
        bad = MagicMock()
        bad.close.side_effect = RuntimeError("boom")
        good = MagicMock()

        register_for_cleanup(good)
        register_for_cleanup(bad)

        # Should not raise
        _cleanup()

        # Both were attempted (bad first in reverse, then good)
        bad.close.assert_called_once()
        good.close.assert_called_once()

    def test_skips_resources_without_close(self) -> None:
        """Objects without a .close() method are silently skipped."""
        obj_no_close = 42  # int has no .close()
        obj_with_close = MagicMock()

        register_for_cleanup(obj_no_close)
        register_for_cleanup(obj_with_close)

        _cleanup()
        obj_with_close.close.assert_called_once()

    def test_idempotent_double_cleanup(self) -> None:
        """Calling _cleanup twice should not fail or double-close."""
        r = MagicMock()
        register_for_cleanup(r)
        _cleanup()
        r.close.assert_called_once()

        # Second call: list is empty, no-op
        _cleanup()
        r.close.assert_called_once()  # still only 1 call
