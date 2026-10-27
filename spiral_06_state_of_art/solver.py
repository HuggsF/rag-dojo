"""Solver — executes proposed improvements.

Analogy for a C dev:
    The proposer is `gcc -fsuggest-attribute`. The solver is the actual patch —
    it modifies the config, re-indexes if needed, and returns what changed.
    Always reversible. Always scoped to the proposal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from spiral_06_state_of_art.config import IMMUTABLE_FILES, TUNABLE_PARAMS
from spiral_06_state_of_art.proposer import Proposal


@dataclass
class SolverAction:
    """Record of an action taken by the solver."""

    param: str
    old_value: object
    new_value: object
    applied: bool
    error: str | None = None

    def __repr__(self) -> str:
        status = "OK" if self.applied else f"FAILED: {self.error}"
        return f"SolverAction({self.param}: {self.old_value} → {self.new_value}, {status})"


@dataclass
class SolverResult:
    """Result of executing a set of proposals."""

    actions: list[SolverAction] = field(default_factory=list)
    config_snapshot_before: dict[str, object] = field(default_factory=dict)
    config_snapshot_after: dict[str, object] = field(default_factory=dict)
    reverted: bool = False

    @property
    def success(self) -> bool:
        """True if all actions were applied successfully."""
        return all(a.applied for a in self.actions) and len(self.actions) > 0

    @property
    def files_modified(self) -> int:
        """Number of config parameters changed."""
        return sum(1 for a in self.actions if a.applied)

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "PARTIAL"
        return f"SolverResult({status}, {self.files_modified} changes)"


class Solver:
    """Executes proposals by modifying config parameters.

    Operates on an in-memory config dict — does NOT modify files directly.
    This makes it safe, testable, and reversible.
    """

    def __init__(self, config: dict[str, object]) -> None:
        self._config = dict(config)
        self._original = dict(config)
        self._version: int = 0
        self._history: list[dict[str, object]] = []

    @property
    def config(self) -> dict[str, object]:
        """Current config state."""
        return dict(self._config)

    def apply(self, proposal: Proposal) -> SolverAction:
        """Apply a single proposal to the config."""
        param = proposal.param

        # Validate param is tunable
        if param not in TUNABLE_PARAMS:
            return SolverAction(
                param=param,
                old_value=proposal.current_value,
                new_value=proposal.proposed_value,
                applied=False,
                error=f"Parameter {param!r} is not tunable",
            )

        tunable = TUNABLE_PARAMS[param]
        proposed = proposal.proposed_value

        # Validate bounds
        if isinstance(proposed, (int, float)):
            min_val = tunable["min"]
            max_val = tunable["max"]
            if proposed < min_val or proposed > max_val:  # type: ignore[operator]
                return SolverAction(
                    param=param,
                    old_value=self._config.get(param),
                    new_value=proposed,
                    applied=False,
                    error=f"Value {proposed} outside bounds [{min_val}, {max_val}]",
                )

        old_value = self._config.get(param)
        self._config[param] = proposed

        # Track version history
        self._version += 1
        self._history.append(
            {
                "version": self._version,
                "timestamp": time.time(),
                "param": param,
                "old_value": old_value,
                "new_value": proposed,
            }
        )

        return SolverAction(
            param=param,
            old_value=old_value,
            new_value=proposed,
            applied=True,
        )

    def apply_proposals(self, proposals: list[Proposal]) -> SolverResult:
        """Apply a list of proposals, return result with full audit trail."""
        snapshot_before = dict(self._config)
        actions: list[SolverAction] = []

        for proposal in proposals:
            action = self.apply(proposal)
            actions.append(action)

        return SolverResult(
            actions=actions,
            config_snapshot_before=snapshot_before,
            config_snapshot_after=dict(self._config),
        )

    def config_version(self) -> int:
        """Return the current config version (incremented on each successful apply)."""
        return self._version

    def config_history(self) -> list[dict]:
        """Return the full change log as a list of dicts.

        Each entry contains: version, timestamp, param, old_value, new_value.
        """
        return list(self._history)

    def revert(self) -> dict[str, object]:
        """Revert config to original state. Returns the reverted config."""
        self._config = dict(self._original)
        return dict(self._config)

    def validate_immutable(self, files: list[str]) -> list[str]:
        """Check if any files in the list are immutable. Returns violations."""
        return [f for f in files if f in IMMUTABLE_FILES]
