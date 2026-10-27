"""Guardrails — blast radius limits and safety checks.

Analogy for a C dev:
    Like compile-time bounds checking + runtime assertions.
    Guardrails catch bad proposals BEFORE they execute.
    Blast radius = how many files/systems can one cycle affect.
    If you wouldn't `chmod 777 /`, don't bypass guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spiral_06_state_of_art.config import (
    CIRCUIT_BREAKER_TRIGGERS,
    IMMUTABLE_FILES,
    MAX_ECOSYSTEMS,
    MAX_FILES_PER_CYCLE,
    MAX_PRS_PER_CYCLE,
)


@dataclass
class GuardrailViolation:
    """A single guardrail violation."""

    rule: str
    message: str
    severity: str = "error"  # "error" = hard stop, "warning" = proceed with caution

    def __repr__(self) -> str:
        return f"GuardrailViolation({self.severity}: {self.rule} — {self.message})"


@dataclass
class GuardrailCheck:
    """Result of running all guardrail checks."""

    passed: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    warnings: list[GuardrailViolation] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({len(self.violations)} violations)"
        return f"GuardrailCheck({status})"


class Guardrails:
    """Enforces safety limits on improvement cycles."""

    def __init__(
        self,
        max_files: int = MAX_FILES_PER_CYCLE,
        max_prs: int = MAX_PRS_PER_CYCLE,
        max_ecosystems: int = MAX_ECOSYSTEMS,
        immutable_files: set[str] | None = None,
        circuit_breaker_triggers: dict[str, float | int] | None = None,
    ) -> None:
        self._max_files = max_files
        self._max_prs = max_prs
        self._max_ecosystems = max_ecosystems
        self._immutable = immutable_files or set(IMMUTABLE_FILES)
        self._triggers = circuit_breaker_triggers or dict(CIRCUIT_BREAKER_TRIGGERS)

    def check_blast_radius(
        self,
        files_modified: int = 0,
        prs_created: int = 0,
        ecosystems_affected: int = 1,
    ) -> GuardrailCheck:
        """Check if a proposed change stays within blast radius limits."""
        violations: list[GuardrailViolation] = []
        warnings: list[GuardrailViolation] = []

        if files_modified > self._max_files:
            violations.append(
                GuardrailViolation(
                    rule="max_files",
                    message=f"Would modify {files_modified} files (max: {self._max_files})",
                )
            )

        if prs_created > self._max_prs:
            violations.append(
                GuardrailViolation(
                    rule="max_prs",
                    message=f"Would create {prs_created} PRs (max: {self._max_prs})",
                )
            )

        if ecosystems_affected > self._max_ecosystems:
            violations.append(
                GuardrailViolation(
                    rule="max_ecosystems",
                    message=f"Would affect {ecosystems_affected} ecosystems (max: {self._max_ecosystems})",
                )
            )

        # Warn if near limits
        if files_modified == self._max_files:
            warnings.append(
                GuardrailViolation(
                    rule="max_files_warning",
                    message=f"At file limit ({files_modified}/{self._max_files})",
                    severity="warning",
                )
            )

        return GuardrailCheck(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def check_immutable(self, files: list[str]) -> GuardrailCheck:
        """Check if any files in the list are immutable."""
        violations: list[GuardrailViolation] = []

        for f in files:
            if f in self._immutable:
                violations.append(
                    GuardrailViolation(
                        rule="immutable_file",
                        message=f"File {f!r} is immutable and cannot be modified",
                    )
                )

        return GuardrailCheck(passed=len(violations) == 0, violations=violations)

    def check_circuit_breakers(
        self,
        budget_exceeded: bool = False,
        consecutive_failures: int = 0,
        score_regression: float = 0.0,
        constraint_violation: bool = False,
        cycle_depth: int = 0,
        cost_spike_pct: float = 0.0,
    ) -> GuardrailCheck:
        """Check all 6 circuit breaker triggers."""
        violations: list[GuardrailViolation] = []

        if budget_exceeded:
            violations.append(
                GuardrailViolation(
                    rule="budget_exceeded",
                    message="Budget has been exceeded",
                )
            )

        max_failures = int(self._triggers.get("consecutive_failures", 3))
        if consecutive_failures >= max_failures:
            violations.append(
                GuardrailViolation(
                    rule="consecutive_failures",
                    message=f"{consecutive_failures} consecutive failures (max: {max_failures})",
                )
            )

        regression_limit = float(self._triggers.get("score_regression", 5.0))
        if score_regression >= regression_limit:
            violations.append(
                GuardrailViolation(
                    rule="score_regression",
                    message=f"Score regressed by {score_regression:.1f} points (max: {regression_limit})",
                )
            )

        if constraint_violation:
            violations.append(
                GuardrailViolation(
                    rule="constraint_violation",
                    message="A guardrail constraint was violated",
                )
            )

        max_depth = int(self._triggers.get("max_depth", 3))
        if cycle_depth >= max_depth:
            violations.append(
                GuardrailViolation(
                    rule="max_depth",
                    message=f"Cycle depth {cycle_depth} reached max ({max_depth})",
                )
            )

        spike_limit = float(self._triggers.get("cost_spike_pct", 200))
        if cost_spike_pct > spike_limit:
            violations.append(
                GuardrailViolation(
                    rule="cost_spike",
                    message=f"Cost spike {cost_spike_pct:.0f}% exceeds limit ({spike_limit:.0f}%)",
                )
            )

        return GuardrailCheck(passed=len(violations) == 0, violations=violations)

    def full_check(
        self,
        files_modified: int = 0,
        prs_created: int = 0,
        ecosystems_affected: int = 1,
        files_list: list[str] | None = None,
        **circuit_kwargs: object,
    ) -> GuardrailCheck:
        """Run all guardrail checks at once."""
        all_violations: list[GuardrailViolation] = []
        all_warnings: list[GuardrailViolation] = []

        blast = self.check_blast_radius(files_modified, prs_created, ecosystems_affected)
        all_violations.extend(blast.violations)
        all_warnings.extend(blast.warnings)

        if files_list:
            immutable = self.check_immutable(files_list)
            all_violations.extend(immutable.violations)

        circuit = self.check_circuit_breakers(**circuit_kwargs)  # type: ignore[arg-type]
        all_violations.extend(circuit.violations)

        return GuardrailCheck(
            passed=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
        )
