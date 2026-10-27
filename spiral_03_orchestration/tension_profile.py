"""TensionProfile — 4 operating profiles that calibrate the tension field.

Analogy for a C dev:
- EXPLORATORY = -O0 debug build — max freedom, explore everything
- STANDARD = -O2 — balanced performance/safety
- CRITICAL = -O3 -Wall -Werror — maximum rigor, zero tolerance
- DESTRUCTIVE = safe mode — when everything is broken, maximum caution

Each profile defines:
- creativity_weight: how much exploration is rewarded
- logic_weight: how much convergence is rewarded
- confidence_threshold: minimum confidence to proceed with an action
- max_chunks: how many chunks to consult
- use_reranker: whether to activate the reranker
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProfileType(str, Enum):
    """The 4 tension profiles."""

    EXPLORATORY = "EXPLORATORY"
    STANDARD = "STANDARD"
    CRITICAL = "CRITICAL"
    DESTRUCTIVE = "DESTRUCTIVE"


@dataclass(frozen=True)
class TensionProfile:
    """Immutable configuration for a tension operating mode."""

    name: ProfileType
    creativity_weight: float
    logic_weight: float
    confidence_threshold: float
    max_chunks: int
    use_reranker: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "name": self.name.value,
            "creativity_weight": self.creativity_weight,
            "logic_weight": self.logic_weight,
            "confidence_threshold": self.confidence_threshold,
            "max_chunks": self.max_chunks,
            "use_reranker": self.use_reranker,
        }


# Pre-built profiles matching the discovery doc spec
PROFILES: dict[ProfileType, TensionProfile] = {
    ProfileType.EXPLORATORY: TensionProfile(
        name=ProfileType.EXPLORATORY,
        creativity_weight=0.8,
        logic_weight=0.2,
        confidence_threshold=0.15,
        max_chunks=10,
        use_reranker=False,
        description="Maximum freedom. Explore broadly, tolerate low confidence.",
    ),
    ProfileType.STANDARD: TensionProfile(
        name=ProfileType.STANDARD,
        creativity_weight=0.5,
        logic_weight=0.5,
        confidence_threshold=0.3,
        max_chunks=5,
        use_reranker=True,
        description="Balanced exploration and convergence. Default mode.",
    ),
    ProfileType.CRITICAL: TensionProfile(
        name=ProfileType.CRITICAL,
        creativity_weight=0.2,
        logic_weight=0.8,
        confidence_threshold=0.6,
        max_chunks=8,
        use_reranker=True,
        description="High rigor. More chunks, stricter threshold, reranker active.",
    ),
    ProfileType.DESTRUCTIVE: TensionProfile(
        name=ProfileType.DESTRUCTIVE,
        creativity_weight=0.1,
        logic_weight=0.9,
        confidence_threshold=0.8,
        max_chunks=3,
        use_reranker=True,
        description="Maximum caution. Only act on very high confidence.",
    ),
}


def get_profile(name: str | ProfileType) -> TensionProfile:
    """Get a profile by name.

    Args:
        name: Profile name (case-insensitive string or ProfileType enum).

    Returns:
        The corresponding TensionProfile.

    Raises:
        ValueError: If profile name is not recognized.
    """
    if isinstance(name, ProfileType):
        return PROFILES[name]

    try:
        profile_type = ProfileType(name.upper())
    except ValueError:
        valid = ", ".join(p.value for p in ProfileType)
        raise ValueError(f"Unknown profile: {name}. Valid: {valid}")

    return PROFILES[profile_type]
