# Spiral 03: Orchestration — The Tension Field
# Unlocked after completing Spiral 02.
#
# Modules:
#   tension.py          — TensionMetrics: creativity, logic, delta_estado
#   tension_profile.py  — 4 profiles: EXPLORATORY, STANDARD, CRITICAL, DESTRUCTIVE
#   confidence.py       — Formula: weighted product + zero-delta collapse
#   compass.py          — MendaCompass: pre-action reasoning block
#   fallback.py         — 7-strategy cascade when delta = 0
#   orchestrator.py     — Main loop: query → tension eval → decide → execute → measure

__all__ = [
    "Action",
    "CompassReading",
    "ConfidenceResult",
    "FallbackCascade",
    "FallbackStrategy",
    "MendaCompass",
    "OrchestrationResult",
    "Orchestrator",
    "ProfileType",
    "TensionHistory",
    "TensionMetrics",
    "TensionProfile",
    "calculate_confidence",
]
