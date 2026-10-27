# Spiral 06: State of the Art — The System That Improves Itself
# Unlocked after completing Spiral 05.
#
# Modules:
#   config.py           — Cycle limits, blast radius, benchmark paths
#   benchmark_suite.py  — 20 questions + ground truth for precision/recall
#   scorer.py           — Evaluate RAG dimensions: precision, recall, latency, cost
#   proposer.py         — Analyze scores → propose improvement
#   solver.py           — Execute proposed improvement (modify config, re-index)
#   verifier.py         — Run benchmarks before/after, compare scores
#   guardrails.py       — Blast radius, immutable files, budget check pre-cycle
#   cycle_manager.py    — Orchestrate PROPOSE → SOLVE → VERIFY (max 3 cycles)
#   reporter.py         — Generate cycle report: what changed, score delta
#   integration.py      — Bridge to spirals 01-05 via build_answer_fn

__all__ = [
    "AnswerFn",
    "BenchmarkQuestion",
    "BenchmarkReport",
    "BenchmarkSuite",
    "CycleManager",
    "CycleResult",
    "Guardrails",
    "GuardrailCheck",
    "Proposer",
    "Reporter",
    "RunResult",
    "ScoreCard",
    "Scorer",
    "Solver",
    "SolverResult",
    "Verifier",
    "VerificationResult",
    "build_answer_fn",
]
