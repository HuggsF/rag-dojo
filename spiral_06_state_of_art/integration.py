"""Integration — bridges spiral_06 with spirals 01-05.

Analogy for a C dev:
    Like a linker that resolves external symbols. Each spiral is a .o file
    with its own symbol table. This module links them into an executable.

Provides:
    - build_answer_fn: creates an AnswerFn from LayerPipeline (spiral_05)
    - build_pipeline: assembles a fully-integrated LayerPipeline with all backends
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from spiral_06_state_of_art.verifier import AnswerFn

if TYPE_CHECKING:
    from spiral_05_economics.layer_pipeline import LayerPipeline


def build_answer_fn(pipeline: LayerPipeline) -> AnswerFn:
    """Create an AnswerFn (spiral_06) from a LayerPipeline (spiral_05).

    The AnswerFn protocol: question → (answer, latency_s, cost).
    LayerPipeline.query() returns LayerResult with answer, cost, metadata.
    This bridge measures wall-clock latency and extracts the fields.
    """

    def answer_fn(question: str) -> tuple[str, float, float]:
        t0 = time.monotonic()
        result = pipeline.query(question)
        latency = time.monotonic() - t0
        return (result.answer, latency, result.cost)

    return answer_fn
