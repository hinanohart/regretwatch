"""Stub adapter for simple-evals sampler logs (v0.2 real-dump reconciliation).

simple-evals records per-question single responses; best-of-N auditing needs the repeated
samples variant. Mapping to schema.v1 below; not implemented in v0.1. Convert to schema.v1
JSONL and use :func:`regretwatch.io.load_logs`.

schema.v1 mapping (TODO v0.2):
    prompt_id  <- question id
    draw_order <- repeat index
    cost       <- completion token count
    correct    <- score == 1.0
"""

from __future__ import annotations

from pathlib import Path

from .._types import PromptSeq


def load_simple_evals(path: str | Path) -> list[PromptSeq]:  # pragma: no cover - stub
    raise NotImplementedError(
        "simple_evals adapter is a v0.2 stub; convert simple-evals logs to "
        "regretwatch.schema.v1 JSONL and use regretwatch.io.load_logs (see module docstring)."
    )
