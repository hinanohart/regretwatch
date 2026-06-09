"""Stub adapter for lm-evaluation-harness sample logs (v0.2 real-dump reconciliation).

lm-eval ``--log_samples`` writes per-doc JSONL. The mapping to schema.v1 is documented
below; the loader is intentionally not implemented in v0.1 (no real dump reconciled yet).
Convert to schema.v1 JSONL and use :func:`regretwatch.io.load_logs` in the meantime.

schema.v1 mapping (TODO v0.2):
    prompt_id  <- doc_id
    draw_order <- index within resampled generations
    cost       <- response token length (or unit)
    correct    <- per-resample exact_match / acc
"""

from __future__ import annotations

from pathlib import Path

from .._types import PromptSeq


def load_lm_eval(path: str | Path) -> list[PromptSeq]:  # pragma: no cover - stub
    raise NotImplementedError(
        "lm_eval adapter is a v0.2 stub; convert lm-eval --log_samples output to "
        "regretwatch.schema.v1 JSONL and use regretwatch.io.load_logs (see module docstring)."
    )
