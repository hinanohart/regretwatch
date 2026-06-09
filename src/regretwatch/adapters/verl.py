"""Stub adapter for verl RLVR rollout logs (v0.2 real-dump reconciliation).

verl logs grouped rollouts with per-rollout rewards. The mapping to schema.v1 is below;
not implemented in v0.1 (no real dump reconciled yet). Convert to schema.v1 JSONL and use
:func:`regretwatch.io.load_logs`.

schema.v1 mapping (TODO v0.2):
    prompt_id  <- prompt uid
    draw_order <- rollout index in the group
    cost       <- response length in tokens
    correct    <- reward >= success threshold   (or score <- reward)
"""

from __future__ import annotations

from pathlib import Path

from .._types import PromptSeq


def load_verl(path: str | Path) -> list[PromptSeq]:  # pragma: no cover - stub
    raise NotImplementedError(
        "verl adapter is a v0.2 stub; convert verl rollout logs to regretwatch.schema.v1 "
        "JSONL and use regretwatch.io.load_logs (see module docstring)."
    )
