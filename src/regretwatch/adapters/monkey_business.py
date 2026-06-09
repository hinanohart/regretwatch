"""Adapter for ScalingIntelligence/monkey_business (Large Language Monkeys) dumps.

Each record is ``{question, prompt, samples[], is_corrects[], gt_answer, ...}`` with up to
10000 samples per prompt and **no per-sample token count**, so cost defaults to unit
(1 per draw). The realized policy is modelled as fixed best-of-N over the first
``realized_n`` draws. Large configs are streamed with ``ijson`` when available, falling
back to ``json.load`` otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._types import PromptSeq


def _iter_records(path: str | Path) -> Any:
    try:
        import ijson
    except ImportError:
        with open(path, encoding="utf-8") as fh:
            yield from json.load(fh)
        return
    with open(path, "rb") as fh:
        yield from ijson.items(fh, "item")


def load_monkey_business(
    path: str | Path, *, realized_n: int = 16, cost_unit: str = "unit"
) -> list[PromptSeq]:
    """Load a monkey_business config into PromptSeqs (first ``realized_n`` draws).

    Parameters
    ----------
    realized_n:
        Number of leading draws treated as the realized best-of-N rollout.
    cost_unit:
        ``"unit"`` (1 per draw) -- monkey_business carries no token counts.
    """
    seqs: list[PromptSeq] = []
    for i, rec in enumerate(_iter_records(path)):
        ic = list(rec.get("is_corrects", []))[:realized_n]
        if not ic:
            continue
        correct = np.array([bool(x) for x in ic], dtype=bool)
        cost = np.ones(correct.size, dtype=float)
        pid = str(rec.get("orig_dset_idx", i))
        seqs.append(
            PromptSeq(
                prompt_id=f"mb::{pid}",
                cost=cost,
                correct=correct,
                bucket={"source": "monkey_business", "cost_unit": cost_unit},
            )
        )
    return seqs
