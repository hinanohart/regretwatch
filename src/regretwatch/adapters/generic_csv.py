"""Most-general adapter: a long-form CSV with one row per sample.

Columns: ``prompt_id, draw_order, cost`` (required) and any of ``correct, score, answer,
gold``. ``correct`` accepts ``1/0/true/false``. This is the lowest-friction path for logs
that are not in schema.v1 JSONL already.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .._types import PromptSeq

_TRUE = {"1", "true", "t", "yes", "y"}
_FALSE = {"0", "false", "f", "no", "n", ""}


def _to_bool(v: str) -> bool:
    s = v.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"cannot parse bool from {v!r}")


def load_csv(path: str | Path) -> list[PromptSeq]:
    """Load a long-form CSV (one row per sample) into PromptSeqs."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            groups[row["prompt_id"]].append(row)

    seqs: list[PromptSeq] = []
    for pid, rows in groups.items():
        rows.sort(key=lambda r: int(r["draw_order"]))
        cost = np.array([float(r["cost"]) for r in rows])
        has_correct = all(r.get("correct", "") != "" for r in rows)
        has_score = all(r.get("score", "") != "" for r in rows)
        has_answer = all(r.get("answer", "") != "" for r in rows)
        correct = np.array([_to_bool(r["correct"]) for r in rows]) if has_correct else None
        score = np.array([float(r["score"]) for r in rows]) if has_score else None
        answer = np.array([str(r["answer"]) for r in rows]) if has_answer else None
        gold = rows[0].get("gold") or None
        seqs.append(PromptSeq(pid, cost, correct=correct, score=score, answer=answer, gold=gold))
    return seqs
