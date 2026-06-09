"""Load and validate ``regretwatch.schema.v1`` JSONL logs (fail-closed).

The validator is dependency-free (no jsonschema): it checks required fields, types, and
the ``correct``-or-``score`` constraint, returning ``False`` on the first violation. When
majority aggregation is requested but answer ids are absent, the loader records that the
collapse fallback (a worst-case lower bound) is in effect.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ._types import PromptSeq

_ALLOWED_UNITS = {"tokens", "unit", "wall_ms"}


def _check_row(row: Any) -> str | None:
    """Return an error string if the row is invalid, else ``None``."""
    if not isinstance(row, dict):
        return "row is not an object"
    if not isinstance(row.get("prompt_id"), str):
        return "prompt_id must be a string"
    do = row.get("draw_order")
    if not isinstance(do, int) or isinstance(do, bool) or do < 0:
        return "draw_order must be an int >= 0"
    cost = row.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
        return "cost must be a number > 0"
    if "correct" not in row and "score" not in row:
        return "row needs at least one of correct / score"
    if "correct" in row and not isinstance(row["correct"], bool):
        return "correct must be a boolean"
    if "score" in row and (not isinstance(row["score"], (int, float)) or isinstance(row["score"], bool)):
        return "score must be a number"
    if "cost_unit" in row and row["cost_unit"] not in _ALLOWED_UNITS:
        return f"cost_unit must be one of {sorted(_ALLOWED_UNITS)}"
    return None


def validate(path: str | Path) -> bool:
    """Validate a JSONL log against schema.v1. Returns ``True`` iff every row is valid."""
    seen: dict[str, set[int]] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"line {lineno}: invalid JSON ({exc})")
                    return False
                err = _check_row(row)
                if err is not None:
                    print(f"line {lineno}: {err}")
                    return False
                pid, do = row["prompt_id"], row["draw_order"]
                if do in seen.setdefault(pid, set()):
                    print(f"line {lineno}: duplicate draw_order {do} for prompt_id {pid!r}")
                    return False
                seen[pid].add(do)
    except OSError as exc:
        print(f"cannot read {path}: {exc}")
        return False
    return True


def load_logs(path: str | Path) -> list[PromptSeq]:
    """Load a schema.v1 JSONL log into draw-ordered :class:`PromptSeq` objects.

    Rows are grouped by ``prompt_id`` and sorted by ``draw_order``. Raises ``ValueError``
    on a malformed log (call :func:`validate` first to fail closed gracefully).
    """
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            err = _check_row(row)
            if err is not None:
                raise ValueError(f"line {lineno}: {err}")
            rows[row["prompt_id"]].append(row)

    seqs: list[PromptSeq] = []
    for pid, group in rows.items():
        group.sort(key=lambda r: r["draw_order"])
        orders = [r["draw_order"] for r in group]
        if len(set(orders)) != len(orders):
            raise ValueError(f"prompt_id {pid!r}: duplicate draw_order values")
        cost = np.array([float(r["cost"]) for r in group])
        has_correct = all("correct" in r for r in group)
        has_score = all("score" in r for r in group)
        has_answer = all("answer" in r for r in group)
        correct = np.array([bool(r["correct"]) for r in group]) if has_correct else None
        score = np.array([float(r["score"]) for r in group]) if has_score else None
        answer = np.array([str(r["answer"]) for r in group]) if has_answer else None
        gold = group[0].get("gold")
        bucket = dict(group[0].get("bucket", {}))
        seqs.append(
            PromptSeq(
                pid,
                cost,
                correct=correct,
                score=score,
                answer=answer,
                gold=gold,
                bucket={str(k): str(v) for k, v in bucket.items()},
            )
        )
    return seqs


def majority_uses_collapse(seqs: list[PromptSeq]) -> bool:
    """True if any prompt lacks answer ids, so majority must use the collapse lower bound."""
    return any(s.answer is None or s.gold is None for s in seqs)
