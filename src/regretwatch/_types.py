"""Core data types shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Agg(str, Enum):
    """Aggregation / selection mode of the realized stopping policy."""

    BON = "bon"  # verifier-pick-best idealization: correct iff any draw is correct
    MAJORITY = "majority"  # self-consistency: strict majority of the prefix
    VERIFIER_RERANK = "verifier_rerank"  # argmax score over the prefix


@dataclass
class PromptSeq:
    """One prompt's realized, draw-ordered sample sequence.

    The log *is* the realized rollout: the policy drew ``len(cost)`` samples in
    ``draw_order`` and emitted ``agg(all draws)``. The oracle considers only prefixes
    of these same draws (no resampling), so ``oracle_cost <= realized_cost`` by
    construction.

    Attributes
    ----------
    prompt_id:
        Unique prompt identifier.
    cost:
        ``(T,)`` positive incremental cost per draw, in draw order.
    correct:
        ``(T,)`` bool correctness per draw, or ``None`` if only scores are available.
    score:
        ``(T,)`` float verifier / reward score per draw, or ``None``.
    answer:
        ``(T,)`` answer ids (for true majority vote), or ``None`` (collapse fallback).
    gold:
        Gold answer id for majority vote, or ``None``.
    bucket:
        Optional reporting key (task / model / temp / difficulty ...).
    """

    prompt_id: str
    cost: np.ndarray
    correct: np.ndarray | None = None
    score: np.ndarray | None = None
    answer: np.ndarray | None = None
    gold: str | None = None
    bucket: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cost = np.asarray(self.cost, dtype=float)
        if self.cost.ndim != 1 or self.cost.size == 0:
            raise ValueError(f"prompt {self.prompt_id}: cost must be non-empty 1-D")
        if (self.cost <= 0).any():
            raise ValueError(f"prompt {self.prompt_id}: all costs must be > 0")
        n = self.cost.size
        if self.correct is not None:
            self.correct = np.asarray(self.correct, dtype=bool)
            if self.correct.size != n:
                raise ValueError(f"prompt {self.prompt_id}: correct length mismatch")
        if self.score is not None:
            self.score = np.asarray(self.score, dtype=float)
            if self.score.size != n:
                raise ValueError(f"prompt {self.prompt_id}: score length mismatch")
        if self.answer is not None:
            self.answer = np.asarray(self.answer)
            if self.answer.size != n:
                raise ValueError(f"prompt {self.prompt_id}: answer length mismatch")
        if self.correct is None and self.score is None:
            raise ValueError(f"prompt {self.prompt_id}: need correct or score")

    @property
    def n_draws(self) -> int:
        return int(self.cost.size)

    @property
    def cum_cost(self) -> np.ndarray:
        return np.cumsum(self.cost)
