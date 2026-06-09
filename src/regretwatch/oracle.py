"""Clairvoyant matched-accuracy oracle -- an O(N) prefix scan, NOT renewal-reward.

For one prompt with draw-ordered sequence ``seq = [(c_1, s_1), ..., (c_N, s_N)]`` and
cumulative cost ``C(k) = sum_{i<=k} c_i``, the realized policy stopped at the full
sequence and emitted ``agg(seq[1:N])`` with outcome ``y_real``. The matched-accuracy
clairvoyant lower bound is::

    OracleCost = min { C(k) : 1 <= k <= N, correct(agg(seq[1:k])) >= y_real }
               = C(N)   if unreachable   (fail-closed; regret is never negative)

This is a single O(N) prefix scan because cost is additive and the same realized draws
are reused (no resampling), so the renewal-reward fixed-point never appears. ``majority``
is non-monotone in k, which is exactly why a plain percentile cutoff is insufficient and
a full prefix scan is required.
"""

from __future__ import annotations

import numpy as np

from ._types import Agg, PromptSeq


def _agg_correct(seq: PromptSeq, k: int, agg: Agg) -> int:
    """Outcome (0/1) of aggregating the first ``k`` draws. Ties fail closed to 0."""
    if k <= 0:
        return 0
    k = min(k, seq.n_draws)
    if agg is Agg.BON:
        if seq.correct is None:
            raise ValueError("bon aggregation requires per-draw correctness")
        return int(bool(seq.correct[:k].any()))
    if agg is Agg.MAJORITY:
        if seq.answer is not None and seq.gold is not None:
            # True majority over answer ids; tie -> fail closed (0).
            ans = seq.answer[:k]
            vals, counts = np.unique(ans, return_counts=True)
            top = counts.max()
            if int((counts == top).sum()) != 1:
                return 0  # tie -> fail closed
            winner = vals[int(np.argmax(counts))]
            return int(winner == seq.gold)
        # Collapse fallback (wrong answers in one bucket): strict-majority-correct is a
        # worst-case lower bound on the true majority outcome.
        if seq.correct is None:
            raise ValueError("majority without answer ids requires correctness")
        s = int(seq.correct[:k].sum())
        return int(2 * s > k)  # strict majority; exact tie 2s==k -> 0
    if agg is Agg.VERIFIER_RERANK:
        if seq.score is None or seq.correct is None:
            raise ValueError("verifier_rerank requires score and correctness")
        idx = int(np.argmax(seq.score[:k]))  # ties -> first (numpy argmax)
        return int(bool(seq.correct[idx]))
    raise ValueError(f"unknown agg {agg!r}")


def realized_outcome(seq: PromptSeq, agg: Agg) -> int:
    """Outcome of the realized policy (aggregating all logged draws)."""
    return _agg_correct(seq, seq.n_draws, agg)


def clairvoyant_cost(seq: PromptSeq, y_target: int, agg: Agg) -> float:
    """Min cumulative cost of a prefix whose outcome matches ``y_target`` (fail-closed)."""
    cum = seq.cum_cost
    for k in range(1, seq.n_draws + 1):
        if _agg_correct(seq, k, agg) >= y_target:
            return float(cum[k - 1])
    return float(cum[-1])  # unreachable -> full cost


def realized_cost(seq: PromptSeq) -> float:
    """Total cost actually spent by the realized policy."""
    return float(seq.cum_cost[-1])
