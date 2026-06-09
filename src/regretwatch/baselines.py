"""Achievable (causal) baselines and the best-fixed-N excess-waste headline.

All baselines observe only a *prefix* of the realized draws (causal), unlike the
clairvoyant oracle. ``excess_waste_pct`` (realized vs best-fixed-N at matched dataset
accuracy) is the product headline: a causal, implementable comparator, so "unfair
hindsight inflation" cannot apply to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._types import Agg, PromptSeq
from .oracle import _agg_correct, clairvoyant_cost, realized_cost, realized_outcome


def _prefix_cost(seq: PromptSeq, k: int) -> float:
    """Cumulative cost of the first ``min(k, N)`` draws."""
    k = min(max(k, 1), seq.n_draws)
    return float(seq.cum_cost[k - 1])


def _acc_at(seqs: list[PromptSeq], k: int, agg: Agg) -> float:
    return float(np.mean([_agg_correct(s, k, agg) for s in seqs]))


def realized_accuracy(seqs: list[PromptSeq], agg: Agg) -> float:
    return float(np.mean([realized_outcome(s, agg) for s in seqs]))


@dataclass(frozen=True)
class WasteReport:
    """Best-fixed-N excess-waste headline (the product's top-line number)."""

    excess_waste_pct: float
    ci_lo: float
    ci_hi: float
    n_match: int
    realized_total: float
    bestfixedN_total: float
    matched_accuracy: float


@dataclass(frozen=True)
class GapReport:
    """Clairvoyant <-> achievable decomposition (always rendered as a pair)."""

    realized_total: float
    achievable_total: float
    clairvoyant_total: float
    closeable_lever: float  # realized - achievable (a causal policy could close this)
    noncausal_residual: float  # achievable - clairvoyant (no causal policy can close)


def best_fixed_n(seqs: list[PromptSeq], agg: Agg) -> tuple[int, float]:
    """Smallest fixed budget ``N*`` whose dataset accuracy matches the realized policy.

    A fixed budget ``N`` means each prompt draws ``min(N, its length)``, so the sweep runs
    up to the *longest* prompt (``max`` n_draws): at that budget every prompt is at full
    length and dataset accuracy equals the realized accuracy by construction, so a match
    always exists even for ragged (unequal-length) logs. (Sweeping only to the *shortest*
    prompt would let ``N*`` silently default to a budget that cannot match realized
    accuracy.) Returns ``(N*, total_cost_at_N*)``. Per-prompt regret under best-fixed-N is
    constant, which is precisely why it cannot localize *which* prompts are wasteful (the
    gap the clairvoyant oracle fills).
    """
    target = realized_accuracy(seqs, agg)
    max_n = max(s.n_draws for s in seqs)
    nstar = max_n
    for k in range(1, max_n + 1):
        if _acc_at(seqs, k, agg) >= target - 1e-12:
            nstar = k
            break
    total = float(sum(_prefix_cost(s, nstar) for s in seqs))
    return nstar, total


def _boot_pct_ci(
    realized: npt.NDArray[np.float64], fixed: npt.NDArray[np.float64], b: int, seed: int
) -> tuple[float, float]:
    """Cluster bootstrap CI of the ratio (sum realized - sum fixed)/sum fixed."""
    n = realized.size
    rng = np.random.default_rng(seed)
    vals = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        rf = float(fixed[idx].sum())
        vals[i] = (float(realized[idx].sum()) - rf) / rf if rf > 0 else np.nan
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def excess_waste_pct(seqs: list[PromptSeq], agg: Agg, *, b: int = 2000, seed: int = 0) -> WasteReport:
    """Headline: ``(realized_total - bestfixedN_total) / bestfixedN_total`` with CI.

    Signed and honest: if the deployed adaptive policy beats the best fixed N, the
    percentage is reported as negative rather than hidden.
    """
    nstar, fixed_total = best_fixed_n(seqs, agg)
    realized_arr = np.array([realized_cost(s) for s in seqs])
    fixed_arr = np.array([_prefix_cost(s, nstar) for s in seqs])
    realized_total = float(realized_arr.sum())
    pct = (realized_total - fixed_total) / fixed_total if fixed_total > 0 else float("nan")
    lo, hi = _boot_pct_ci(realized_arr, fixed_arr, b, seed)
    return WasteReport(
        excess_waste_pct=pct,
        ci_lo=lo,
        ci_hi=hi,
        n_match=nstar,
        realized_total=realized_total,
        bestfixedN_total=fixed_total,
        matched_accuracy=realized_accuracy(seqs, agg),
    )


def one_step_lookahead(seqs: list[PromptSeq], agg: Agg) -> float:
    """B2: stop at the first k where one more draw would not raise the prompt outcome."""
    total = 0.0
    for s in seqs:
        n = s.n_draws
        stop = n
        for k in range(1, n):
            if _agg_correct(s, k + 1, agg) <= _agg_correct(s, k, agg):
                stop = k
                break
        total += _prefix_cost(s, stop)
    return total


def running_agreement(seqs: list[PromptSeq], *, margin: int = 3) -> tuple[float, float]:
    """B3: label-free deployable policy -- stop when vote margin ``|2*sum(y)-k| >= m``.

    Returns ``(total_cost, accuracy)`` using per-draw correctness as the vote signal.
    """
    total = 0.0
    correct_flags: list[int] = []
    for s in seqs:
        if s.correct is None:
            raise ValueError("running_agreement requires per-draw correctness")
        n = s.n_draws
        stop = n
        for k in range(1, n + 1):
            sk = int(s.correct[:k].sum())
            if abs(2 * sk - k) >= margin:
                stop = k
                break
        total += _prefix_cost(s, stop)
        correct_flags.append(int(2 * int(s.correct[:stop].sum()) > stop))
    return total, float(np.mean(correct_flags))


def achievable_gap(seqs: list[PromptSeq], agg: Agg) -> GapReport:
    """Split realized -> achievable (closeable) and achievable -> clairvoyant (residual)."""
    realized_total = float(sum(realized_cost(s) for s in seqs))
    _, achievable_total = best_fixed_n(seqs, agg)
    clair_total = float(sum(clairvoyant_cost(s, realized_outcome(s, agg), agg) for s in seqs))
    return GapReport(
        realized_total=realized_total,
        achievable_total=achievable_total,
        clairvoyant_total=clair_total,
        closeable_lever=realized_total - achievable_total,
        noncausal_residual=achievable_total - clair_total,
    )
