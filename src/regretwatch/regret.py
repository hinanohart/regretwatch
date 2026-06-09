"""Per-prompt clairvoyant regret + prompt-cluster bootstrap CI + RM-noise fail-closed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import Agg, PromptSeq
from .oracle import clairvoyant_cost, realized_cost, realized_outcome


@dataclass(frozen=True)
class RegretReport:
    """Aggregate clairvoyant compute-regret with a prompt-cluster bootstrap CI."""

    mean: float
    ci_lo: float
    ci_hi: float
    total: float
    n_prompts: int
    per_bucket: dict[str, RegretReport]


def per_prompt_regret(seqs: list[PromptSeq], agg: Agg) -> np.ndarray:
    """``realized_cost - clairvoyant_cost`` per prompt, clipped at 0 (fail-closed)."""
    out = np.empty(len(seqs), dtype=float)
    for j, seq in enumerate(seqs):
        y = realized_outcome(seq, agg)
        rc = realized_cost(seq)
        oc = clairvoyant_cost(seq, y, agg)
        out[j] = max(rc - oc, 0.0)
    return out


def _bucket_key(seq: PromptSeq, keys: tuple[str, ...]) -> str:
    return "|".join(f"{k}={seq.bucket.get(k, '?')}" for k in keys)


def _boot_ci(values: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    """Prompt-cluster percentile bootstrap (resample prompts, never draws)."""
    n = values.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def aggregate_regret(
    seqs: list[PromptSeq],
    agg: Agg,
    *,
    strata: tuple[str, ...] | None = None,
    b: int = 2000,
    seed: int = 0,
) -> RegretReport:
    """Aggregate per-prompt regret with cluster bootstrap CI, optionally stratified."""
    r = per_prompt_regret(seqs, agg)
    lo, hi = _boot_ci(r, b, seed)
    per_bucket: dict[str, RegretReport] = {}
    if strata:
        groups: dict[str, list[int]] = {}
        for j, seq in enumerate(seqs):
            groups.setdefault(_bucket_key(seq, strata), []).append(j)
        for key, idxs in sorted(groups.items()):
            rb = r[np.asarray(idxs)]
            blo, bhi = _boot_ci(rb, b, seed + 1)
            per_bucket[key] = RegretReport(
                mean=float(rb.mean()),
                ci_lo=blo,
                ci_hi=bhi,
                total=float(rb.sum()),
                n_prompts=len(idxs),
                per_bucket={},
            )
    return RegretReport(
        mean=float(r.mean()) if r.size else float("nan"),
        ci_lo=lo,
        ci_hi=hi,
        total=float(r.sum()),
        n_prompts=len(seqs),
        per_bucket=per_bucket,
    )


def propagate_rm_noise(
    seqs: list[PromptSeq],
    agg: Agg,
    eta: float,
    *,
    b: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Parametric bootstrap of aggregate regret under two-sided label flips at rate eta.

    Each draw's correctness is independently flipped with probability ``eta`` (RM
    miscalibration), regret recomputed, and the 95% interval of the mean returned. Used
    by :func:`refuse_headline` to fail closed when the clairvoyant headline is dominated
    by reward-model noise.
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    means = np.empty(b, dtype=float)
    for i in range(b):
        flipped: list[PromptSeq] = []
        for seq in seqs:
            if seq.correct is None:
                flipped.append(seq)
                continue
            flip = rng.random(seq.correct.size) < eta
            new_correct = np.logical_xor(seq.correct, flip)
            flipped.append(
                PromptSeq(
                    seq.prompt_id,
                    seq.cost,
                    correct=new_correct,
                    score=seq.score,
                    answer=seq.answer,
                    gold=seq.gold,
                    bucket=seq.bucket,
                )
            )
        means[i] = per_prompt_regret(flipped, agg).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def refuse_headline(
    ci: tuple[float, float], *, clean_mean: float | None = None, rho_tol: float = 0.5
) -> bool:
    """REFUSE the clairvoyant headline when reward-model noise makes it untrustworthy.

    Triggers when (a) the noisy CI crosses 0, (b) its relative width exceeds ``rho_tol``,
    or (c) ``clean_mean`` is given and plausible label flips shift the regret estimate by
    more than ``rho_tol`` of its noise-free value. Criterion (c) is the load-bearing one:
    with many prompts the bootstrap CI of the *mean* is tight (CLT), so pure CI width
    rarely fires -- the noise-induced *shift* is what signals an untrustworthy headline.
    """
    lo, hi = ci
    if not np.isfinite(lo) or not np.isfinite(hi):
        return True
    if lo <= 0.0:
        return True
    mean = 0.5 * (lo + hi)
    if abs(mean) < 1e-12:
        return True
    if (hi - lo) / abs(mean) > rho_tol:
        return True
    if clean_mean is not None and clean_mean > 1e-12:
        if abs(mean - clean_mean) / clean_mean > rho_tol:
            return True
    return False
