"""KILL-GATE: does the clairvoyant oracle have measurement power beyond best-fixed-N?

best-fixed-N yields a *constant* per-prompt regret, so it cannot rank prompts by waste.
``measurement_power`` quantifies whether the clairvoyant per-prompt regret carries
information a global fixed-N percentile does not. ``K2_distinguished`` (the gate) is
true when the two disagree on prompt ranking AND the clairvoyant regret has strictly
more spread. If the gate fails (clairvoyant == fixed-N verdict), the build degrades to
``achievable-only`` rather than shipping metric theater.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import kendalltau as _kendalltau
from scipy.stats import spearmanr as _spearmanr

# scipy's SignificanceResult is poorly typed across versions; route through Any so both
# mypy (scipy ignored) and pyright stay quiet without unused-ignore churn.
kendalltau: Any = _kendalltau
spearmanr: Any = _spearmanr


@dataclass(frozen=True)
class MeasurementPower:
    std_ratio: float
    spearman: float
    kendall_tau: float
    ci_overlap: bool
    loc_jaccard: float
    k2_distinguished: bool


def _boot_total_ci(values: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    n = values.size
    rng = np.random.default_rng(seed)
    totals = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        totals[i] = values[idx].sum()
    return (float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5)))


def measurement_power(
    r_clair: np.ndarray,
    r_fixedn: np.ndarray,
    *,
    theta2: float = 0.7,
    top_frac: float = 0.1,
    b: int = 2000,
    seed: int = 0,
) -> MeasurementPower:
    """Compare per-prompt clairvoyant regret against per-prompt best-fixed-N regret.

    Parameters
    ----------
    r_clair, r_fixedn:
        Per-prompt regret of the clairvoyant oracle and of the best-fixed-N policy.
    theta2:
        Kendall-tau threshold below which the two rankings count as disagreeing.
    """
    r_clair = np.asarray(r_clair, dtype=float)
    r_fixedn = np.asarray(r_fixedn, dtype=float)
    std_c = float(np.std(r_clair))
    std_f = float(np.std(r_fixedn))
    std_ratio = float("inf") if std_f < 1e-12 else std_c / std_f

    # Rank correlation; undefined when fixed-N is constant -> treat as 0 (no shared rank).
    if std_f < 1e-12 or std_c < 1e-12:
        tau = 0.0
        spear = 0.0
    else:
        tau_stat = kendalltau(r_clair, r_fixedn).statistic
        spear_stat = spearmanr(r_clair, r_fixedn).statistic
        tau = 0.0 if np.isnan(tau_stat) else float(tau_stat)
        spear = 0.0 if np.isnan(spear_stat) else float(spear_stat)

    clo, chi = _boot_total_ci(r_clair, b, seed)
    flo, fhi = _boot_total_ci(r_fixedn, b, seed + 7)
    ci_overlap = not (clo > fhi or flo > chi)

    n = r_clair.size
    k = max(1, int(round(top_frac * n)))
    set_c = set(np.argsort(r_clair)[::-1][:k].tolist())
    set_f = set(np.argsort(r_fixedn)[::-1][:k].tolist())
    union = set_c | set_f
    loc_jaccard = len(set_c & set_f) / len(union) if union else 1.0

    # Measurement power requires the clairvoyant regret to actually vary across prompts.
    # - std_c ~ 0  : clairvoyant cannot localize either -> NOT distinguished (degrade case).
    # - std_f ~ 0  : best-fixed-N is constant but clairvoyant varies -> distinguished.
    # - both vary  : distinguished iff rankings disagree AND clairvoyant has more spread.
    if std_c < 1e-12:
        k2 = False
    elif std_f < 1e-12:
        k2 = True
    else:
        k2 = (tau < theta2) and (std_ratio > 1.0)
    return MeasurementPower(
        std_ratio=std_ratio,
        spearman=spear,
        kendall_tau=tau,
        ci_overlap=ci_overlap,
        loc_jaccard=loc_jaccard,
        k2_distinguished=bool(k2),
    )
