"""EXPERIMENTAL (v0.2): majority-vote "can't win" region detection.

Flags prompts where, by draw ``k``, the leading answer's anytime upper-confidence share
has already dropped below 0.5, so no continuation can produce a majority winner -- i.e.
further sampling is provably wasted under the majority rule. Gated behind ``--experimental``
and never part of the headline: the false-continue-rate calibration is out of v0.1 scope.
See README for the contrast with Certified Self-Consistency.
"""

from __future__ import annotations

import numpy as np

from ._types import PromptSeq


def cant_win_flags(seqs: list[PromptSeq], *, delta: float = 0.05, kmax: int | None = None) -> np.ndarray:
    """Per-prompt bool: did the prompt enter a provable majority "can't win" region?

    Uses an anytime union Hoeffding bound on the leading-answer share. EXPERIMENTAL --
    calibration of the false-continue rate is deferred to v0.2.
    """
    flags = np.zeros(len(seqs), dtype=bool)
    for j, seq in enumerate(seqs):
        if seq.answer is None:
            continue
        n = seq.n_draws if kmax is None else min(kmax, seq.n_draws)
        for k in range(2, n + 1):
            ans = seq.answer[:k]
            _, counts = np.unique(ans, return_counts=True)
            phat = counts.max() / k
            ucb = phat + np.sqrt(np.log(max(n, 2) / delta) / (2 * k))
            if ucb < 0.5:
                flags[j] = True
                break
    return flags
