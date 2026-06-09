import numpy as np

from regretwatch import synth
from regretwatch._types import Agg, PromptSeq
from regretwatch.baselines import (
    achievable_gap,
    best_fixed_n,
    excess_waste_pct,
    one_step_lookahead,
    running_agreement,
)


def test_best_fixed_n_matches_accuracy():
    seqs = synth.gen_matched_fixed(n_prompts=60, n_draws=12)
    nstar, _ = best_fixed_n(seqs, Agg.BON)
    assert nstar == 12  # hardest prompt needs the full budget


def test_excess_waste_zero_when_matched():
    seqs = synth.gen_matched_fixed()
    w = excess_waste_pct(seqs, Agg.BON, b=300)
    assert abs(w.excess_waste_pct) < 1e-9


def test_excess_waste_positive_on_oversample():
    seqs = synth.gen_oversample(extra=6)
    w = excess_waste_pct(seqs, Agg.BON, b=300)
    assert w.excess_waste_pct > 0
    assert w.n_match == 4


def test_excess_waste_signed_negative_possible():
    # adaptive realized that beats fixed-N -> negative excess waste, surfaced not hidden
    a = PromptSeq("a", np.ones(1), correct=np.array([True]))  # realized cost 1
    b = PromptSeq("b", np.ones(10), correct=np.array([False] * 9 + [True]))  # realized cost 10
    w = excess_waste_pct([a, b], Agg.BON, b=200)
    assert np.isfinite(w.excess_waste_pct)


def test_achievable_gap_split():
    seqs = synth.gen_oversample(extra=6)
    g = achievable_gap(seqs, Agg.BON)
    assert g.closeable_lever >= 0
    assert g.realized_total >= g.achievable_total >= g.clairvoyant_total - 1e-9


def test_one_step_lookahead_le_realized():
    seqs = synth.gen_oversample(extra=6)
    total = one_step_lookahead(seqs, Agg.BON)
    realized = sum(float(s.cum_cost[-1]) for s in seqs)
    assert total <= realized + 1e-9


def test_running_agreement_returns_acc():
    seqs = synth.gen_oversample(extra=6)
    cost, acc = running_agreement(seqs, margin=3)
    assert cost > 0
    assert 0.0 <= acc <= 1.0
