import numpy as np

from regretwatch import synth
from regretwatch._types import Agg
from regretwatch.baselines import best_fixed_n
from regretwatch.killgate import measurement_power
from regretwatch.oracle import realized_cost
from regretwatch.regret import (
    aggregate_regret,
    per_prompt_regret,
    propagate_rm_noise,
    refuse_headline,
)


def _fixedn_regret(seqs, agg):
    nstar, _ = best_fixed_n(seqs, agg)
    from regretwatch.baselines import _prefix_cost

    return np.array([realized_cost(s) - _prefix_cost(s, nstar) for s in seqs])


def test_regret_nonnegative():
    seqs = synth.gen_oversample(extra=6)
    r = per_prompt_regret(seqs, Agg.BON)
    assert (r >= 0).all()


def test_aggregate_regret_ci_brackets_mean():
    seqs = synth.gen_oversample(extra=6)
    rep = aggregate_regret(seqs, Agg.BON, b=500)
    assert rep.ci_lo <= rep.mean <= rep.ci_hi


def test_aggregate_regret_per_bucket():
    seqs = synth.gen_two_bucket()
    rep = aggregate_regret(seqs, Agg.BON, strata=("bucket",), b=300)
    assert set(rep.per_bucket) == {"bucket=A", "bucket=B"}


def test_k2_distinguished_heterogeneous():
    seqs = synth.gen_heterogeneous()
    mp = measurement_power(per_prompt_regret(seqs, Agg.BON), _fixedn_regret(seqs, Agg.BON))
    assert mp.k2_distinguished is True
    assert mp.std_ratio > 1.0


def test_k2_not_distinguished_homogeneous():
    seqs = synth.gen_homogeneous()
    mp = measurement_power(per_prompt_regret(seqs, Agg.BON), _fixedn_regret(seqs, Agg.BON))
    assert mp.k2_distinguished is False  # clairvoyant also constant -> no localization power


def test_rm_noise_widens_and_refuses():
    seqs = synth.gen_oversample(extra=1, n_prompts=80)
    clean = float(per_prompt_regret(seqs, Agg.BON).mean())
    lo = propagate_rm_noise(seqs, Agg.BON, 0.03, b=400, seed=3)
    hi = propagate_rm_noise(seqs, Agg.BON, 0.4, b=400, seed=3)
    assert refuse_headline(lo, clean_mean=clean) is False
    assert refuse_headline(hi, clean_mean=clean) is True


def test_refuse_when_ci_crosses_zero():
    assert refuse_headline((-0.1, 2.0)) is True
