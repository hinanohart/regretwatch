import numpy as np
import pytest

from regretwatch._types import Agg, PromptSeq
from regretwatch.oracle import _agg_correct, clairvoyant_cost, realized_cost, realized_outcome


def _seq(correct, cost=None, **kw):
    correct = np.array(correct, dtype=bool)
    cost = np.ones(correct.size) if cost is None else np.asarray(cost, float)
    return PromptSeq("p", cost, correct=correct, **kw)


def test_p1_bon_monotone_in_k():
    # bon outcome never decreases as k grows
    s = _seq([False, False, True, False, True])
    vals = [_agg_correct(s, k, Agg.BON) for k in range(1, 6)]
    assert vals == sorted(vals)
    assert vals == [0, 0, 1, 1, 1]


def test_p2_majority_nonmonotone_true_vote():
    # true majority over answer ids is non-monotone: [1,0,0,0,1] (arch s1) -- a wrong
    # answer transiently leads, so no percentile cutoff suffices; a full prefix scan is needed.
    s = PromptSeq(
        "p",
        np.ones(5),
        correct=np.array([True, False, False, True, True]),
        answer=np.array(["A", "B", "B", "A", "A"]),
        gold="A",
    )
    vals = [_agg_correct(s, k, Agg.MAJORITY) for k in range(1, 6)]
    assert vals == [1, 0, 0, 0, 1]
    assert vals != sorted(vals)


def test_collapse_majority_also_nonmonotone():
    s = _seq([True, False, False])
    vals = [_agg_correct(s, k, Agg.MAJORITY) for k in range(1, 4)]
    assert vals == [1, 0, 0]
    assert vals != sorted(vals)


def test_p3_ties_fail_closed():
    # even split -> 2*sum == k -> strict majority is 0
    s = _seq([True, False])
    assert _agg_correct(s, 2, Agg.MAJORITY) == 0


def test_p4_oracle_le_realized_fuzz():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(1, 12))
        s = _seq(rng.random(n) < 0.4, cost=rng.uniform(0.5, 3.0, n))
        for agg in (Agg.BON, Agg.MAJORITY):
            y = realized_outcome(s, agg)
            assert clairvoyant_cost(s, y, agg) <= realized_cost(s) + 1e-9


def test_oracle_first_correct_bon():
    s = _seq([False, False, True, True], cost=[2, 3, 4, 5])
    y = realized_outcome(s, Agg.BON)  # 1
    assert clairvoyant_cost(s, y, Agg.BON) == pytest.approx(2 + 3 + 4)  # cum cost to first correct


def test_oracle_unreachable_failclosed():
    # realized never correct -> y=0 -> any prefix matches -> oracle = cost of 1 draw
    s = _seq([False, False, False], cost=[1, 1, 1])
    y = realized_outcome(s, Agg.BON)
    assert y == 0
    assert clairvoyant_cost(s, y, Agg.BON) == pytest.approx(1.0)


def test_verifier_rerank_uses_argmax():
    s = PromptSeq("p", np.ones(3), correct=np.array([False, True, False]), score=np.array([0.1, 0.9, 0.5]))
    assert _agg_correct(s, 3, Agg.VERIFIER_RERANK) == 1  # argmax score idx=1 is correct


def test_true_majority_with_answers():
    s = PromptSeq(
        "p", np.ones(3), correct=np.array([True, True, False]), answer=np.array(["72", "72", "9"]), gold="72"
    )
    assert _agg_correct(s, 3, Agg.MAJORITY) == 1


def test_majority_requires_signal():
    s = PromptSeq("p", np.ones(2), score=np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        _agg_correct(s, 2, Agg.MAJORITY)
