"""EXPERIMENTAL majority can't-win detector (votecheck.cant_win_flags) coverage."""

import numpy as np

from regretwatch._types import PromptSeq
from regretwatch.votecheck import cant_win_flags


def _seq(pid, answers):
    n = len(answers)
    return PromptSeq(pid, np.ones(n), correct=np.zeros(n, dtype=bool), answer=np.array(answers))


def test_clear_majority_not_flagged():
    seq = _seq("maj", ["A"] * 20)  # unanimous leader -> can always still win
    assert cant_win_flags([seq]).tolist() == [False]


def test_dispersed_answers_flagged():
    # 5 distinct answers, ~20 each over 100 draws -> leading share ~0.2, anytime UCB < 0.5.
    answers = [f"v{i % 5}" for i in range(100)]
    assert cant_win_flags([_seq("disp", answers)]).tolist() == [True]


def test_no_answer_ids_not_flagged():
    seq = PromptSeq("noans", np.ones(4), correct=np.array([False, False, True, True]))
    assert cant_win_flags([seq]).tolist() == [False]
