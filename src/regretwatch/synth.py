"""Deterministic synthetic log generators (the only bundled "data").

Every generator is seeded and reproducible. These drive the sensitivity gates G1-G8 and
the K1/K2 measurement-power discriminator without shipping any external dataset.
"""

from __future__ import annotations

import numpy as np

from ._types import PromptSeq


def _bon_prompt(
    pid: str,
    first_correct: int | None,
    n_draws: int,
    *,
    cost: float = 1.0,
    bucket: dict[str, str] | None = None,
) -> PromptSeq:
    """A best-of-N prompt whose first correct draw is at ``first_correct`` (1-indexed).

    ``first_correct=None`` means never correct within ``n_draws``.
    """
    correct = np.zeros(n_draws, dtype=bool)
    if first_correct is not None and 1 <= first_correct <= n_draws:
        correct[first_correct - 1 :] = True  # once solvable, later draws also correct
    costs = np.full(n_draws, cost, dtype=float)
    return PromptSeq(pid, costs, correct=correct, bucket=bucket or {})


def gen_known_optimal(n_prompts: int = 60, n_draws: int = 16, *, seed: int = 0) -> list[PromptSeq]:
    """G1: realized policy already stops optimally (first correct == only needed draw).

    Here every prompt is solved exactly at the last realized draw, so a policy that
    matched the oracle would have zero excess waste.
    """
    rng = np.random.default_rng(seed)
    seqs = []
    for i in range(n_prompts):
        fc = int(rng.integers(1, n_draws + 1))
        # realized == oracle: truncate so the realized last draw is the first-correct draw
        seqs.append(_bon_prompt(f"opt::{i}", fc, fc))
    return seqs


def gen_matched_fixed(n_prompts: int = 60, n_draws: int = 12, *, seed: int = 0) -> list[PromptSeq]:
    """G1: realized policy already equals best-fixed-N (uniform budget) -> excess waste 0.

    All prompts share the realized budget ``n_draws``; first-correct draws span
    ``1..n_draws`` with at least one prompt requiring the full budget, so ``N* == n_draws``
    and realized cost equals the best-fixed-N cost exactly.
    """
    seqs = []
    for i in range(n_prompts):
        fc = (i % n_draws) + 1  # guarantees some prompts have fc == n_draws
        seqs.append(_bon_prompt(f"mf::{i}", fc, n_draws))
    return seqs


def gen_oversample(extra: int, n_prompts: int = 60, base_first: int = 4, *, seed: int = 0) -> list[PromptSeq]:
    """G2: solved at ``base_first`` then padded with ``extra`` wasted draws.

    Clairvoyant regret per prompt == extra (recover >= 0.95 of injected waste expected).
    """
    return [_bon_prompt(f"over::{i}", base_first, base_first + extra) for i in range(n_prompts)]


def gen_heterogeneous(
    n_easy: int = 40, n_hard: int = 40, realized_n: int = 16, *, seed: int = 0
) -> list[PromptSeq]:
    """K2 discriminating: easy (first-correct 1-2) + hard (first-correct 8-20).

    Best-fixed-N must hold N high to keep the hard prompts, reporting ~0 savings, while
    the clairvoyant oracle localizes the waste onto the easy prompts -> the two disagree.
    """
    rng = np.random.default_rng(seed)
    seqs = []
    for i in range(n_easy):
        fc = int(rng.integers(1, 3))
        seqs.append(_bon_prompt(f"easy::{i}", fc, realized_n, bucket={"diff": "easy"}))
    for i in range(n_hard):
        draw = int(rng.integers(8, 21))
        fc_hard: int | None = draw if draw <= realized_n else None  # unsolved within budget
        seqs.append(_bon_prompt(f"hard::{i}", fc_hard, realized_n, bucket={"diff": "hard"}))
    return seqs


def gen_homogeneous(
    n_prompts: int = 80, first_correct: int = 5, realized_n: int = 16, *, seed: int = 0
) -> list[PromptSeq]:
    """K2 control: all prompts first-correct at the same draw -> clairvoyant == fixed-N."""
    return [_bon_prompt(f"hom::{i}", first_correct, realized_n) for i in range(n_prompts)]


def gen_noisy(
    p_flip: float, n_prompts: int = 80, base_first: int = 4, realized_n: int = 16, *, seed: int = 0
) -> list[PromptSeq]:
    """G6: oversample scenario with per-draw label flips at rate ``p_flip``."""
    rng = np.random.default_rng(seed)
    seqs = []
    for i in range(n_prompts):
        base = _bon_prompt(f"noisy::{i}", base_first, realized_n)
        flip = rng.random(realized_n) < p_flip
        noisy = np.logical_xor(base.correct, flip)  # type: ignore[arg-type]
        seqs.append(PromptSeq(base.prompt_id, base.cost, correct=noisy))
    return seqs


def gen_two_bucket(realized_n: int = 16, *, seed: int = 0) -> list[PromptSeq]:
    """G7: two buckets with different waste levels (per-bucket excess waste must separate)."""
    a = [_bon_prompt(f"A::{i}", 2, realized_n, bucket={"bucket": "A"}) for i in range(40)]
    b = [_bon_prompt(f"B::{i}", 10, realized_n, bucket={"bucket": "B"}) for i in range(40)]
    return a + b


def scale_cost(seqs: list[PromptSeq], c: float) -> list[PromptSeq]:
    """G5: multiply every cost by ``c`` (excess-waste percent must be invariant)."""
    return [
        PromptSeq(
            s.prompt_id,
            s.cost * c,
            correct=s.correct,
            score=s.score,
            answer=s.answer,
            gold=s.gold,
            bucket=s.bucket,
        )
        for s in seqs
    ]
