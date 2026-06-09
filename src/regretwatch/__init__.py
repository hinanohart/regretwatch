"""regretwatch -- offline post-hoc compute-regret instrument.

Grades how much test-time compute an already-deployed best-of-N / self-consistency /
verl-rollout stopping policy wasted, versus a clairvoyant (hindsight) matched-accuracy
lower bound, with the causal best-fixed-N excess-waste percentage as the headline.

This is a measurement instrument: it reports on existing logs and does not rewrite any
runtime policy. The clairvoyant regret is a non-causal lower bound and is never surfaced
without its achievable gap (see :mod:`regretwatch._claims`).
"""

from __future__ import annotations

from ._claims import CLAIRVOYANCE_LABEL, ClairvoyantRegret
from ._types import Agg, PromptSeq
from .baselines import achievable_gap, best_fixed_n, excess_waste_pct
from .io import load_logs, validate
from .killgate import measurement_power
from .oracle import clairvoyant_cost
from .regret import aggregate_regret, per_prompt_regret
from .report import AuditResult, build_report, to_json, to_markdown

__version__ = "0.1.0a1"

__all__ = [
    "Agg",
    "PromptSeq",
    "ClairvoyantRegret",
    "CLAIRVOYANCE_LABEL",
    "load_logs",
    "validate",
    "clairvoyant_cost",
    "per_prompt_regret",
    "aggregate_regret",
    "best_fixed_n",
    "excess_waste_pct",
    "achievable_gap",
    "measurement_power",
    "build_report",
    "to_json",
    "to_markdown",
    "AuditResult",
    "__version__",
]
