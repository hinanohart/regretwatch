"""Adapters that turn external eval logs into ``regretwatch.schema.v1`` PromptSeqs.

Primary (v0.1):
    * :mod:`regretwatch.adapters.monkey_business` -- Large Language Monkeys dumps.
    * :mod:`regretwatch.adapters.generic_csv` -- the most general CSV/JSONL path.

Stubs (optional ``[adapters]`` extra, v0.2 real-dump reconciliation):
    * :mod:`regretwatch.adapters.lm_eval`, ``verl``, ``simple_evals``.
"""

from .generic_csv import load_csv
from .monkey_business import load_monkey_business

__all__ = ["load_csv", "load_monkey_business"]
