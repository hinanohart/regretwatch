# Changelog

## 0.1.0a2 (2026-06-10)

Post-release hardening pass (3-agent re-audit after a context compaction). No published
number changes — the synthetic gates G1–G8 and the real monkey_business dump figures are
identical; this release adds robustness, honesty, and the experimental wiring the first
alpha left unconnected.

- **Ragged-log fix** (`baselines.best_fixed_n`): the best-fixed-N budget sweep now runs to
  the *longest* prompt instead of the shortest. On unequal-length logs the old cap let
  `N*` silently default to a budget that did not match realized accuracy (inflating the
  headline and recommending an accuracy-halving `N`). Uniform-length logs — every shipped
  result — are unaffected. Regression test added.
- **`--experimental` wired** (`cli` → `report` → `votecheck`): the flag now actually
  surfaces the uncalibrated majority "can't-win" rate in a separate `experimental` payload
  field (still out of the headline). It was previously parsed but read nowhere; `votecheck`
  gains direct test coverage.
- **draw_order uniqueness enforced** (`io.validate` / `io.load_logs`): a duplicate
  `draw_order` within a prompt is now rejected fail-closed, matching the documented schema
  contract (it was silently accepted before).
- **Provenance robustness** (`scripts/run_gates.py`): when the (unbundled) real dump is
  absent, the prior real-dump block is carried forward marked `source: cached` instead of
  being nulled, so regeneration never destroys the committed real measurement. README now
  states the Real Dump row requires the external dump.

## 0.1.0a1 (2026-06-09)

First alpha. Offline post-hoc compute-regret instrument for best-of-N / self-consistency /
RLVR-rollout stopping policies.

- **Core**: O(N) prefix-scan clairvoyant matched-accuracy oracle (`oracle.py`) for `bon`,
  `majority` (true vote with answer ids, collapse worst-case lower bound otherwise), and
  `verifier_rerank`; ties fail closed. This is a prefix scan, **not** a renewal-reward
  fixed point.
- **Headline**: best-fixed-N excess-waste percentage with prompt-cluster bootstrap CI
  (`baselines.py`), a causal/implementable comparator. Signed (adaptive policies beating
  every fixed N report negative). Clairvoyant regret is second-tier and only ever rendered
  paired with its non-causal achievable gap (`ClairvoyantRegret`, `_claims.py`).
- **KILL-GATE** (`killgate.py`): measurement power of the clairvoyant oracle vs best-fixed-N.
- **Baselines**: best-fixed-N, 1-step-lookahead, label-free running-agreement.
- **Adapters**: `monkey_business` (primary) and `generic_csv`; `lm_eval` / `verl` /
  `simple_evals` stubs (v0.2).
- **Sensitivity gates G1–G8** + `scripts/run_gates.py`; validated on a real
  monkey_business dump (GSM8K/Llama-3-8B).
- **Experimental** (`votecheck.py`, `--experimental`, v0.2): majority "can't-win" detection.

### Honest implementation note (R17⑤ — discovered while running the gates)

The frozen architecture's RM-noise fail-closed criterion (refuse when the bootstrap CI of
the regret crosses 0 or its relative width exceeds `rho_tol`) turned out to be a near-dead
branch in practice: with many prompts the bootstrap CI of the *mean* concentrates (CLT), so
its relative width rarely exceeds the threshold, and discrete label flips tend to *inflate*
clipped regret rather than collapse it. `refuse_headline` was therefore augmented with the
load-bearing criterion actually used: **refuse when plausible label flips shift the regret
estimate by more than `rho_tol` of its noise-free value.** The original CI-crosses-zero and
relative-width checks are retained. This is a faithfulness-preserving refinement of §7, not
a scope change.
