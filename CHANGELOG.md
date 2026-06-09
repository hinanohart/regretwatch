# Changelog

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
