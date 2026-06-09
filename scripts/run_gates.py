#!/usr/bin/env python3
"""S4 numeric gate: run G1-G8 + K1/K2/K3 on synthetic (+ real dump if present).

Writes ``results/v0.1.0a2_metrics.json`` with an env stamp. The build is allowed to ship
only if ``all(G1..G8)`` and the K-gate ship_mode is decided. No README number may be typed
by hand before this runs (see ``<!--MEASURED@S4-->``).

The real monkey_business dump is not bundled; when it is absent the previously recorded
real-dump block in the output file is carried forward (marked ``source: cached``) so a
regeneration never silently nulls the real measurement.

Usage::

    python scripts/run_gates.py [--real-npz PATH] [--out results/v0.1.0a2_metrics.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

import regretwatch as rw
from regretwatch import synth
from regretwatch._types import Agg, PromptSeq
from regretwatch.baselines import _prefix_cost, best_fixed_n, excess_waste_pct, realized_accuracy
from regretwatch.killgate import measurement_power
from regretwatch.oracle import realized_cost
from regretwatch.regret import per_prompt_regret

THETA1 = 0.8  # pre-registered (closeable-lever)
THETA2 = 0.7  # pre-registered (kendall-tau)


def _per_prompt_fixedn_regret(seqs: list[PromptSeq], agg: Agg) -> np.ndarray:
    nstar, _ = best_fixed_n(seqs, agg)
    return np.array([realized_cost(s) - _prefix_cost(s, nstar) for s in seqs])


def g1_calibration() -> dict:
    """Realized policy already optimal -> excess waste ~ 0 (matched-fixed) and |regret| ~ 0
    (realized == oracle)."""
    w = excess_waste_pct(synth.gen_matched_fixed(), Agg.BON)  # realized == best-fixed-N
    r = per_prompt_regret(synth.gen_known_optimal(), Agg.BON)  # realized == clairvoyant oracle
    excess_ok = abs(w.excess_waste_pct) < 1e-9
    regret_ok = float(np.abs(r).max()) < 1e-9
    return {
        "pass": bool(excess_ok and regret_ok),
        "excess_waste_pct": w.excess_waste_pct,
        "max_regret": float(np.abs(r).max()),
    }


def g2_oversample() -> dict:
    """Inject extra wasted draws -> excess waste monotone, recovered >= 0.95, Spearman=1."""
    extras = [0, 2, 4, 8, 16]
    pcts, recovered = [], []
    for e in extras:
        seqs = synth.gen_oversample(extra=e)
        w = excess_waste_pct(seqs, Agg.BON)
        pcts.append(w.excess_waste_pct)
        r = per_prompt_regret(seqs, Agg.BON)
        recovered.append(float(r.mean()))  # injected extra per prompt == e
    monotone = all(pcts[i] <= pcts[i + 1] + 1e-9 for i in range(len(pcts) - 1))
    # recovered[i] should equal extras[i]
    recov_ratio = min((recovered[i] / extras[i]) for i in range(1, len(extras)))
    from scipy.stats import spearmanr

    sp = float(spearmanr(extras, recovered).statistic)
    ok = monotone and recov_ratio >= 0.95 and sp > 0.999
    return {"pass": bool(ok), "monotone": monotone, "recover_ratio": recov_ratio, "spearman": sp}


def _k2_on(seqs: list[PromptSeq]) -> dict:
    agg = Agg.BON
    r_clair = per_prompt_regret(seqs, agg)
    r_fixed = _per_prompt_fixedn_regret(seqs, agg)
    mp = measurement_power(r_clair, r_fixed, theta2=THETA2)
    return {
        "std_ratio": mp.std_ratio,
        "kendall_tau": mp.kendall_tau,
        "k2_distinguished": mp.k2_distinguished,
        "loc_jaccard": mp.loc_jaccard,
        "std_clair": float(np.std(r_clair)),
        "std_fixed": float(np.std(r_fixed)),
    }


def g3_killgate(real_npz: str | None, cached_real: dict | None = None) -> dict:
    """Discriminator both ways + real-dump pre-measurement.

    If the external dump is absent but a ``cached_real`` block from a prior build is given,
    it is carried forward (annotated ``source``) so regeneration preserves provenance.
    """
    het = _k2_on(synth.gen_heterogeneous())  # should distinguish (True)
    hom = _k2_on(synth.gen_homogeneous())  # should NOT distinguish (False) -> degrade signal
    real = None
    if real_npz and Path(real_npz).exists():
        m = np.load(real_npz)["is_corrects"]  # (P, draws) bool
        seqs = []
        for i, row in enumerate(m):
            ic = row[:16]
            seqs.append(PromptSeq(f"mb::{i}", np.ones(ic.size), correct=ic.astype(bool)))
        real = _k2_on(seqs)
        real["realized_acc"] = realized_accuracy(seqs, Agg.BON)
        real["excess_waste_pct"] = excess_waste_pct(seqs, Agg.BON).excess_waste_pct
    elif cached_real is not None:
        real = {k: v for k, v in cached_real.items() if k != "source"}
        real["source"] = (
            "cached from prior build (real monkey_business dump absent at regeneration; not re-run)"
        )
    # G3 passes if the discriminator works both ways; real-dump (if present) must also distinguish.
    ok = het["k2_distinguished"] and (not hom["k2_distinguished"])
    if real is not None:
        ok = ok and bool(real["k2_distinguished"]) and (real["std_ratio"] > 1.0)
    return {"pass": bool(ok), "heterogeneous": het, "homogeneous": hom, "real_dump": real}


def g4_determinism() -> dict:
    """Same input + seed -> identical report JSON sha256."""
    seqs = synth.gen_oversample(extra=5)
    h = []
    for _ in range(2):
        res = rw.build_report(seqs, Agg.BON, b=500, seed=11)
        payload = {k: v for k, v in res.payload.items() if k != "caveats"}
        h.append(sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    return {"pass": h[0] == h[1], "sha256": h[0]}


def g5_scale_invariance() -> dict:
    """Multiply all costs by c -> excess_waste_pct invariant."""
    seqs = synth.gen_oversample(extra=6)
    base = excess_waste_pct(seqs, Agg.BON).excess_waste_pct
    scaled = excess_waste_pct(synth.scale_cost(seqs, 7.3), Agg.BON).excess_waste_pct
    ok = abs(base - scaled) < 1e-9
    return {"pass": bool(ok), "base": base, "scaled": scaled, "abs_diff": abs(base - scaled)}


def g6_noise_failclosed() -> dict:
    """Fail-closed under reward-model noise: low noise trusted, high noise refused.

    Uses a fragile small-regret scenario (realized barely sub-optimal). The clean regret
    is the reference; plausible label flips that shift it by more than rho_tol refuse the
    clairvoyant headline. Also checks the noise CI widens from no-noise to high-noise.
    """
    from regretwatch.regret import propagate_rm_noise, refuse_headline

    seqs = synth.gen_oversample(extra=1, n_prompts=80)  # fragile: clean regret == 1
    clean = float(per_prompt_regret(seqs, Agg.BON).mean())
    ci0 = propagate_rm_noise(seqs, Agg.BON, 0.0, b=500, seed=3)
    ci_lo_noise = propagate_rm_noise(seqs, Agg.BON, 0.03, b=500, seed=3)
    ci_hi_noise = propagate_rm_noise(seqs, Agg.BON, 0.4, b=500, seed=3)
    refuse_low = refuse_headline(ci_lo_noise, clean_mean=clean)
    refuse_high = refuse_headline(ci_hi_noise, clean_mean=clean)
    widened = (ci_hi_noise[1] - ci_hi_noise[0]) > (ci0[1] - ci0[0])
    ok = (not refuse_low) and refuse_high and widened
    return {
        "pass": bool(ok),
        "clean_regret": clean,
        "ci_low_noise": list(ci_lo_noise),
        "ci_high_noise": list(ci_hi_noise),
        "refuse_low_noise": bool(refuse_low),
        "refuse_high_noise": bool(refuse_high),
        "ci_widened": bool(widened),
    }


def g7_bucket_heterogeneity() -> dict:
    """Two buckets with different waste -> per-bucket excess waste separates."""
    seqs = synth.gen_two_bucket()
    from regretwatch.regret import aggregate_regret

    reg = aggregate_regret(seqs, Agg.BON, strata=("bucket",), b=500)
    means = {k: v.mean for k, v in reg.per_bucket.items()}
    keys = sorted(means)
    sep = abs(means[keys[0]] - means[keys[1]]) > 1.0 if len(keys) == 2 else False
    return {"pass": bool(sep), "per_bucket_mean": means}


def g8_framing() -> dict:
    """Clairvoyant only ever rendered with achievable gap; caveat field non-empty."""
    seqs = synth.gen_oversample(extra=6)
    res = rw.build_report(seqs, Agg.BON, b=300)
    caveat_ok = bool(res.payload.get("clairvoyance_caveat"))
    # render() must require achievable_gap
    raised = False
    try:
        rw.ClairvoyantRegret(value=1.0, achievable_gap=None, unit="unit").render()  # type: ignore[arg-type]
    except RuntimeError:
        raised = True
    label_ok = rw.CLAIRVOYANCE_LABEL in res.payload["clairvoyant_regret"]["rendered"]
    return {
        "pass": bool(caveat_ok and raised and label_ok),
        "caveat_nonempty": caveat_ok,
        "render_requires_gap": raised,
        "label_present": label_ok,
    }


def k_gate(g3: dict) -> dict:
    """K1/K2/K3 -> ship_mode (full / achievable-only / honest_synthetic_done)."""
    # K1: closeable lever exists (achievable < realized meaningfully); use oversample case.
    seqs = synth.gen_oversample(extra=6)
    gap = rw.achievable_gap(seqs, Agg.BON)
    clair_regret = gap.realized_total - gap.clairvoyant_total
    achievable_gap_val = gap.realized_total - gap.achievable_total
    k1_lever = (achievable_gap_val / clair_regret) if clair_regret > 0 else 0.0
    k1_ok = k1_lever >= THETA1  # closeable lever is a large fraction of total regret
    # K2: real-dump (or heterogeneous) distinguished
    k2 = g3["pass"]
    # K3: synthetic generator guarantees stopping is the cost driver (structural).
    k3_cost_driver = "stopping"
    if k2 and k1_ok:
        ship_mode = "full"
    elif k1_ok:
        ship_mode = "achievable-only"
    else:
        ship_mode = "honest_synthetic_done"
    return {
        "k1_lever": k1_lever,
        "k1_ok": bool(k1_ok),
        "k2_distinguished": bool(k2),
        "k3_cost_driver": k3_cost_driver,
        "ship_mode": ship_mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real-npz", default=str(Path.home() / "regretwatch/_oq3_tmp/mb_gsm8k_8b_iscorrects.npz")
    )
    ap.add_argument("--out", default="results/v0.1.0a2_metrics.json")
    args = ap.parse_args()

    # Preserve any previously recorded real-dump measurement so regenerating without the
    # external monkey_business dump does not silently null it (provenance robustness).
    cached_real = None
    out = Path(args.out)
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            rd = prev.get("gates", {}).get("G3", {}).get("real_dump")
            cached_real = rd if isinstance(rd, dict) else None
        except (json.JSONDecodeError, OSError):
            cached_real = None

    gates = {
        "G1": g1_calibration(),
        "G2": g2_oversample(),
        "G3": g3_killgate(args.real_npz, cached_real=cached_real),
        "G4": g4_determinism(),
        "G5": g5_scale_invariance(),
        "G6": g6_noise_failclosed(),
        "G7": g7_bucket_heterogeneity(),
        "G8": g8_framing(),
    }
    all_pass = all(g["pass"] for g in gates.values())
    kg = k_gate(gates["G3"])
    metrics = {
        "version": "0.1.0a2",
        "env": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__},
        "theta1_lever": THETA1,
        "theta2_kendall": THETA2,
        "gates": gates,
        "all_gates_pass": all_pass,
        "k_gate": kg,
        "ship_decision": (
            "SHIP"
            if all_pass and kg["ship_mode"] in ("full", "achievable-only", "honest_synthetic_done")
            else "HOLD"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"all_gates_pass={all_pass}  ship_mode={kg['ship_mode']}  ship={metrics['ship_decision']}")
    for name, g in gates.items():
        print(f"  {name}: {'PASS' if g['pass'] else 'FAIL'}")
    return 0 if metrics["ship_decision"] == "SHIP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
