"""Assemble the audit report. Honesty is enforced structurally, not by convention.

The clairvoyant regret only ever reaches output through :class:`ClairvoyantRegret`, which
requires the achievable gap, and ``clairvoyance_caveat`` is always populated. The headline
is the causal best-fixed-N excess-waste percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._claims import CLAIM_POSITIONING, CLAIRVOYANCE_LABEL, NON_CLAIMS, ClairvoyantRegret
from ._types import Agg, PromptSeq
from .baselines import (
    achievable_gap,
    excess_waste_pct,
    one_step_lookahead,
    running_agreement,
)
from .oracle import clairvoyant_cost, realized_cost, realized_outcome
from .regret import aggregate_regret, propagate_rm_noise, refuse_headline
from .votecheck import cant_win_flags


@dataclass(frozen=True)
class AuditResult:
    payload: dict[str, Any]


def _top_waste_prompts(seqs: list[PromptSeq], agg: Agg, k: int = 10) -> list[dict[str, Any]]:
    regrets = []
    for s in seqs:
        y = realized_outcome(s, agg)
        regrets.append((s.prompt_id, realized_cost(s) - clairvoyant_cost(s, y, agg)))
    regrets.sort(key=lambda t: t[1], reverse=True)
    return [{"prompt_id": pid, "clairvoyant_regret": float(r)} for pid, r in regrets[:k]]


def build_report(
    seqs: list[PromptSeq],
    agg: Agg = Agg.BON,
    *,
    strata: tuple[str, ...] | None = None,
    cost_unit: str = "unit",
    rm_noise_eta: float | None = None,
    b: int = 2000,
    seed: int = 0,
    include_clairvoyant: bool = True,
    experimental: bool = False,
) -> AuditResult:
    """Produce the audit payload with the best-fixed-N headline and honest framing."""
    waste = excess_waste_pct(seqs, agg, b=b, seed=seed)
    gap = achievable_gap(seqs, agg)
    reg = aggregate_regret(seqs, agg, strata=strata, b=b, seed=seed)
    b3_cost, b3_acc = (
        running_agreement(seqs) if all(s.correct is not None for s in seqs) else (float("nan"), float("nan"))
    )

    caveats: list[str] = list(NON_CLAIMS)
    if cost_unit == "unit":
        caveats.append(
            "cost unit = draws (no per-sample token count in source); percentages are "
            "in draw-count terms, not token economics."
        )

    payload: dict[str, Any] = {
        "schema": "regretwatch.report.v1",
        "positioning": CLAIM_POSITIONING,
        "agg": agg.value,
        "n_prompts": len(seqs),
        "cost_unit": cost_unit,
        "headline_excess_waste_pct": waste.excess_waste_pct,
        "headline_ci": [waste.ci_lo, waste.ci_hi],
        "n_match_recommended": waste.n_match,
        "matched_accuracy": waste.matched_accuracy,
        "realized_total_cost": waste.realized_total,
        "bestfixedN_total_cost": waste.bestfixedN_total,
        "one_step_lookahead_total": one_step_lookahead(seqs, agg),
        "running_agreement_total": b3_cost,
        "running_agreement_acc": b3_acc,
        "achievable_gap": {
            "realized_total": gap.realized_total,
            "achievable_total": gap.achievable_total,
            "clairvoyant_total": gap.clairvoyant_total,
            "closeable_lever": gap.closeable_lever,
            "noncausal_residual": gap.noncausal_residual,
        },
        "per_bucket": {
            key: {"mean_regret": r.mean, "ci": [r.ci_lo, r.ci_hi], "n_prompts": r.n_prompts}
            for key, r in reg.per_bucket.items()
        },
        "top_waste_prompts": _top_waste_prompts(seqs, agg),
        "clairvoyance_caveat": CLAIRVOYANCE_LABEL,
        "caveats": caveats,
    }

    if include_clairvoyant:
        refuse = False
        if rm_noise_eta is not None:
            ci = propagate_rm_noise(seqs, agg, rm_noise_eta, b=b, seed=seed)
            refuse = refuse_headline(ci, clean_mean=reg.mean)
            payload["rm_noise_eta"] = rm_noise_eta
            payload["rm_noise_regret_ci"] = list(ci)
        if refuse:
            payload["clairvoyant_regret"] = None
            payload["clairvoyant_status"] = "REFUSED: RM-noise-dominated"
            caveats.append("clairvoyant headline refused: dominated by reward-model noise.")
        else:
            cr = ClairvoyantRegret(value=reg.mean, achievable_gap=gap.noncausal_residual, unit=cost_unit)
            payload["clairvoyant_regret"] = {
                "mean": cr.value,
                "ci": [reg.ci_lo, reg.ci_hi],
                "noncausal_residual": cr.achievable_gap,
                "rendered": cr.render(),  # render() requires achievable_gap -> honesty by type
            }

    if experimental:
        flags = cant_win_flags(seqs)
        payload["experimental"] = {
            "majority_cant_win_rate": float(flags.mean()) if flags.size else 0.0,
            "n_flagged": int(flags.sum()),
            "note": (
                "EXPERIMENTAL (v0.2): uncalibrated majority can't-win detector; not part of "
                "the headline. Requires answer ids (reports 0 otherwise)."
            ),
        }
    return AuditResult(payload=payload)


def to_json(result: AuditResult) -> dict[str, Any]:
    return result.payload


def to_markdown(result: AuditResult) -> str:
    p = result.payload
    pct = p["headline_excess_waste_pct"]
    lo, hi = p["headline_ci"]
    unit = p["cost_unit"]
    lines = [
        "# regretwatch audit",
        "",
        f"_{p['positioning']}_",
        "",
        f"**Headline — excess compute waste vs best-fixed-N:** {pct * 100:.2f}% "
        f"(95% CI {lo * 100:.2f}% .. {hi * 100:.2f}%)",
        "",
        f"- aggregation: `{p['agg']}`  |  prompts: {p['n_prompts']}  |  cost unit: `{unit}`",
        f"- recommended fixed budget N_match = **{p['n_match_recommended']}** "
        f"(matches realized accuracy {p['matched_accuracy']:.4f})",
        f"- realized total: {p['realized_total_cost']:.0f}  |  best-fixed-N total: "
        f"{p['bestfixedN_total_cost']:.0f}",
    ]
    if unit == "unit":
        lines.append("- _cost unit = draws; percentages are draw-count terms, not token economics._")
    cr = p.get("clairvoyant_regret")
    lines += ["", "## Clairvoyant vs achievable (non-causal lower bound, paired)"]
    if cr is None:
        lines.append(f"- {p.get('clairvoyant_status', 'clairvoyant omitted')}")
    else:
        g = p["achievable_gap"]
        lines += [
            f"- {cr['rendered']}",
            f"- closeable lever (realized - achievable) = {g['closeable_lever']:.0f}",
            f"- non-causal residual (achievable - clairvoyant) = {g['noncausal_residual']:.0f} "
            "(no causal online policy can close this)",
        ]
    if p["per_bucket"]:
        lines += ["", "## Per-bucket", "", "| bucket | mean regret | 95% CI | n |", "|---|---|---|---|"]
        for key, r in p["per_bucket"].items():
            c = r["ci"]
            lines.append(f"| {key} | {r['mean_regret']:.2f} | [{c[0]:.2f}, {c[1]:.2f}] | {r['n_prompts']} |")
    lines += ["", "## Caveats"]
    lines += [f"- {c}" for c in p["caveats"]]
    return "\n".join(lines) + "\n"
