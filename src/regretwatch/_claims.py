"""Honesty layer: claim constants + the ClairvoyantRegret type.

The clairvoyant compute-regret is a *non-causal* lower bound: no online stopping
policy can reach it, because it stops using knowledge of which draw will succeed.
To make it impossible to surface this number on its own (the R1 framing risk), the
value is wrapped in ``ClairvoyantRegret`` whose ``.render()`` *requires* the matched
achievable gap. There is no public API that returns a bare clairvoyant float.
"""

from __future__ import annotations

from dataclasses import dataclass

# Verbatim label prefixed to every clairvoyant display (asserted literally in tests).
CLAIRVOYANCE_LABEL = "non-causal matched-accuracy cost lower bound (no causal online policy can reach it)"

# The only positioning sentence we make about novelty.
CLAIM_POSITIONING = "post-hoc compute-regret instrument + honest achievable<->clairvoyant framing"

# Banned marketing-claim ERE used by the CI honest-marketing grep (grep -REn). Precise by
# design: catches promotional claims but NOT the ordinary word "first". The definition lines
# below carry "# honest:ok" so the guard does not flag itself (no content is whitelisted).
BANNED_CLAIM_ERE = (  # honest:ok
    r"state.of.the.art|\bSOTA\b|world.?s? first|"  # honest:ok
    r"the first (tool|library|framework|package|method|system|instrument|to |ever)|"  # honest:ok
    r"初の|fully automatic|permanent(ly)?|永続|outperform|"  # honest:ok
    r"\+[0-9]+(\.[0-9]+)?\s*%|invent(ed|s)? (the )?(renewal|prophet)"  # honest:ok
)

# Things we explicitly do NOT claim (mirrored by the CI honest-marketing grep).
NON_CLAIMS = (
    "regretwatch does not claim to have originated renewal-reward or prophet inequalities; "
    "it applies a clairvoyant (hindsight) lower bound as a grading oracle on existing logs.",
    "the clairvoyant regret is not achievable by any online policy; the actionable "
    "headline is the best-fixed-N excess-waste percentage.",
    "regretwatch is an offline measurement instrument; it does not rewrite stopping policies at runtime.",
)


@dataclass(frozen=True)
class ClairvoyantRegret:
    """A clairvoyant regret value that cannot be displayed without its achievable gap.

    Parameters
    ----------
    value:
        The non-causal clairvoyant compute-regret (>= 0).
    achievable_gap:
        ``(achievable_cost - clairvoyant_cost)`` -- the non-causal residual that an
        implementable (causal) policy cannot close. Required for any rendering.
    unit:
        Cost unit label (``"tokens"`` / ``"unit"`` / ``"wall_ms"``).
    """

    value: float
    achievable_gap: float | None
    unit: str = "unit"

    def render(self) -> str:
        """Render the clairvoyant number, always paired with the non-causal residual.

        Raises
        ------
        RuntimeError
            If the achievable gap is missing (``None``), so a raw clairvoyant value
            can never be emitted alone.
        """
        if self.achievable_gap is None:
            raise RuntimeError(
                "ClairvoyantRegret.render() requires achievable_gap: a clairvoyant "
                "regret may never be shown without its non-causal residual."
            )
        unit_note = " (cost unit = draws)" if self.unit == "unit" else ""
        return (
            f"{CLAIRVOYANCE_LABEL}: regret={self.value:.4g}{unit_note}; "
            f"non-causal residual (achievable - clairvoyant)={self.achievable_gap:.4g}"
        )
