import pytest

from regretwatch import build_report, synth
from regretwatch._claims import CLAIRVOYANCE_LABEL, NON_CLAIMS, ClairvoyantRegret
from regretwatch._types import Agg


def test_render_requires_achievable_gap():
    cr = ClairvoyantRegret(value=5.0, achievable_gap=None, unit="unit")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        cr.render()


def test_render_includes_verbatim_label():
    cr = ClairvoyantRegret(value=5.0, achievable_gap=2.0, unit="unit")
    out = cr.render()
    assert CLAIRVOYANCE_LABEL in out
    assert "non-causal residual" in out


def test_report_has_caveat_field():
    seqs = synth.gen_oversample(extra=6)
    res = build_report(seqs, Agg.BON, b=200)
    assert res.payload["clairvoyance_caveat"] == CLAIRVOYANCE_LABEL
    assert res.payload["clairvoyance_caveat"]  # non-empty (G8)


def test_clairvoyant_always_paired_with_gap():
    seqs = synth.gen_oversample(extra=6)
    res = build_report(seqs, Agg.BON, b=200)
    cr = res.payload["clairvoyant_regret"]
    assert "noncausal_residual" in cr
    assert CLAIRVOYANCE_LABEL in cr["rendered"]


def test_no_clairvoyant_flag_omits_it():
    seqs = synth.gen_oversample(extra=6)
    res = build_report(seqs, Agg.BON, b=200, include_clairvoyant=False)
    assert "clairvoyant_regret" not in res.payload


def test_non_claims_present_in_caveats():
    seqs = synth.gen_oversample(extra=6)
    res = build_report(seqs, Agg.BON, b=200)
    for nc in NON_CLAIMS:
        assert nc in res.payload["caveats"]


def test_unit_cost_draws_note():
    seqs = synth.gen_oversample(extra=6)
    res = build_report(seqs, Agg.BON, b=200, cost_unit="unit")
    assert any("draw" in c for c in res.payload["caveats"])
