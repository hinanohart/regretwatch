"""Exercise the S4 sensitivity gates G1-G8 inside pytest (CI guard)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_gates", Path(__file__).resolve().parents[1] / "scripts" / "run_gates.py"
)
assert _spec and _spec.loader
run_gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_gates)


def test_g1_calibration():
    assert run_gates.g1_calibration()["pass"] is True


def test_g2_oversample_recovery():
    assert run_gates.g2_oversample()["pass"] is True


def test_g3_killgate_discriminates_both_ways():
    g3 = run_gates.g3_killgate(real_npz=None)  # synthetic-only (CI has no real dump)
    assert g3["heterogeneous"]["k2_distinguished"] is True
    assert g3["homogeneous"]["k2_distinguished"] is False


def test_g4_determinism():
    assert run_gates.g4_determinism()["pass"] is True


def test_g5_scale_invariance():
    assert run_gates.g5_scale_invariance()["pass"] is True


def test_g6_noise_failclosed():
    assert run_gates.g6_noise_failclosed()["pass"] is True


def test_g7_bucket_heterogeneity():
    assert run_gates.g7_bucket_heterogeneity()["pass"] is True


def test_g8_framing():
    assert run_gates.g8_framing()["pass"] is True


@pytest.mark.parametrize("name", ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"])
def test_each_gate_exposed(name):
    # all eight gate functions exist (no silently-dropped gate)
    fnmap = {
        "G1": "g1_calibration",
        "G2": "g2_oversample",
        "G3": "g3_killgate",
        "G4": "g4_determinism",
        "G5": "g5_scale_invariance",
        "G6": "g6_noise_failclosed",
        "G7": "g7_bucket_heterogeneity",
        "G8": "g8_framing",
    }
    assert hasattr(run_gates, fnmap[name])
