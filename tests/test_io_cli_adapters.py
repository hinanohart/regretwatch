import json
import subprocess
import sys

import pytest

from regretwatch._types import Agg
from regretwatch.adapters import load_csv, load_monkey_business
from regretwatch.io import load_logs, majority_uses_collapse, validate
from regretwatch.report import build_report


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_validate_ok(tmp_path):
    p = tmp_path / "ok.jsonl"
    _write(
        p,
        [
            {"prompt_id": "a", "draw_order": 0, "cost": 1.0, "correct": True},
            {"prompt_id": "a", "draw_order": 1, "cost": 2.0, "correct": False},
        ],
    )
    assert validate(p) is True


def test_validate_rejects_bad_cost(tmp_path):
    p = tmp_path / "bad.jsonl"
    _write(p, [{"prompt_id": "a", "draw_order": 0, "cost": 0, "correct": True}])
    assert validate(p) is False


def test_validate_rejects_missing_signal(tmp_path):
    p = tmp_path / "bad2.jsonl"
    _write(p, [{"prompt_id": "a", "draw_order": 0, "cost": 1.0}])
    assert validate(p) is False


def test_load_logs_sorts_by_draw_order(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(
        p,
        [
            {"prompt_id": "a", "draw_order": 2, "cost": 3.0, "correct": True},
            {"prompt_id": "a", "draw_order": 0, "cost": 1.0, "correct": False},
            {"prompt_id": "a", "draw_order": 1, "cost": 2.0, "correct": False},
        ],
    )
    seqs = load_logs(p)
    assert len(seqs) == 1
    assert list(seqs[0].cost) == [1.0, 2.0, 3.0]


def test_validate_rejects_duplicate_draw_order(tmp_path):
    p = tmp_path / "dup.jsonl"
    _write(
        p,
        [
            {"prompt_id": "a", "draw_order": 0, "cost": 1.0, "correct": True},
            {"prompt_id": "a", "draw_order": 0, "cost": 2.0, "correct": False},
        ],
    )
    assert validate(p) is False  # draw_order documented unique-within-prompt -> fail closed


def test_load_logs_raises_on_duplicate_draw_order(tmp_path):
    p = tmp_path / "dup.jsonl"
    _write(
        p,
        [
            {"prompt_id": "a", "draw_order": 1, "cost": 1.0, "correct": True},
            {"prompt_id": "a", "draw_order": 1, "cost": 2.0, "correct": False},
        ],
    )
    with pytest.raises(ValueError):
        load_logs(p)


def test_cli_experimental_flag_surfaces_field(tmp_path):
    p = tmp_path / "log.jsonl"
    rows = []
    for i in range(12):
        for d in range(6):
            rows.append(
                {
                    "prompt_id": f"p{i}",
                    "draw_order": d,
                    "cost": 1.0,
                    "correct": d >= 2,
                    "answer": "A" if d >= 2 else f"x{d}",
                }
            )
    _write(p, rows)
    out = tmp_path / "rep"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "regretwatch.cli",
            "audit",
            str(p),
            "--out",
            str(out),
            "--experimental",
            "--bootstrap",
            "100",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads((out / "report.json").read_text())
    assert "experimental" in report  # --experimental must actually do something (not a dead flag)
    assert "majority_cant_win_rate" in report["experimental"]


def test_majority_collapse_detected(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"prompt_id": "a", "draw_order": 0, "cost": 1.0, "correct": True}])
    seqs = load_logs(p)
    assert majority_uses_collapse(seqs) is True  # no answer ids


def test_generic_csv(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(
        "prompt_id,draw_order,cost,correct\na,0,1.0,1\na,1,2.0,0\nb,0,1.0,0\nb,1,1.0,1\n",
        encoding="utf-8",
    )
    seqs = load_csv(p)
    assert len(seqs) == 2


def test_monkey_business_adapter(tmp_path):
    p = tmp_path / "mb.json"
    p.write_text(
        json.dumps(
            [
                {"is_corrects": [False, False, True, True], "orig_dset_idx": 7},
                {"is_corrects": [True, True], "orig_dset_idx": 9},
            ]
        ),
        encoding="utf-8",
    )
    seqs = load_monkey_business(p, realized_n=4)
    assert len(seqs) == 2
    assert seqs[0].prompt_id == "mb::7"
    assert seqs[0].correct is not None
    assert list(seqs[0].correct) == [False, False, True, True]


def test_cli_audit_end_to_end(tmp_path):
    p = tmp_path / "log.jsonl"
    rows = []
    for i in range(20):
        for d in range(8):
            rows.append(
                {
                    "prompt_id": f"p{i}",
                    "draw_order": d,
                    "cost": 1.0,
                    "correct": d >= 3,
                    "bucket": {"task": "x"},
                }
            )
    _write(p, rows)
    out = tmp_path / "rep"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "regretwatch.cli",
            "audit",
            str(p),
            "--out",
            str(out),
            "--bucket",
            "task",
            "--bootstrap",
            "200",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads((out / "report.json").read_text())
    assert "headline_excess_waste_pct" in report
    assert report["clairvoyance_caveat"]


def test_cli_validate_and_demo(tmp_path):
    p = tmp_path / "log.jsonl"
    _write(p, [{"prompt_id": "a", "draw_order": 0, "cost": 1.0, "correct": True}])
    rv = subprocess.run(
        [sys.executable, "-m", "regretwatch.cli", "validate", str(p)], capture_output=True, text=True
    )
    assert rv.returncode == 0 and "VALID" in rv.stdout
    rd = subprocess.run([sys.executable, "-m", "regretwatch.cli", "demo"], capture_output=True, text=True)
    assert rd.returncode == 0 and "excess compute waste" in rd.stdout


def test_build_report_runs():
    from regretwatch import synth

    res = build_report(synth.gen_oversample(extra=4), Agg.BON, b=100)
    assert res.payload["n_prompts"] == 60
