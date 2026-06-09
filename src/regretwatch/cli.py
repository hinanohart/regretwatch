"""``rw`` command-line interface: validate, audit, demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._types import Agg
from .io import load_logs, majority_uses_collapse, validate
from .report import build_report, to_json, to_markdown


def _audit(args: argparse.Namespace) -> int:
    if not validate(args.log):
        print("validation failed (fail-closed); fix the log and retry.", file=sys.stderr)
        return 2
    seqs = load_logs(args.log)
    agg = Agg(args.agg)
    if agg is Agg.MAJORITY and majority_uses_collapse(seqs):
        print(
            "note: majority aggregation without answer ids -> using collapse worst-case lower bound.",
            file=sys.stderr,
        )
    strata = tuple(args.bucket.split(",")) if args.bucket else None
    result = build_report(
        seqs,
        agg,
        strata=strata,
        cost_unit=args.cost_unit,
        rm_noise_eta=args.rm_noise_eta,
        b=args.bootstrap,
        include_clairvoyant=not args.no_clairvoyant,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(result), encoding="utf-8")
    pct = result.payload["headline_excess_waste_pct"]
    print(f"excess_waste_pct vs best-fixed-N = {pct * 100:.2f}%  ->  {out}/report.md")
    return 0


def _validate(args: argparse.Namespace) -> int:
    ok = validate(args.log)
    print("VALID" if ok else "INVALID")
    return 0 if ok else 1


def _demo(args: argparse.Namespace) -> int:
    from . import synth

    seqs = synth.gen_oversample(extra=6)
    result = build_report(seqs, Agg.BON, cost_unit="unit")
    print(to_markdown(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rw",
        description="regretwatch: offline post-hoc compute-regret instrument for "
        "best-of-N / self-consistency stopping policies.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="audit a schema.v1 JSONL log")
    a.add_argument("log")
    a.add_argument("--out", default="report")
    a.add_argument("--agg", default="bon", choices=[m.value for m in Agg])
    a.add_argument("--bucket", default="", help="comma-separated bucket keys for stratification")
    a.add_argument("--cost-unit", default="unit", choices=["tokens", "unit", "wall_ms"])
    a.add_argument("--rm-noise-eta", type=float, default=None)
    a.add_argument("--bootstrap", type=int, default=2000)
    a.add_argument("--no-clairvoyant", action="store_true")
    a.add_argument("--experimental", action="store_true", help="enable v0.2 experimental checks")
    a.set_defaults(func=_audit)

    v = sub.add_parser("validate", help="validate a schema.v1 JSONL log (fail-closed)")
    v.add_argument("log")
    v.set_defaults(func=_validate)

    d = sub.add_parser("demo", help="run a synthetic demo audit")
    d.set_defaults(func=_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
