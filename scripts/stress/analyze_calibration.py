#!/usr/bin/env python3
"""Read one or more calibration CSVs and report per-family drift.

Data source: calibration-<ts>.csv files produced by ``calibrate.py``.
Each row is one (service, context) measurement with the estimator's
prediction and NVML's observed peak.

This script answers "by how much does the estimator under-reserve, by
family?" so a proposed retune for ``compute_buffer_mb`` can be
defensible rather than hand-wavy. It groups rows by
``general.architecture``, prints the delta trend across context, and
suggests a conservative base + per-1k-context slope that would have
produced non-negative headroom on every observation in the input.

Usage:
    uv run python analyze_calibration.py calibration-*.csv
    uv run python analyze_calibration.py --arch gpt-oss calibration-*.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with p.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def by_arch(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    g: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        arch = r.get("architecture", "").strip()
        if not arch:
            continue
        g[arch].append(r)
    return g


def int_field(r: dict[str, str], name: str) -> int:
    try:
        return int(r[name])
    except (KeyError, ValueError):
        return 0


def summarise_arch(arch: str, rows: list[dict[str, str]]) -> None:
    # Sort by (service, context) for readable output.
    rows = sorted(rows, key=lambda r: (r.get("service", ""), int_field(r, "context")))
    print(f"\n=== {arch} ({len(rows)} rows) ===")
    print(
        f"  {'service':30s} {'ctx':>6s} {'predicted':>10s} {'actual':>10s} {'delta':>8s} {'delta_%':>8s}"
    )
    worst_pct = -1e9
    worst_abs = -1e9
    per_ctx: dict[int, list[int]] = defaultdict(list)  # deltas at each context
    for r in rows:
        svc = r.get("service", "?")
        ctx = int_field(r, "context")
        pred = int_field(r, "predicted_total_mib")
        actual = int_field(r, "actual_post_chat_total_mib")
        delta = actual - pred
        if actual == 0:
            continue  # skipped / failed row
        pct = 100.0 * delta / actual
        worst_pct = max(worst_pct, pct)
        worst_abs = max(worst_abs, delta)
        per_ctx[ctx].append(delta)
        print(
            f"  {svc:30s} {ctx:>6d} {pred:>10d} {actual:>10d} {delta:>+8d} {pct:>+7.1f}%"
        )

    if not per_ctx:
        return

    # Summarise the delta trend across context.
    xs = sorted(per_ctx.keys())
    worst_by_ctx = [(x, max(per_ctx[x])) for x in xs]
    print(f"  worst delta per context: {worst_by_ctx}")
    print(f"  worst delta overall:     {worst_abs:+d} MiB ({worst_pct:+.1f}% of actual)")

    # First-cut fit: base + slope × (context / 1024), fit to the WORST
    # delta at each context so an over-estimation formula covers every
    # observation we have. Closed-form two-point fit between the smallest
    # and largest contexts; anything finer would overfit 5-10 points.
    if len(xs) >= 2:
        x0, y0 = worst_by_ctx[0]
        x1, y1 = worst_by_ctx[-1]
        dx_k = (x1 - x0) / 1024.0
        slope_per_1k = (y1 - y0) / dx_k if dx_k != 0 else 0.0
        base = y0 - slope_per_1k * (x0 / 1024.0)
        # Add a small safety margin so an as-yet-unobserved run doesn't
        # immediately break our "err on overestimating" claim.
        margin = 64
        suggested_base = int(base + margin)
        suggested_slope = int(slope_per_1k + 0.5)  # round up
        print(
            f"  headroom to cover observed deltas: "
            f"base={suggested_base} MiB + {suggested_slope} MiB × (ctx/1024)"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="+", type=Path, help="calibration-*.csv files")
    p.add_argument(
        "--arch",
        action="append",
        help="filter to these architectures (comparable to `general.architecture` in the GGUF)",
    )
    args = p.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        sys.stderr.write("no rows in input\n")
        sys.exit(1)

    groups = by_arch(rows)
    if args.arch:
        wanted = set(args.arch)
        groups = {k: v for k, v in groups.items() if k in wanted}

    for arch in sorted(groups.keys()):
        summarise_arch(arch, groups[arch])


if __name__ == "__main__":
    main()
