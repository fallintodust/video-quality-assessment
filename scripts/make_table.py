#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render runs/results.json as a markdown table for the README / report.

Usage:
    python3 scripts/make_table.py
    python3 scripts/make_table.py --out results_table.md
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="./runs/results.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.exists(args.json):
        raise SystemExit(f"not found: {args.json}")
    res = json.load(open(args.json, encoding="utf-8"))
    if not res:
        raise SystemExit("results.json is empty")

    rows = []
    for tag, r in res.items():
        branch, agg = tag.split("_", 1)
        rows.append((branch, agg, r["srocc"], r["plcc"],
                     (r["srocc"] + r["plcc"]) / 2, r["epoch"]))
    rows.sort(key=lambda x: -x[4])

    lines = [
        "| Branch | Temporal aggregation | SROCC | PLCC | Score | Best epoch |",
        "|---|---|---|---|---|---|",
    ]
    for b, a, s, p, sc, e in rows:
        lines.append(f"| {b} | {a} | {s:.4f} | {p:.4f} | **{sc:.4f}** | {e} |")

    table = "\n".join(lines)
    print(table)
    print(f"\nbest: {rows[0][0]} / {rows[0][1]}  Score {rows[0][4]:.4f}")

    if args.out:
        open(args.out, "w", encoding="utf-8").write(table + "\n")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
