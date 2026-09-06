#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep MaxWell annotation axes.

For each axis, train the head twice on the same feature cache - once with
plain frame averaging (the assignment's scheme) and once with temporal
features added - and report the gain. The gain is a measure of how much
temporal information that axis actually carries.

Nothing is overwritten: every run appends to runs/<axis>/results.json and the
summary is written to runs/axis_sweep.json.

Usage:
    python3 scripts/run_axis_sweep.py --feats ~/trkv/feats --labels-dir ~/trkv
    python3 scripts/run_axis_sweep.py --axes T-3 T-4 T-5 T-8 --branch r50
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# MaxWell axes and what they mean, from positive_descs in demo_maxvqa.py
AXES = {
    "O":     ("high quality",            "整体质量"),
    "A-all": ("good aesthetics",         "美学整体"),
    "T-1":   ("sharp",                   "锐利度"),
    "T-2":   ("in-focus",                "对焦"),
    "T-3":   ("noiseless",               "噪点"),
    "T-4":   ("clear-motion",            "运动模糊"),
    "T-5":   ("stable / shaky",          "抖动"),
    "T-6":   ("well-exposed",            "曝光"),
    "T-7":   ("original / compressed",   "压缩伪影"),
    "T-8":   ("fluent / choppy",         "卡顿"),
    "T-all": ("clear",                   "技术质量整体"),
}


def slug(axis):
    return axis.lower().replace("-", "")


def find_make_labels(labels_dir):
    """make_labels.py may live in scripts/, next to the labels, or in cwd."""
    for d in (HERE, labels_dir, os.getcwd()):
        p = os.path.join(d, "make_labels.py")
        if os.path.exists(p):
            return p
    raise SystemExit(
        "make_labels.py not found. Copy it into scripts/ or pass --labels-dir "
        "to the directory that holds it.")


def ensure_labels(axis, labels_dir):
    """Generate <axis>_train_lable_{train,test}.txt if missing."""
    pre = f"{slug(axis)}_"
    tr = os.path.join(labels_dir, f"{pre}train_lable_train.txt")
    te = os.path.join(labels_dir, f"{pre}train_lable_test.txt")
    if os.path.exists(tr) and os.path.exists(te):
        return tr, te
    print(f"  generating labels for {axis} ...")
    subprocess.run([sys.executable, find_make_labels(labels_dir),
                    "--col", axis, "--out-prefix", pre,
                    "--workdir", labels_dir], check=True)
    return tr, te


def train(feats, tr, te, agg, branch, out, epochs):
    cmd = [sys.executable, os.path.join(HERE, "train_head.py"),
           "--feats", feats, "--train-labels", tr, "--val-labels", te,
           "--agg", agg, "--branch", branch, "--out", out,
           "--epochs", str(epochs)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        return None
    tag = f"{branch}_{agg}"
    res = json.load(open(os.path.join(out, "results.json"), encoding="utf-8"))
    return res.get(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--labels-dir", required=True,
                    help="where make_labels.py writes its output")
    ap.add_argument("--axes", nargs="*", default=list(AXES),
                    help="which axes to sweep (default: all)")
    ap.add_argument("--branch", default="all", choices=["all", "r50", "vit"])
    ap.add_argument("--aggs", nargs="*", default=["mean", "mean+std+diff"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/axis_sweep.json")
    args = ap.parse_args()

    summary = {}
    if os.path.exists(args.out):                       # never discard old runs
        summary = json.load(open(args.out, encoding="utf-8"))

    for axis in args.axes:
        if axis not in AXES:
            print(f"unknown axis {axis}, skipping")
            continue
        en, zh = AXES[axis]
        print(f"\n=== {axis}  ({en} / {zh}) ===")
        tr, te = ensure_labels(axis, args.labels_dir)
        out_dir = os.path.join(args.runs, slug(axis))
        entry = summary.setdefault(axis, {"en": en, "zh": zh, "runs": {}})
        for agg in args.aggs:
            key = f"{args.branch}_{agg}"
            print(f"  training {key} ...", flush=True)
            r = train(args.feats, tr, te, agg, args.branch, out_dir, args.epochs)
            if r is None:
                print("    failed")
                continue
            score = (r["srocc"] + r["plcc"]) / 2
            entry["runs"][key] = {"srocc": r["srocc"], "plcc": r["plcc"],
                                  "score": score, "epoch": r["epoch"]}
            print(f"    SROCC {r['srocc']:.4f}  PLCC {r['plcc']:.4f}  "
                  f"Score {score:.4f}")
        json.dump(summary, open(args.out, "w"), indent=2, ensure_ascii=False)

    # ---- summary table, sorted by how much temporal features help
    base, full = f"{args.branch}_mean", f"{args.branch}_mean+std+diff"
    rows = []
    for axis, e in summary.items():
        b = e["runs"].get(base, {}).get("score")
        f = e["runs"].get(full, {}).get("score")
        if b is not None and f is not None:
            rows.append((axis, e["zh"], e["en"], b, f, f - b, (f / b - 1) * 100))
    rows.sort(key=lambda r: -r[5])

    print("\n\n| 轴 | 含义 | mean | mean+std+diff | 变化 | 相对 |")
    print("|---|---|---|---|---|---|")
    for a, zh, en, b, f, d, p in rows:
        print(f"| {a} | {zh}（{en}） | {b:.4f} | {f:.4f} | "
              f"**{d:+.4f}** | {p:+.1f}% |")
    print(f"\nsummary written to {args.out}")


if __name__ == "__main__":
    main()
