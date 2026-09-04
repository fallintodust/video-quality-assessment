#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a "temporal consistency" target that MaxWell does not provide directly.

The assignment asks for 时域一致性 (temporal consistency) with 闪烁 (flicker)
scenes, but MaxWell has no such axis. Its two temporal axes are:

    T-5  stable / shaky    - camera shake, seconds-scale
    T-8  fluent / choppy   - stutter, frame-scale

Both are contaminated by general picture quality: a shaky video also tends to
be rated blurrier and noisier, so a model can score well on T-5 partly by
reading frame-level cues that have nothing to do with time.

This script removes that contamination. It regresses the temporal mix on the
purely frame-level axes and keeps the residual - the part of "shaky/choppy"
that the single-frame axes cannot explain. That residual is as close to a
clean temporal-consistency target as this dataset allows.

Frame-level (spatial) axes used as regressors:
    T-1 sharp, T-2 in-focus, T-3 noiseless, T-6 well-exposed, T-7 original
(T-4 clear-motion is left out: motion blur is itself partly temporal.)

Usage:
    python3 scripts/make_temporal_target.py --workdir ~/trkv
    python3 scripts/make_temporal_target.py --workdir ~/trkv --mix T-8
    python3 scripts/make_temporal_target.py --workdir ~/trkv --no-residual
"""
import argparse
import csv
import os
import urllib.request

import numpy as np

BASE = ("https://ghfast.top/https://raw.githubusercontent.com/"
        "VQAssessment/ExplainableVQA/master")
FILES = {
    "MaxWell_train.csv": f"{BASE}/MaxWell_train.csv",
    "MaxWell_val.csv": f"{BASE}/MaxWell_val.csv",
    "MW_train_names.txt": f"{BASE}/examplar_data_labels/MaxWell/train_labels.txt",
    "MW_test_names.txt": f"{BASE}/examplar_data_labels/MaxWell/test_labels.txt",
}

SPATIAL = ["T-1", "T-2", "T-3", "T-6", "T-7"]


def ensure_sources(workdir):
    for fname, url in FILES.items():
        p = os.path.join(workdir, fname)
        if not os.path.exists(p):
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(url, p)


def load_split(csv_path, names_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    names = [l.split(",")[0].strip()
             for l in open(names_path, encoding="utf-8") if l.strip()]
    if len(rows) != len(names):
        raise SystemExit(f"out of sync: {len(rows)} rows, {len(names)} names")
    return names, rows


def col(rows, name):
    return np.array([float(r[name]) for r in rows], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--mix", nargs="*", default=["T-5", "T-8"],
                    help="temporal axes to combine (averaged after z-scoring)")
    ap.add_argument("--spatial", nargs="*", default=SPATIAL,
                    help="frame-level axes to regress out")
    ap.add_argument("--no-residual", action="store_true",
                    help="just average the temporal axes, skip residualisation")
    ap.add_argument("--out-prefix", default="tcons_")
    args = ap.parse_args()

    ensure_sources(args.workdir)
    p = lambda f: os.path.join(args.workdir, f)
    ntr, rtr = load_split(p("MaxWell_train.csv"), p("MW_train_names.txt"))
    nva, rva = load_split(p("MaxWell_val.csv"), p("MW_test_names.txt"))

    def mix(rows, mu=None, sd=None):
        cols = [col(rows, a) for a in args.mix]
        if mu is None:
            mu = [c.mean() for c in cols]
            sd = [c.std() + 1e-9 for c in cols]
        z = [(c - m) / s for c, m, s in zip(cols, mu, sd)]
        return np.mean(z, axis=0), mu, sd

    ytr, mu_m, sd_m = mix(rtr)
    yva, _, _ = mix(rva, mu_m, sd_m)
    print(f"temporal mix: {' + '.join(args.mix)} (z-scored, averaged)")

    if not args.no_residual:
        # design matrix from the frame-level axes, with an intercept
        def design(rows):
            return np.column_stack([np.ones(len(rows))] +
                                   [col(rows, a) for a in args.spatial])
        Xtr, Xva = design(rtr), design(rva)
        # fit on train only, apply to both - the val labels must not depend
        # on val statistics
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        r2 = 1 - ((ytr - Xtr @ beta) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum()
        print(f"regressed out: {' '.join(args.spatial)}")
        print(f"  R^2 of the frame-level axes on the temporal mix: {r2:.3f}")
        print(f"  -> {r2*100:.0f}% of 'shaky/choppy' is explainable from "
              f"single-frame quality; the residual is what is left")
        ytr = ytr - Xtr @ beta
        yva = yva - Xva @ beta

    # map to 0..100 using train range, clip val into the same scale
    lo, hi = ytr.min(), ytr.max()
    print(f"raw residual range on train: [{lo:.3f}, {hi:.3f}] -> [0, 100]")

    def write(names, y, split):
        out = p(f"{args.out_prefix}train_lable_{split}.txt")
        with open(out, "w", encoding="utf-8") as fh:
            for n, v in zip(names, y):
                s = float(np.clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0))
                fh.write(f"{os.path.splitext(n)[0]}: {s:.1f}\n")
        print(f"wrote {len(names):5d} lines -> {out}")

    write(ntr, ytr, "train")
    write(nva, yva, "test")

    # how different is this target from the raw axes?
    from scipy.stats import spearmanr
    print("\nSpearman correlation of the new target with the raw axes (train):")
    for a in args.mix + args.spatial + ["O"]:
        print(f"  {a:6s} {spearmanr(ytr, col(rtr, a)).correlation:+.3f}")


if __name__ == "__main__":
    main()
