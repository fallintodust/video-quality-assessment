#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct the MaxWell labels in the format used by the course assignment.

Sources (downloaded automatically from the ExplainableVQA repository):
  MaxWell_train.csv / MaxWell_val.csv          - 16 columns of scores, WITHOUT filenames
  examplar_data_labels/MaxWell/train_labels.txt
  examplar_data_labels/MaxWell/test_labels.txt - the filename order, line for line

Column-to-axis mapping (from demo_maxvqa.py, positive_descs):
  O     high quality            (overall quality)
  A-1   good content
  A-2   organized composition
  A-3   vibrant color
  A-4   contrastive lighting
  A-5   consistent trajectory
  A-all good aesthetics
  T-1   sharp
  T-2   in-focus
  T-3   noiseless
  T-4   clear-motion   <-> blurry-motion
  T-5   stable         <-> shaky        <-- shake / flicker
  T-6   well-exposed
  T-7   original       <-> compressed
  T-8   fluent         <-> choppy       <-- fluency (stutter)
  T-all clear

Usage:
  python make_labels.py --col T-5
  python make_labels.py --col T-8 --out-prefix fluent_

Downloads MaxWell_train.csv / MaxWell_val.csv and the filename-order
files from VQAssessment/ExplainableVQA, joins them line-by-line and
writes train_lable_train.txt / train_lable_test.txt (3634 / 909 lines).

Dataset: MaxWell (DIVIDE), Wu et al., "Towards Explainable In-the-Wild
Video Quality Assessment". Please cite the original work.
"""
import argparse
import csv
import os
import urllib.request

BASE = "https://ghfast.top/https://raw.githubusercontent.com/VQAssessment/ExplainableVQA/master"
FILES = {
    "MaxWell_train.csv": f"{BASE}/MaxWell_train.csv",
    "MaxWell_val.csv": f"{BASE}/MaxWell_val.csv",
    "MW_train_names.txt": f"{BASE}/examplar_data_labels/MaxWell/train_labels.txt",
    "MW_test_names.txt": f"{BASE}/examplar_data_labels/MaxWell/test_labels.txt",
}


def ensure_sources(workdir):
    for fname, url in FILES.items():
        path = os.path.join(workdir, fname)
        if not os.path.exists(path):
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(url, path)


def load_split(csv_path, names_path, col):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    names = [l.split(",")[0].strip() for l in open(names_path, encoding="utf-8") if l.strip()]
    if len(rows) != len(names):
        raise SystemExit(f"out of sync: {len(rows)} rows in the CSV, {len(names)} names")
    if col not in rows[0]:
        raise SystemExit(f"no column {col}; available: {list(rows[0])}")
    return list(zip(names, (float(r[col]) for r in rows)))


def rescale(v, lo, hi):
    """Linearly map [lo, hi] to [0, 100], clamped."""
    x = (v - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="T-5", help="which axis to export (default T-5, stable/shaky)")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--keep-ext", action="store_true", help="keep the .mp4 suffix in names")
    args = ap.parse_args()

    ensure_sources(args.workdir)
    p = lambda f: os.path.join(args.workdir, f)

    train = load_split(p("MaxWell_train.csv"), p("MW_train_names.txt"), args.col)
    val = load_split(p("MaxWell_val.csv"), p("MW_test_names.txt"), args.col)

    # Scale bounds are computed over the COMBINED set. Otherwise train and val
    # end up on different scales and PLCC comes out systematically lower.
    allv = [v for _, v in train + val]
    lo, hi = min(allv), max(allv)
    print(f"axis {args.col}: raw range [{lo:.4f}, {hi:.4f}] -> [0, 100]")

    for split, data in (("train", train), ("test", val)):
        out = p(f"{args.out_prefix}train_lable_{split}.txt")
        with open(out, "w", encoding="utf-8") as fh:
            for name, v in data:
                if not args.keep_ext:
                    name = os.path.splitext(name)[0]
                fh.write(f"{name}: {rescale(v, lo, hi):.1f}\n")
        print(f"wrote {len(data):5d} lines -> {out}")


if __name__ == "__main__":
    main()
