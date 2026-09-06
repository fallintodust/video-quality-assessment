#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the regression head on cached features.

The backbones are frozen and only the head is trained, so an epoch takes
seconds and the whole ablation table fits into one evening.

Temporal aggregation modes (--agg):
    mean        average over frames - the scheme from the assignment, baseline
    std         spread over time
    diff        mean absolute difference between ADJACENT frames within a clip
                - a direct indicator of flicker
Combine them with '+':  --agg mean+diff  /  --agg mean+std+diff

Usage:
    python3 train_head.py --agg mean                 # baseline
    python3 train_head.py --agg mean+std+diff        # full variant
    python3 train_head.py --agg mean --branch r50    # without the ViT part
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

CONV_DIM = 13568          # ResNet-50 part
VIT_DIM = 2304            # ViT part


# ---------------------------------------------------------------- aggregation

def aggregate(seq, modes, clip_len):
    """seq: [T, D] float32 -> a vector of size len(modes)*D."""
    out = []
    for m in modes:
        if m == "mean":
            out.append(seq.mean(0))
        elif m == "std":
            out.append(seq.std(0))
        elif m == "diff":
            # differences WITHIN a clip only: frames from different clips are
            # far apart, so their difference reflects a scene change, not flicker
            T = seq.shape[0]
            d = seq.reshape(T // clip_len, clip_len, -1)
            out.append(np.abs(np.diff(d, axis=1)).mean((0, 1)))
        else:
            raise ValueError(f"unknown aggregation mode: {m}")
    return np.concatenate(out)


def load_labels(path):
    """Keys are stored without the extension: on main the labels are written as
    '0001.mp4: 1.213', while the feature cache lives in files named '0001.npy'."""
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.rsplit(":", 1)
        d[os.path.splitext(k.strip())[0]] = float(v)
    return d


def build_matrix(names, feats_dir, modes, clip_len, branch):
    X = None
    for i, n in enumerate(names):
        seq = np.load(os.path.join(feats_dir, f"{n}.npy")).astype(np.float32)
        if branch == "r50":
            seq = seq[:, :CONV_DIM]
        elif branch == "vit":
            seq = seq[:, CONV_DIM:]
        v = aggregate(seq, modes, clip_len)
        if X is None:
            X = np.empty((len(names), v.shape[0]), dtype=np.float32)
        X[i] = v
        if (i + 1) % 500 == 0:
            print(f"  loaded {i+1}/{len(names)}", flush=True)
    return X


class Head(nn.Module):
    def __init__(self, d_in, hidden=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def evaluate(model, X, y, dev, bs=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev)
            preds.append(model(xb).cpu().numpy())
    p = np.concatenate(preds)
    return spearmanr(p, y).correlation, pearsonr(p, y)[0], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="./feats")
    ap.add_argument("--train-labels", default="./train_lable_train.txt")
    ap.add_argument("--val-labels", default="./train_lable_test.txt")
    ap.add_argument("--agg", default="mean", help="mean / std / diff, joined by '+'")
    ap.add_argument("--branch", choices=["all", "r50", "vit"], default="all")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", default="./runs")
    args = ap.parse_args()

    modes = args.agg.split("+")
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.branch}_{args.agg}"

    tr_lab, va_lab = load_labels(args.train_labels), load_labels(args.val_labels)
    have = {f[:-4] for f in os.listdir(args.feats) if f.endswith(".npy")}
    tr_names = sorted(set(tr_lab) & have)
    va_names = sorted(set(va_lab) & have)
    print(f"train {len(tr_names)} / val {len(va_names)}  ({len(have)} in cache)")
    if len(tr_names) < 50:
        raise SystemExit("too few features - has the caching pass finished?")

    t0 = time.time()
    print("building train...")
    Xtr = build_matrix(tr_names, args.feats, modes, args.clip_len, args.branch)
    print("building val...")
    Xva = build_matrix(va_names, args.feats, modes, args.clip_len, args.branch)

    ytr_raw = np.array([tr_lab[n] for n in tr_names], np.float32)
    yva_raw = np.array([va_lab[n] for n in va_names], np.float32)
    # label scales differ (1~5 on main, 0~100 in the T-* axis variant), so we
    # normalise using train statistics; this does not affect SROCC/PLCC, both
    # of which are invariant under a linear transform
    y_mu, y_sd = ytr_raw.mean(), ytr_raw.std() + 1e-6
    ytr = (ytr_raw - y_mu) / y_sd
    yva = (yva_raw - y_mu) / y_sd
    print(f"labels: train {ytr_raw.min():.2f}..{ytr_raw.max():.2f}, "
          f"normalised with mu={y_mu:.3f} sd={y_sd:.3f}")
    print(f"matrices ready in {time.time()-t0:.0f} s, dimension {Xtr.shape[1]}")

    # standardise using train stats - with 15k-45k features the head will not
    # converge without it
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Head(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    best = {"obj": -9}

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        srocc, plcc, _ = evaluate(model, Xva, yva, dev)
        obj = srocc + plcc
        flag = ""
        if obj > best["obj"]:
            best = {"obj": float(obj), "srocc": float(srocc),
                    "plcc": float(plcc), "epoch": ep}
            torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                        "y_mu": float(y_mu), "y_sd": float(y_sd),
                        "args": vars(args)},
                       os.path.join(args.out, f"best_{tag}.pt"))
            flag = "  <-- best"
        if ep % 5 == 0 or flag:
            print(f"ep {ep:3d}  loss {tot/len(perm):.4f}  "
                  f"SROCC {srocc:.4f}  PLCC {plcc:.4f}  OBJ {obj:.4f}{flag}")

    print(f"\n=== {tag} ===")
    print(f"best epoch {best['epoch']}: "
          f"SROCC {best['srocc']:.4f}  PLCC {best['plcc']:.4f}")
    print(f"Score (before the time penalty) = {(best['srocc']+best['plcc'])/2:.4f}")

    res_path = os.path.join(args.out, "results.json")
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    allres[tag] = best
    json.dump(allres, open(res_path, "w"), indent=2, ensure_ascii=False)
    print(f"results appended to {res_path}")


if __name__ == "__main__":
    main()
