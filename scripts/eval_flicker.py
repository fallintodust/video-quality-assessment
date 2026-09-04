#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark flicker detectors against the T-5 (stable/shaky) labels.

Both detectors output "severity" (1 = severe flicker) while the T-5 label is
"stability" (high = stable), so the two are expected to be anti-correlated.
The script reports SROCC/PLCC of -score against the label, i.e. a positive
number means the detector agrees with the human ratings.

The model detector reads the cached features, so it costs nothing extra.
The heuristic has to decode video, which dominates the runtime.

Usage:
    python3 scripts/eval_flicker.py --videos ~/trkv/data --feats ~/trkv/feats \
        --labels ~/trkv/train_lable_test.txt --ckpt runs/t5/best_all_mean+std+diff.pt
    python3 scripts/eval_flicker.py ... --limit 200        # quick check
    python3 scripts/eval_flicker.py ... --skip-model       # heuristic only
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_feats import clip_indices  # noqa: E402
from flicker_detectors import heuristic_flicker_v2  # noqa: E402


def load_labels(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.rsplit(":", 1)
        d[os.path.splitext(k.strip())[0]] = float(v)
    return d


def read_frames(path, n_clips, clip_len, size, uniform=False):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), width=size, height=size, num_threads=1)
    n = len(vr)
    if uniform:
        idx = np.clip(np.linspace(0, n - 1, n_clips * clip_len).round().astype(int),
                      0, n - 1)
    else:
        idx = clip_indices(n, n_clips, clip_len)
    return vr.get_batch(idx).asnumpy()


def report(name, scores, labels, elapsed, n):
    """scores: higher = more distorted. labels: higher = more stable."""
    s = -np.asarray(scores)
    y = np.asarray(labels)
    srocc = spearmanr(s, y).correlation
    plcc = pearsonr(s, y)[0]
    print(f"\n=== {name} ===")
    print(f"SROCC {srocc:.4f}   PLCC {plcc:.4f}   Score {(srocc+plcc)/2:.4f}")
    print(f"{elapsed/n*1000:.1f} ms/video   ({elapsed:.1f} s total)")
    return {"srocc": float(srocc), "plcc": float(plcc),
            "score": float((srocc + plcc) / 2), "ms_per_video": elapsed / n * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True, help="T-5 validation labels")
    ap.add_argument("--feats", default="", help="feature cache (for the model)")
    ap.add_argument("--ckpt", default="", help="T-5 head checkpoint")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--model-live", action="store_true",
                    help="also run the model on freshly decoded frames, "
                         "clips vs uniform (answers the integration question)")
    ap.add_argument("--skip-heuristic", action="store_true")
    ap.add_argument("--out", default="runs/flicker_eval.json")
    args = ap.parse_args()

    lab = load_labels(args.labels)
    files = sorted(f for f in os.listdir(args.videos)
                   if f.lower().endswith(".mp4")
                   and os.path.splitext(f)[0] in lab)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} videos with labels")
    results = {}

    # ---- heuristic: needs pixels, so it decodes
    if not args.skip_heuristic:
        for uniform in (False, True):
            tag = "heuristic_v2 (uniform frames)" if uniform else "heuristic_v2 (clips)"
            sc, y, t0, bad = [], [], time.time(), 0
            for i, f in enumerate(files, 1):
                try:
                    fr = read_frames(os.path.join(args.videos, f), args.clips,
                                     args.clip_len, args.size, uniform)
                    r = heuristic_flicker_v2(
                        fr, clip_len=None if uniform else args.clip_len)
                    sc.append(r["score"])
                    y.append(lab[os.path.splitext(f)[0]])
                except Exception:
                    bad += 1
                if i % 200 == 0:
                    print(f"  {tag}: {i}/{len(files)}", flush=True)
            if bad:
                print(f"  skipped {bad} unreadable videos")
            results[tag] = report(tag, sc, y, time.time() - t0, len(sc))

    # ---- model on freshly extracted features, both sampling schemes.
    # This is the integration question: diagnose.py hands the detector
    # uniformly sampled frames, but the head was trained on clips.
    if args.model_live and args.ckpt:
        from flicker_detectors import ModelFlicker
        det = ModelFlicker(args.ckpt)
        for uniform in (False, True):
            tag = ("model (live, uniform frames)" if uniform
                   else "model (live, clips)")
            sc, y, t0, bad = [], [], time.time(), 0
            for i, f in enumerate(files, 1):
                try:
                    fr = read_frames(os.path.join(args.videos, f), args.clips,
                                     args.clip_len, args.size, uniform)
                    pred = det.predict_from_features(det.features(fr))
                    sc.append(1.0 - (pred - det.lo) / (det.hi - det.lo))
                    y.append(lab[os.path.splitext(f)[0]])
                except Exception:
                    bad += 1
                if i % 100 == 0:
                    print(f"  {tag}: {i}/{len(files)}", flush=True)
            if bad:
                print(f"  skipped {bad}")
            results[tag] = report(tag, sc, y, time.time() - t0, len(sc))

    # ---- model: reads the cache, no decoding
    if not args.skip_model:
        if not (args.feats and args.ckpt):
            print("\n(model skipped: pass --feats and --ckpt)")
        else:
            from flicker_detectors import ModelFlicker
            det = ModelFlicker(args.ckpt)
            sc, y, t0, bad = [], [], time.time(), 0
            for i, f in enumerate(files, 1):
                stem = os.path.splitext(f)[0]
                p = os.path.join(args.feats, f"{stem}.npy")
                if not os.path.exists(p):
                    bad += 1
                    continue
                pred = det.predict_from_features(np.load(p).astype(np.float32))
                sc.append(1.0 - (pred - det.lo) / (det.hi - det.lo))
                y.append(lab[stem])
                if i % 200 == 0:
                    print(f"  model: {i}/{len(files)}", flush=True)
            if bad:
                print(f"  {bad} videos missing from the cache")
            results["model (cached features)"] = report(
                "model (cached features)", sc, y, time.time() - t0, len(sc))
            print("  note: timing excludes decoding and backbone forward "
                  "(~1.4 s/video when run end to end)")

    if results:
        import json
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nwritten to {args.out}")
        print("\n| detector | SROCC | PLCC | Score | ms/video |")
        print("|---|---|---|---|---|")
        for k, v in sorted(results.items(), key=lambda kv: -kv[1]["score"]):
            print(f"| {k} | {v['srocc']:.4f} | {v['plcc']:.4f} | "
                  f"**{v['score']:.4f}** | {v['ms_per_video']:.1f} |")


if __name__ == "__main__":
    main()
