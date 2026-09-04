#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference: a folder of videos -> score.txt

Loads a head checkpoint produced by train_head.py, runs the frozen backbones
over each video and writes the predicted quality scores.

The checkpoint stores every setting the head was trained with (aggregation
mode, branch, clip layout, feature and label normalisation), so nothing has
to be passed twice - the flags below only control input/output and speed.

Usage:
    python3 scripts/predict.py --videos ./test_videos --ckpt runs/t5/best_all_mean.pt
    python3 scripts/predict.py --videos ./test_videos --ckpt runs/o/best_all_mean.pt --scale-0-100
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_feats import Extractor, clip_indices, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from train_head import Head, aggregate, CONV_DIM  # noqa: E402


def read_frames(path, n_clips, clip_len, size):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), width=size, height=size, num_threads=1)
    idx = clip_indices(len(vr), n_clips, clip_len)
    frames = vr.get_batch(idx).asnumpy()
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.)
    x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
        torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with test videos")
    ap.add_argument("--ckpt", required=True, help="runs/<...>/best_<tag>.pt")
    ap.add_argument("--out", default="score.txt")
    ap.add_argument("--chunk", type=int, default=16, help="frames per forward (VRAM)")
    ap.add_argument("--keep-ext", action="store_true",
                    help="write names as 'video1.mp4' instead of 'video1'")
    ap.add_argument("--scale-0-100", action="store_true",
                    help="linearly rescale predictions into 0..100 "
                         "(use when the head was trained on a 1..5 label axis)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    modes = ta["agg"].split("+")
    branch = ta["branch"]
    clip_len = ta["clip_len"]
    n_clips = ta.get("clips", 4)
    size = ta.get("size", 224)
    use_vit = branch != "r50"

    print(f"checkpoint: {args.ckpt}")
    print(f"  aggregation {ta['agg']} | branch {branch} | "
          f"{n_clips} clips x {clip_len} frames")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Extractor(use_vit=use_vit).to(dev)

    mu, sd = ck["mu"], ck["sd"]
    head = Head(len(mu)).to(dev)
    head.load_state_dict(ck["state"])
    head.eval()
    y_mu, y_sd = ck["y_mu"], ck["y_sd"]

    files = sorted(f for f in os.listdir(args.videos)
                   if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    if not files:
        raise SystemExit(f"no videos found in {args.videos}")
    print(f"{len(files)} videos to score")

    t0 = time.time()
    results, failed = [], []
    with torch.no_grad():
        for i, name in enumerate(files, 1):
            stem = name if args.keep_ext else os.path.splitext(name)[0]
            try:
                x = read_frames(os.path.join(args.videos, name),
                                n_clips, clip_len, size).to(dev)
                outs = []
                for j in range(0, x.shape[0], args.chunk):
                    with torch.autocast(dev, dtype=torch.float16,
                                        enabled=(dev == "cuda")):
                        outs.append(extractor(x[j:j + args.chunk]).float())
                seq = torch.cat(outs).cpu().numpy()
                if branch == "r50":
                    seq = seq[:, :CONV_DIM]
                elif branch == "vit":
                    seq = seq[:, CONV_DIM:]
                v = aggregate(seq.astype(np.float32), modes, clip_len)
                v = (v - mu) / sd
                pred = head(torch.from_numpy(v[None]).float().to(dev)).item()
                score = pred * y_sd + y_mu          # back to the label scale
            except Exception as e:
                print(f"  !! {name}: {e}")
                failed.append(stem)
                score = y_mu                        # fall back to the mean
            results.append((stem, score))
            if i % 20 == 0 or i == len(files):
                el = time.time() - t0
                print(f"[{i}/{len(files)}] {el/i:.2f} s/video", flush=True)

    elapsed = time.time() - t0

    if args.scale_0_100:
        vals = [s for _, s in results]
        lo, hi = min(vals), max(vals)
        if hi - lo > 1e-6:
            results = [(n, (s - lo) / (hi - lo) * 100.0) for n, s in results]

    with open(args.out, "w", encoding="utf-8") as fh:
        for name, score in results:
            fh.write(f"{name}: {score:.1f}\n")

    vals = [s for _, s in results]
    print(f"\nwrote {len(results)} lines -> {args.out}")
    print(f"score range {min(vals):.1f}..{max(vals):.1f}")
    if failed:
        print(f"failed on {len(failed)} videos, filled with the mean: {failed[:5]}")

    # T in the assignment's scoring formula is measured in minutes
    mins = elapsed / 60
    penalty = min(1.0, 0.01 * max(0.0, mins - 20))
    print(f"\nelapsed {mins:.2f} min for {len(files)} videos "
          f"({elapsed/len(files):.2f} s/video)")
    print(f"time penalty at this rate: {penalty:.3f}")
    if len(files) != 100:
        est = elapsed / len(files) * 100 / 60
        print(f"extrapolated to 100 videos: {est:.2f} min, "
              f"penalty {min(1.0, 0.01*max(0.0, est-20)):.3f}")


if __name__ == "__main__":
    main()
