#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a flicker benchmark on REAL videos.

MaxWell has no flicker axis, and a fully synthetic set (moving gradients,
checkerboards) is too easy: there is no real motion, no compression, no
natural lighting change for a detector to confuse with flicker.

This script takes real MaxWell clips and injects brightness flicker at a
controlled severity, so the label is known exactly while the content stays
realistic:

    frame *= (1 + s * sin(2*pi*f*t + phi)),   f in [3, 8] Hz
    MOS = 100 * (1 - s)                       (higher = more stable)

Severity 0 clips are kept as controls: a detector that flags everything will
fail on them.

With --paired, every source video is written twice - once untouched and once
flickered - so the two differ ONLY by the injected distortion. That removes
content as a confounder and makes the comparison much sharper.

Usage:
    python3 scripts/make_real_flicker.py --src ~/trkv/data --out data/real_flicker --n 60
    python3 scripts/make_real_flicker.py --src ~/trkv/data --out data/real_flicker --n 40 --paired
"""
import argparse
import os
import random

import cv2
import numpy as np


def read_video(path, max_frames=0, size=0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if size:
            h, w = fr.shape[:2]
            scale = size / max(h, w)
            if scale < 1:
                fr = cv2.resize(fr, (int(w * scale), int(h * scale)))
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def apply_flicker(frames, fps, severity, freq, phase):
    """Periodic brightness modulation of the whole frame."""
    out = []
    for t, fr in enumerate(frames):
        gain = 1.0 + severity * np.sin(2 * np.pi * freq * t / fps + phase)
        out.append(np.clip(fr.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    return out


def write_video(frames, path, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of real videos")
    ap.add_argument("--out", default="data/real_flicker")
    ap.add_argument("--n", type=int, default=60, help="number of source videos")
    ap.add_argument("--paired", action="store_true",
                    help="write each source twice: clean + flickered")
    ap.add_argument("--max-frames", type=int, default=180,
                    help="cap length to keep encoding fast (0 = full)")
    ap.add_argument("--size", type=int, default=480,
                    help="longest side, 0 = keep original")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    srcs = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".mp4"))
    rng.shuffle(srcs)
    srcs = srcs[:args.n]
    print(f"{len(srcs)} source videos -> {vid_dir}")

    labels, idx = [], 0
    for k, name in enumerate(srcs, 1):
        frames, fps = read_video(os.path.join(args.src, name),
                                 args.max_frames, args.size)
        if len(frames) < 16:
            print(f"  skip {name}: only {len(frames)} frames")
            continue

        if args.paired:
            plan = [0.0, rng.uniform(0.15, 0.90)]     # clean + flickered
        else:
            # a quarter of the set is left clean as controls
            plan = [0.0 if rng.random() < 0.25 else rng.uniform(0.10, 0.95)]

        for sev in plan:
            idx += 1
            out_name = f"video{idx}.mp4"
            if sev > 0:
                freq = rng.uniform(3.0, 8.0)
                seq = apply_flicker(frames, fps, sev, freq, rng.uniform(0, 6.28))
            else:
                freq, seq = 0.0, frames
            write_video(seq, os.path.join(vid_dir, out_name), fps)
            labels.append((out_name, 100.0 * (1 - sev), name, sev, freq))
        if k % 10 == 0:
            print(f"  [{k}/{len(srcs)}] {idx} clips written", flush=True)

    lab_path = os.path.join(args.out, "labels.txt")
    with open(lab_path, "w", encoding="utf-8") as fh:
        for n, mos, *_ in labels:
            fh.write(f"{os.path.splitext(n)[0]}: {mos:.1f}\n")

    meta_path = os.path.join(args.out, "meta.csv")
    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write("clip,mos,source,severity,freq_hz\n")
        for n, mos, src, sev, freq in labels:
            fh.write(f"{n},{mos:.1f},{src},{sev:.4f},{freq:.2f}\n")

    clean = sum(1 for _, _, _, s, _ in labels if s == 0)
    print(f"\n{len(labels)} clips  ({clean} clean, {len(labels)-clean} flickered)")
    print(f"labels -> {lab_path}")
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
