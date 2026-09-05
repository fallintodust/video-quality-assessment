#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST-VQA / FasterVQA batch zero-shot evaluation.

Scores every video in a directory with an LSVQ-pretrained FAST-VQA model and,
if labels are given, reports SROCC / PLCC against them.

This is the lightweight stand-in for the CAMP-VQA comparison: FAST-VQA uses a
Video Swin-T backbone (~110 MB of weights) instead of BLIP-2 + Swin-L + CLIP +
SlowFast (~16 GB), so it runs on a 4 GB card in seconds per video rather than
minutes.

Put this file in the root of the FAST-VQA-and-FasterVQA checkout.

Usage:
    python3 fastvqa_eval_batch.py \
        --videos-dir ~/trkv/data \
        --labels ~/video-quality-assessment/data/divide/train_lable_test.txt \
        --model FAST-VQA \
        --out fastvqa_scores.txt
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

import decord
from fastvqa.datasets import (FragmentSampleFrames, SampleFrames,
                              get_spatial_fragments)
from fastvqa.models import DiViDeAddEvaluator

# score normalisation constants, copied from the upstream vqa.py
MEAN_STDS = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA": (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006),
}
OPTS = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml",
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml",
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml",
    "FAST-VQA": "./options/fast/fast-b.yml",
    "FAST-VQA-M": "./options/fast/fast-m.yml",
}


def sigmoid_rescale(score, model):
    mean, std = MEAN_STDS[model]
    return 1 / (1 + np.exp(-(score - mean) / std))


def load_labels(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, v = line.rsplit(":", 1)
        d[os.path.splitext(k.strip())[0]] = float(v)
    return d


def build_samplers(opt, split="val-kv1k"):
    """Return (sampler, sample_args) per sample type, as upstream vqa.py does."""
    t_opt = opt["data"][split]["args"]
    s_opt = t_opt["sample_types"]
    out = {}
    for stype, sargs in s_opt.items():
        if t_opt.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sargs["clip_len"] // sargs.get("t_frag", 1),
                fragments_t=sargs.get("t_frag", 1),
                num_clips=sargs.get("num_clips", 1))
        else:
            sampler = SampleFrames(clip_len=sargs["clip_len"],
                                   num_clips=sargs.get("num_clips", 1))
        out[stype] = (sampler, sargs)
    return out


def score_video(path, evaluator, samplers, device):
    vr = decord.VideoReader(path)
    vsamples = {}
    for stype, (sampler, sargs) in samplers.items():
        frames = sampler(len(vr))
        frame_dict = {i: vr[i] for i in np.unique(frames)}
        video = torch.stack([frame_dict[i] for i in frames], 0).permute(3, 0, 1, 2)
        sampled = get_spatial_fragments(video, **sargs)
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])
        sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        n_clips = sargs.get("num_clips", 1)
        sampled = sampled.reshape(sampled.shape[0], n_clips, -1,
                                  *sampled.shape[2:]).transpose(0, 1)
        vsamples[stype] = sampled.to(device)
    with torch.no_grad():
        return evaluator(vsamples).mean().item()


def main():
    ap = argparse.ArgumentParser(description="FAST-VQA batch zero-shot evaluation")
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--labels", default="", help="optional, format 'name: score'")
    ap.add_argument("--model", default="FAST-VQA", choices=list(OPTS))
    ap.add_argument("--max-n", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="fastvqa_scores.txt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(OPTS[args.model], "r") as f:
        opt = yaml.safe_load(f)

    print(f"model: {args.model}   device: {args.device}")
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(args.device)
    ckpt = opt["test_load_path"]
    if not os.path.exists(ckpt):
        raise SystemExit(
            f"weights not found: {ckpt}\n"
            "download them from the repo releases into ./pretrained_weights/")
    evaluator.load_state_dict(
        torch.load(ckpt, map_location=args.device)["state_dict"])
    evaluator.eval()
    samplers = build_samplers(opt)
    print(f"weights: {ckpt}")

    lab = load_labels(args.labels) if args.labels else {}
    files = sorted(f for f in os.listdir(args.videos_dir)
                   if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    if lab:
        files = [f for f in files if os.path.splitext(f)[0] in lab]
    if args.max_n:
        files = files[:args.max_n]
    print(f"videos to score: {len(files)}")

    t0, rows, bad = time.time(), [], 0
    for i, name in enumerate(files, 1):
        stem = os.path.splitext(name)[0]
        try:
            raw = score_video(os.path.join(args.videos_dir, name),
                              evaluator, samplers, args.device)
            rows.append((stem, sigmoid_rescale(raw, args.model) * 100))
        except Exception as e:
            print(f"  !! {name}: {e}")
            bad += 1
        if i % 25 == 0 or i == len(files):
            el = time.time() - t0
            print(f"[{i}/{len(files)}] {el/i:.2f} s/video, "
                  f"~{(len(files)-i)*el/i/60:.1f} min left", flush=True)
    elapsed = time.time() - t0

    with open(args.out, "w", encoding="utf-8") as fh:
        for n, s in rows:
            fh.write(f"{n}: {s:.4f}\n")
    print(f"\nwrote {len(rows)} lines -> {args.out}   (failed: {bad})")
    print(f"elapsed {elapsed/60:.2f} min, {elapsed/max(len(rows),1):.2f} s/video")

    if lab and rows:
        pred = np.array([s for _, s in rows])
        gt = np.array([lab[n] for n, _ in rows])
        srocc = spearmanr(pred, gt).correlation
        plcc = pearsonr(pred, gt)[0]
        print(f"\nvs labels (n={len(rows)}): "
              f"SROCC={srocc:.4f}  PLCC={plcc:.4f}  OBJ={srocc+plcc:.4f}")
        import json
        json.dump({"model": args.model, "n": len(rows),
                   "srocc": float(srocc), "plcc": float(plcc),
                   "obj": float(srocc + plcc),
                   "s_per_video": elapsed / len(rows)},
                  open(os.path.splitext(args.out)[0] + "_metrics.json", "w"),
                  indent=2)


if __name__ == "__main__":
    main()
