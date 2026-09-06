# -*- coding: utf-8 -*-
"""DOVER 对任意目录视频打分，输出课程任务书格式 score.txt（videoN: 分数，0~100）。

分数 = fuse_results 的 overall（0~1）x100，与官方 evaluate_a_set_of_videos.py 一致。

用法：
    python score_test.py --videos D:/videoquality/data/test_videos \
        --out D:/videoquality/data/test_videos/score_dover.txt --fp16
"""
import argparse
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dover.datasets import ViewDecompositionDataset
from dover.models import DOVER


def fuse_results(results):
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


def natural_key(name):
    import re
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yml", default="divide_repro.yml")
    p.add_argument("--ckpt", default="pretrained_weights/DOVER.pth")
    p.add_argument("--videos", default="D:/videoquality/data/test_videos")
    p.add_argument("--out", default="score_dover.txt")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    opt = yaml.safe_load(open(args.yml, encoding="utf-8"))
    dopt = opt["data"]["val-dividemaxwell"]["args"]
    dopt["anno_file"] = None            # 无标注：全目录打分
    dopt["data_prefix"] = args.videos + "/"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DOVER(**opt["model"]["args"]).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    ds = ViewDecompositionDataset(dopt)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=4, pin_memory=True
    )
    sample_types = ["aesthetic", "technical"]
    scores = {}

    for data in tqdm(loader, total=len(ds)):
        name = data["name"][0].replace("\\", "/").split("/")[-1]
        if len(data.keys()) == 1:
            print("跳过失败:", name)
            continue
        video = {}
        for key in sample_types:
            if key in data:
                v = data[key].to(device)
                b, c, t, h, w = v.shape
                nc = data["num_clips"][key]
                video[key] = (
                    v.reshape(b, c, nc, t // nc, h, w)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(b * nc, c, t // nc, h, w)
                )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            results = model(video, reduce_scores=False)
            results = [np.mean(x.float().cpu().numpy()) for x in results]
        scores[name] = fuse_results(results)["overall"] * 100.0

    with open(args.out, "w", encoding="utf-8") as f:
        for n in sorted(scores, key=natural_key):
            f.write(f"{n}: {scores[n]:.1f}\n")
    print(f"已写入 {args.out}（{len(scores)} 个视频）")
    for n in sorted(scores, key=natural_key):
        print(f"  {n}: {scores[n]:.1f}")


if __name__ == "__main__":
    main()
