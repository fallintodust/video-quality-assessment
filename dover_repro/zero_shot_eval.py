# -*- coding: utf-8 -*-
"""DOVER 零样本验证：官方 DOVER.pth 在 DIVIDE-MaxWell 909 验证集上打分，
并与官方 overall 标注 + 课程重建标注对比 SROCC/PLCC/KROCC。

复现目标（官方 README 报告）：SROCC=0.7477 / PLCC=0.7546 / KROCC=0.5510
"""
import argparse
import csv

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm

from dover.datasets import ViewDecompositionDataset
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results):
    """官方 evaluate_a_set_of_videos.py 的融合公式（LSVQ 标定常量）。"""
    t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1 / (1 + np.exp(-a)),
        "technical": 1 / (1 + np.exp(-t)),
        "overall": 1 / (1 + np.exp(-x)),
    }


def parse_labels(path, col=3):
    """官方标注 CSV：[filename, aesthetic, technical, overall] -> {name: overall}"""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            d[parts[0]] = float(parts[col])
    return d


def parse_user_labels(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, v = line.split(":")
            d[name.strip()] = float(v.strip())
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yml", default="divide_repro.yml")
    p.add_argument("--ckpt", default="pretrained_weights/DOVER.pth")
    p.add_argument("--out", default="zero_shot_predictions.csv")
    p.add_argument("--fp16", action="store_true", help="推理用 fp16 autocast 加速")
    args = p.parse_args()

    opt = yaml.safe_load(open(args.yml, encoding="utf-8"))
    dopt = opt["data"]["val-dividemaxwell"]["args"]
    anno = dopt["anno_file"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = DOVER(**opt["model"]["args"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    official = parse_labels(anno, col=3)  # overall 列
    user = parse_user_labels("D:/videoquality/data/divide/train_lable_test.txt")

    dataset = ViewDecompositionDataset(dopt)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4, pin_memory=True
    )

    sample_types = ["aesthetic", "technical"]
    rows = []
    names, pred_overall, pred_a, pred_t = [], [], [], []
    gt_off, gt_usr = [], []

    # 断点续跑：已完成视频
    done = set()
    import os
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            next(f)
            for line in f:
                done.add(line.split(",")[0])
    print(f"已完成 {len(done)} 个视频，跳过")

    out_f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if not done:
        writer.writerow(["video", "aesthetic", "technical", "overall",
                         "official_overall", "user_rebuilt_label"])
    n_new = 0

    for data in tqdm(loader, desc="Zero-shot 打分", total=len(dataset)):
        name = data["name"][0].replace("\\", "/").split("/")[-1]
        if name in done:
            continue
        if len(data.keys()) == 1:
            print("跳过失败视频:", name)
            continue
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w)
                )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            results = model(video, reduce_scores=False)
            results = [np.mean(x.float().cpu().numpy()) for x in results]
        fused = fuse_results(results)

        names.append(name)
        pred_a.append(fused["aesthetic"])
        pred_t.append(fused["technical"])
        pred_overall.append(fused["overall"])
        gt_off.append(official.get(name, np.nan))
        gt_usr.append(user.get(name, np.nan))
        writer.writerow(
            [name, fused["aesthetic"], fused["technical"], fused["overall"],
             official.get(name, ""), user.get(name, "")]
        )
        out_f.flush()
        n_new += 1
        if n_new % 50 == 0:
            print(f"已新完成 {n_new} 个")
    out_f.close()
    print(f"本次新完成 {n_new} 个，预测已保存: {args.out}")

    def metrics(pred, gt, label):
        valid = np.isfinite(gt)
        p_, g_ = np.array(pred)[valid], np.array(gt)[valid]
        s = spearmanr(p_, g_)[0]
        pl = pearsonr(p_, g_)[0]
        k = kendalltau(p_, g_)[0]
        print(f"[{label}] n={valid.sum()} SROCC={s:.4f} PLCC={pl:.4f} KROCC={k:.4f}")
        return s, pl

    print("=" * 60)
    # 指标从完整 CSV 重算（含断点续跑的历史行）
    all_rows = list(csv.reader(open(args.out, encoding="utf-8")))
    hdr, body = all_rows[0], all_rows[1:]
    names = [r[0] for r in body]
    pred_a = [float(r[1]) for r in body]
    pred_t = [float(r[2]) for r in body]
    pred_overall = [float(r[3]) for r in body]
    gt_off = [official.get(n, np.nan) for n in names]
    gt_usr = [user.get(n, np.nan) for n in names]
    metrics(pred_overall, gt_off, "overall vs 官方overall (复现目标 0.7477/0.7546)")
    metrics(pred_a, gt_off, "aesthetic vs 官方overall")
    metrics(pred_t, gt_off, "technical vs 官方overall")
    metrics(pred_overall, gt_usr, "overall vs 课程重建标注")


if __name__ == "__main__":
    main()
