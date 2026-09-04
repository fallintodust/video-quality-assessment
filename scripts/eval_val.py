"""在独立验证集上重评模型，打印高精度指标并保存分数。

与训练脚本同源的加载/打分路径（帧缓存 + fp16），用于：
  - 复核训练日志里 4 位小数 SROCC/PLCC 的真实值
  - 导出逐视频分数文件（报告/答辩素材）

用法：
    python scripts/eval_val.py --model-dir runs/divide_baseline \
        --data-dir data/divide/videos --labels data/divide/train_lable_test.txt \
        --frame-cache data/frames_cache --fp16 --out runs/divide_baseline/val_scores_best.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vqa.config import Config
from vqa.dataset import VideoDataset, parse_labels, resolve_labels
from vqa.metrics import evaluate_metrics
from vqa.train_utils import (build_model, get_device, load_checkpoint,
                             score_videos, set_seed)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="独立验证集重评（高精度 + 导出分数）")
    p.add_argument("--model-dir", required=True, help="checkpoint 目录")
    p.add_argument("--tag", default="_best", help="checkpoint tag：_best / 无")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--frame-cache", default="")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--t", type=int, default=8)
    p.add_argument("--out", default="", help="逐视频分数输出文件（txt）")
    args = p.parse_args()

    Config.FRAME_CACHE = args.frame_cache or None
    set_seed(Config.SEED)
    device = get_device()
    print(f"设备: {device}  fp16: {args.fp16}  帧缓存: {Config.FRAME_CACHE or '无'}")

    labels = parse_labels(args.labels)
    labels, missing = resolve_labels(labels, args.data_dir)
    if missing:
        print(f"警告: 标注中 {len(missing)} 个视频文件不存在，已跳过")
    print(f"验证标注视频数: {len(labels)}")

    model = build_model(device, T=args.t)
    load_checkpoint(model, args.model_dir, tag=args.tag or "")
    print(f"权重: {args.model_dir} (tag={args.tag or '默认'})")

    ds = VideoDataset.from_labels(args.data_dir, labels, T=args.t)
    scores = score_videos(model, ds, device, batch_size=args.bs, use_fp16=args.fp16)

    names = sorted(scores)
    y_true = [labels[n] for n in names]
    y_pred = [scores[n] for n in names]
    m = evaluate_metrics(y_true, y_pred)
    print(f"SROCC = {m['SROCC']:.6f}")
    print(f"PLCC  = {m['PLCC']:.6f}")
    print(f"OBJ   = {m['OBJ']:.6f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for n in names:
                f.write(f"{n} {scores[n]:.6f}\n")
        print(f"分数已保存: {args.out}（{len(names)} 条）")


if __name__ == "__main__":
    main()
