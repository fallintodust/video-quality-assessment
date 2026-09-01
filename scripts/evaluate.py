"""模型评估：在标注集上计算 SROCC / PLCC / OBJ。

默认与训练脚本使用相同的种子/比例划出验证集（仅在该子集上评估，避免训练数据污染），
也可用 --eval-all 在全部标注视频上评估。

用法：
    python scripts/evaluate.py --model runs/baseline/model_best.pt \
        --data-dir data/synthetic/videos --labels data/synthetic/labels.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vqa.dataset import VideoDataset, parse_labels, resolve_labels
from vqa.metrics import evaluate_metrics
from vqa.train_utils import build_model, get_device, load_checkpoint, score_videos, set_seed


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model.pt / model_best.pt 文件路径")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-all", action="store_true", help="在全部标注视频上评估（训练数据未剔除，指标偏乐观）")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--t", type=int, default=8)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    labels = parse_labels(args.labels)
    labels, _ = resolve_labels(labels, args.data_dir)
    print(f"标注视频: {len(labels)} | 设备: {device}")

    model = build_model(device, T=args.t)
    if os.path.isdir(args.model):
        load_checkpoint(model, args.model)  # 目录：加载 model.pt
    else:
        state = torch.load(args.model, map_location="cpu", weights_only=True)
        model.extractor.load_state_dict(state["extractor"])
        model.head.load_state_dict(state["head"])
    print(f"权重: {args.model}")

    if args.eval_all:
        names = sorted(labels.keys())
    else:
        names = sorted(labels.keys())
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(names), generator=g).tolist()
        n_val = max(1, int(len(names) * args.val_ratio))
        names = [names[i] for i in idx[:n_val]]
        print(f"评估子集: 验证划分 {len(names)} 个（与训练脚本同 seed/ratio）")

    ds = VideoDataset.from_labels(args.data_dir, {n: labels[n] for n in names}, T=args.t)
    scores = score_videos(model, ds, device, batch_size=args.bs)
    y_true = [labels[n] for n in names]
    y_pred = [scores[n] for n in names]
    m = evaluate_metrics(y_true, y_pred)
    print(f"SROCC = {m['SROCC']:.4f}")
    print(f"PLCC  = {m['PLCC']:.4f}")
    print(f"OBJ   = {m['OBJ']:.4f}")


if __name__ == "__main__":
    main()
