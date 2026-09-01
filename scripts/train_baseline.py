"""（A）阶段 baseline 训练（任务书第五章）。

用标注视频（videoN: MOS）监督训练"视觉特征提取器 + 回归头"，损失为 MSE；
训练中监控验证集 loss，遇到更优指标保存 best（extractor / head / 综合三份权重）。

用法：
    python scripts/train_baseline.py --data-dir data/synthetic/videos \
        --labels data/synthetic/labels.txt --out runs/baseline --epochs 20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from vqa.config import Config
from vqa.dataset import (VideoDataset, collate_videos, list_videos,
                         parse_labels, resolve_labels)
from vqa.metrics import evaluate_metrics
from vqa.train_utils import (build_model, eval_loss, get_device, load_checkpoint,
                             save_checkpoint, score_videos, set_seed, train_one_epoch)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="baseline 监督训练")
    p.add_argument("--data-dir", default="data/synthetic/videos", help="视频目录")
    p.add_argument("--labels", default="data/synthetic/labels.txt", help="标注文件")
    p.add_argument("--out", default="runs/baseline", help="权重输出目录")
    p.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p.add_argument("--bs", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=Config.LR)
    p.add_argument("--wd", type=float, default=Config.WEIGHT_DECAY)
    p.add_argument("--t", type=int, default=Config.T, help="每视频抽帧数")
    p.add_argument("--val-ratio", type=float, default=Config.VAL_RATIO)
    p.add_argument("--seed", type=int, default=Config.SEED)
    p.add_argument("--resume", default="", help="从该目录加载权重继续训练")
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}")

    # ---- 数据准备 ----
    labels = parse_labels(args.labels)
    labels, missing = resolve_labels(labels, args.data_dir)
    if missing:
        print(f"警告: 标注中 {len(missing)} 个视频文件不存在，已跳过")
    print(f"标注视频数: {len(labels)}")

    all_ds = VideoDataset.from_labels(args.data_dir, labels, weight=1.0, T=args.t)
    n_val = max(1, int(len(all_ds) * args.val_ratio))
    n_train = len(all_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(all_ds), generator=g).tolist()
    train_ds = Subset(all_ds, idx[:n_train])
    val_ds = Subset(all_ds, idx[n_train:])
    print(f"划分: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=collate_videos, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            collate_fn=collate_videos, num_workers=0)

    # ---- 模型与优化器 ----
    model = build_model(device, T=args.t)
    if args.resume:
        load_checkpoint(model, args.resume)
        print(f"从 {args.resume} 加载权重")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_params/1e6:.1f}M")

    # ---- 训练循环 ----
    best_val_loss = float("inf")
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     log_interval=args.log_interval, epoch=epoch)
        vloss = eval_loss(model, val_loader, device)
        msg = f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.3f} val_loss={vloss:.3f}"
        if vloss < best_val_loss:
            best_val_loss = vloss
            save_checkpoint(model, args.out, tag="_best",
                            meta={"epoch": epoch, "val_loss": vloss})
            msg += "  <- best, saved"
        print(msg)

    save_checkpoint(model, args.out, meta={"epoch": args.epochs, "val_loss": vloss})
    print(f"最终权重已保存: {args.out}")

    # ---- best 模型在验证集上的指标（样本足够时）----
    model = build_model(device, T=args.t)
    load_checkpoint(model, args.out, tag="_best")
    if len(val_ds) >= 8:
        scores = score_videos(model, val_ds, device, batch_size=args.bs)
        names = sorted(scores, key=lambda n: [int(x) if x.isdigit() else x
                                              for x in __import__("re").split(r"(\d+)", n)])
        y_true = [labels[n] for n in names]
        y_pred = [scores[n] for n in names]
        m = evaluate_metrics(y_true, y_pred)
        print(f"验证集指标 -> SROCC={m['SROCC']:.4f} PLCC={m['PLCC']:.4f} OBJ={m['OBJ']:.4f}")
    else:
        print(f"验证集仅 {len(val_ds)} 个样本，跳过指标计算（建议 >=8）")


if __name__ == "__main__":
    main()
