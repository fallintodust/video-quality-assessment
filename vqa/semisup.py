"""（B）阶段半监督伪标签循环（任务书第五章 B 部分）。

每轮两阶段：
  B.1 伪标签生成：N 次独立训练-打分（每次从基础权重重新加载、随机抽样训练子集），
      对未标注视频计算 N 次预测的方差，方差小于阈值的取均值作为伪标签加入累积池。
  B.2 验证/微调：用全部训练集（真实 w=1.0 + 伪标签 w=W_PSEUDO）训练若干 epoch，
      在锁定验证集上算 SROCC/PLCC，OBJ=SROCC+PLCC 优于 best 则保存，连续无提升早停。
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import Config
from .dataset import (VideoDataset, collate_videos, list_videos,
                      parse_labels, resolve_labels)
from .metrics import evaluate_metrics
from .train_utils import (build_model, eval_loss, load_checkpoint,
                          save_checkpoint, score_videos, set_seed,
                          train_one_epoch)


def split_labeled(labels, val_ratio, hide_ratio, seed, video_dir, out_dir):
    """划分标注数据：验证集(锁定) / 训练标注集 / 隐藏(视为未标注) / 无标注视频。

    返回 (train_labels, val_labels, hidden_names, unlabeled_names)
    """
    names = sorted(labels.keys())
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = names[:n_val]
    rest = names[n_val:]
    n_hide = int(len(rest) * hide_ratio)
    hidden_names = rest[:n_hide]
    train_names = rest[n_hide:]

    train_labels = {n: labels[n] for n in train_names}
    val_labels = {n: labels[n] for n in val_names}
    # 目录中没有任何标注的视频（真实"未标注"数据）
    extra = [n for n in list_videos(video_dir) if n not in labels]
    unlabeled_names = hidden_names + extra

    info = {"val": val_names, "hidden": hidden_names, "extra_unlabeled": extra}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return train_labels, val_labels, unlabeled_names


def build_train_dataset(video_dir, train_labels, pool, T):
    """真实标签(w=1.0) + 累积伪标签(w=PSEUDO_WEIGHT) 合成训练集。"""
    items = [(n, v, 1.0) for n, v in sorted(train_labels.items())]
    items += [(n, mos, Config.PSEUDO_WEIGHT) for n, (mos, _var) in sorted(pool.items())]
    return VideoDataset(video_dir, items, T=T)


def pseudo_label_round(model_template_fn, video_dir, train_labels, pool,
                       unlabeled_names, n_runs, sub_ratio, epochs, lr, bs,
                       device, seed_offset, T, log_interval):
    """B.1 一轮：N 次独立训练-打分 + 稳定性筛选，返回新增伪标签 dict {name: (mos, var)}。

    model_template_fn: 每次调用返回"已加载基础权重"的新模型实例。
    """
    rng = random.Random(seed_offset)
    base_items = [(n, v, 1.0) for n, v in sorted(train_labels.items())]
    base_items += [(n, mos, Config.PSEUDO_WEIGHT) for n, (mos, _v) in sorted(pool.items())]
    train_ds = VideoDataset(video_dir, base_items, T=T)
    all_names = sorted(set(list(train_labels.keys()) + unlabeled_names))
    all_ds = VideoDataset.unlabeled(video_dir, all_names, T=T)

    preds = {n: [] for n in unlabeled_names}
    for run in range(1, n_runs + 1):
        # 随机抽样子集训练（每次重新加载基础权重，控制随机性来源）
        n_sub = max(1, int(len(train_ds) * sub_ratio))
        idx = rng.sample(range(len(train_ds)), n_sub)
        sub_ds = Subset(train_ds, idx)
        loader = DataLoader(sub_ds, batch_size=bs, shuffle=True,
                            collate_fn=collate_videos, num_workers=0)
        model = model_template_fn()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for ep in range(1, epochs + 1):
            train_one_epoch(model, loader, optimizer, device,
                            log_interval=log_interval, epoch=ep)
        scores = score_videos(model, all_ds, device, batch_size=bs)
        for n in unlabeled_names:
            preds[n].append(scores[n])
        print(f"  run {run}/{n_runs} 完成")
        del model
        torch.cuda.empty_cache()

    new_pseudo = {}
    for n in unlabeled_names:
        if n in pool:
            continue
        arr = np.array(preds[n])
        var = float(arr.var())
        if var < Config.VAR_THRESHOLD:
            new_pseudo[n] = (float(arr.mean()), var)
    return new_pseudo


def validation_round(model_template_fn, video_dir, train_labels, pool, val_labels,
                     epochs, lr, bs, device, T, log_interval):
    """B.2 一轮：全量训练 + 验证集指标，返回 (model, metrics)。"""
    train_ds = build_train_dataset(video_dir, train_labels, pool, T)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                        collate_fn=collate_videos, num_workers=0)
    val_ds = VideoDataset.from_labels(video_dir, val_labels, weight=1.0, T=T)
    model = model_template_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        train_one_epoch(model, loader, optimizer, device,
                        log_interval=log_interval, epoch=ep)
    train_loss = eval_loss(model, loader, device)
    scores = score_videos(model, val_ds, device, batch_size=bs)
    names = [it[0] for it in sorted(val_ds.items, key=lambda it: it[0])]
    y_true = [val_labels[n] for n in names]
    y_pred = [scores[n] for n in names]
    m = evaluate_metrics(y_true, y_pred)
    m["train_loss"] = train_loss
    return model, m


def run_semisup(args):
    """完整半监督主循环，返回汇总 dict。"""
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    labels = parse_labels(args.labels)
    labels, missing = resolve_labels(labels, args.data_dir)
    for n in missing:
        print(f"警告: 跳过缺失视频 {n}")
    print(f"标注视频总数: {len(labels)}")

    train_labels, val_labels, unlabeled_names = split_labeled(
        labels, args.val_ratio, args.hide_ratio, args.seed,
        args.data_dir, args.out)
    print(f"训练标注集: {len(train_labels)} | 验证集(锁定): {len(val_labels)} | "
          f"未标注视频: {len(unlabeled_names)}")

    def model_template():
        m = build_model(device, T=args.t)
        load_checkpoint(m, args.baseline)
        return m

    pool = {}          # name -> (pseudo_mos, var)
    best = {"OBJ": -float("inf"), "SROCC": float("nan"), "PLCC": float("nan"),
            "round": 0, "pool_size": 0}
    no_improve = 0
    history = []

    for rnd in range(1, args.pseudo_rounds + 1):
        print(f"\n===== 轮次 {rnd}/{args.pseudo_rounds} | B.1 伪标签生成 =====")
        new_pseudo = pseudo_label_round(
            model_template, args.data_dir, train_labels, pool, unlabeled_names,
            args.n_runs, args.sub_ratio, args.sub_epochs, args.lr, args.bs,
            device, args.seed + rnd * 1000, args.t, args.log_interval)
        for n, (mos, var) in new_pseudo.items():
            pool[n] = (mos, var)
        print(f"本轮新增伪标签 {len(new_pseudo)} 个，累积 {len(pool)} 个")
        with open(os.path.join(args.out, "pool.json"), "w", encoding="utf-8") as f:
            json.dump({n: {"mos": v[0], "var": v[1]} for n, v in pool.items()},
                      f, ensure_ascii=False, indent=2)

        print(f"===== 轮次 {rnd} | B.2 验证/微调 =====")
        model, m = validation_round(
            model_template, args.data_dir, train_labels, pool, val_labels,
            args.val_epochs, args.lr, args.bs, device, args.t, args.log_interval)
        entry = {"round": rnd, "pool_size": len(pool), **m}
        history.append(entry)
        print(f"SROCC={m['SROCC']:.4f} PLCC={m['PLCC']:.4f} OBJ={m['OBJ']:.4f}")

        if m["OBJ"] > best["OBJ"]:
            best = {"OBJ": m["OBJ"], "SROCC": m["SROCC"], "PLCC": m["PLCC"],
                    "round": rnd, "pool_size": len(pool)}
            save_checkpoint(model, args.out, tag="_best",
                            meta={"round": rnd, "pool_size": len(pool), **m})
            print(f"  -> 新 best (OBJ={m['OBJ']:.4f})，权重已保存")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  未提升 (连续 {no_improve}/{args.early_stop})")
        del model
        torch.cuda.empty_cache()
        if no_improve >= args.early_stop:
            print("触发早停")
            break

    summary = {"best": best, "history": history, "pool_size": len(pool)}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成。best: round={best['round']} SROCC={best['SROCC']:.4f} "
          f"PLCC={best['PLCC']:.4f} OBJ={best['OBJ']:.4f}")
    print(f"输出目录: {args.out} (model_best.pt / pool.json / metrics.json)")
    return summary
