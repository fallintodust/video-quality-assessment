"""训练工具：加权 MSE、单轮训练、打分、checkpoint 保存/加载。

checkpoint 布局（任务书要求三份权重）：
    <dir>/extractor.pt   特征提取器单独权重
    <dir>/head.pt        回归头单独权重
    <dir>/model.pt       综合 checkpoint（含 meta）
"""

import json
import os
import random

import numpy as np
import torch

from .models import VQAModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_mse(pred, target, weight):
    """加权均方误差：loss = mean(w * (pred - target)^2)"""
    return (weight * (pred - target) ** 2).mean()


def train_one_epoch(model, loader, optimizer, device, log_interval=10, epoch=0):
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss, n = 0.0, 0
    for i, (frames, labels, weights) in enumerate(loader):
        frames = frames.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        optimizer.zero_grad()
        pred = model(frames)
        loss = weighted_mse(pred, labels, weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
        if log_interval > 0 and (i + 1) % log_interval == 0:
            print(f"  [epoch {epoch}] step {i + 1}/{len(loader)} loss={loss.item():.3f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device):
    """在数据集上计算平均（加权）MSE。"""
    model.eval()
    total_loss, n = 0.0, 0
    for frames, labels, weights in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        pred = model(frames)
        total_loss += weighted_mse(pred, labels, weights).item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


def _name_at(dataset, i):
    """按序取第 i 个样本的视频名（兼容 Subset 包装）。"""
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        return _name_at(dataset.dataset, dataset.indices[i])
    return dataset.items[i][0]


@torch.no_grad()
def score_videos(model, dataset, device, batch_size=4):
    """对数据集中的视频打分，返回 {视频名: 预测分}。

    适用于带标签或不带标签的数据集（label 被忽略，只用视频帧）。
    """
    from torch.utils.data import DataLoader
    from .dataset import collate_videos

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_videos, num_workers=0)
    scores = {}
    seen = 0  # 已处理样本数（独立计数，避免与 dict 长度混淆）
    for frames, _, _ in loader:
        pred = model(frames.to(device)).cpu().numpy()
        for j in range(frames.shape[0]):
            name = _name_at(dataset, seen + j)
            scores[name] = float(pred[j])
        seen += frames.shape[0]
    return scores


def save_checkpoint(model, out_dir, meta=None, tag=""):
    """保存 extractor / head / 综合 三份权重（任务书 (A) 第 4 条）。"""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.extractor.state_dict(), os.path.join(out_dir, f"extractor{tag}.pt"))
    torch.save(model.head.state_dict(), os.path.join(out_dir, f"head{tag}.pt"))
    torch.save({"extractor": model.extractor.state_dict(),
                "head": model.head.state_dict(),
                "meta": meta or {}},
               os.path.join(out_dir, f"model{tag}.pt"))
    if meta:
        with open(os.path.join(out_dir, f"meta{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def load_checkpoint(model, ckpt_dir, tag=""):
    """从 checkpoint 目录加载权重（优先综合 model.pt，缺省回退单独文件）。"""
    model_pt = os.path.join(ckpt_dir, f"model{tag}.pt")
    if os.path.isfile(model_pt):
        state = torch.load(model_pt, map_location="cpu", weights_only=True)
        model.extractor.load_state_dict(state["extractor"])
        model.head.load_state_dict(state["head"])
        return state.get("meta", {})
    extractor_pt = os.path.join(ckpt_dir, f"extractor{tag}.pt")
    head_pt = os.path.join(ckpt_dir, f"head{tag}.pt")
    if os.path.isfile(extractor_pt):
        model.extractor.load_state_dict(
            torch.load(extractor_pt, map_location="cpu", weights_only=True))
    if os.path.isfile(head_pt):
        model.head.load_state_dict(
            torch.load(head_pt, map_location="cpu", weights_only=True))
    return {}


def build_model(device, T=None):
    """构造模型并移到设备。"""
    model = VQAModel(T=T)
    model.to(device)
    return model


def get_device():
    """优先 GPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
