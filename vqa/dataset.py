"""数据集与标注解析。

标注文件格式与任务书一致，每行一个样本：`videoN: MOS`。
兼容 `videoN  MOS`（空白分隔）与 CSV（逗号分隔）；支持 `#` 注释行。
MOS 为 0~100 的时域一致性主观分。
"""

import os
import re

import torch
from torch.utils.data import Dataset

from .sampling import read_video_frames
from .config import Config

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def parse_labels(path):
    """解析标注文件 -> {视频名: MOS}。"""
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
            else:
                parts = re.split(r"[,\s]+", line)
                k, v = parts[0], parts[1]
            labels[k.strip()] = float(v)
    return labels


def natural_key(name):
    """自然排序键：video2 < video10。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def resolve_name(name, video_dir):
    """把标注里的名字对应到视频目录中的实际文件。

    兼容标注名不带扩展名的情况（如 "0319" -> "0319.mp4"）。
    返回实际文件名；目录中不存在时返回 None。
    """
    if os.path.isfile(os.path.join(video_dir, name)):
        return name
    for ext in sorted(VIDEO_EXTS):
        cand = name + ext
        if os.path.isfile(os.path.join(video_dir, cand)):
            return cand
    return None


def resolve_labels(labels, video_dir):
    """把标注名统一为目录中的实际文件名，缺失的视频剔除并返回。"""
    resolved = {}
    missing = []
    for n, v in labels.items():
        real = resolve_name(n, video_dir)
        if real is None:
            missing.append(n)
        else:
            resolved[real] = v
    return resolved, missing


def list_videos(video_dir):
    """列出目录下所有视频文件（按自然序）。"""
    if not os.path.isdir(video_dir):
        return []
    names = [n for n in os.listdir(video_dir)
             if os.path.splitext(n)[1].lower() in VIDEO_EXTS]
    return sorted(names, key=natural_key)


class VideoDataset(Dataset):
    """样本 = (frames [T,3,S,S], label, weight)。

    items: [(视频名, MOS 或 None, 样本权重)]。label=None 表示未标注（仅打分用）。
    """

    def __init__(self, video_dir, items, T=None, size=None,
                 mean=None, std=None):
        self.video_dir = video_dir
        self.items = items
        self.T = T if T is not None else Config.T
        self.size = size if size is not None else Config.SIZE
        self.mean = mean if mean is not None else tuple(Config.MEAN)
        self.std = std if std is not None else tuple(Config.STD)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, label, weight = self.items[i]
        path = os.path.join(self.video_dir, name)
        frames = read_video_frames(path, T=self.T, size=self.size,
                                   mean=self.mean, std=self.std)
        y = float(label) if label is not None else -1.0
        return frames, torch.tensor(y), torch.tensor(weight)

    @staticmethod
    def from_labels(video_dir, labels, weight=1.0, T=None):
        """由 {视频名: MOS} 构建数据集（真实标签，权重 1.0）。"""
        items = [(n, v, float(weight)) for n, v in sorted(labels.items(), key=lambda kv: natural_key(kv[0]))]
        return VideoDataset(video_dir, items, T=T)

    @staticmethod
    def unlabeled(video_dir, names, T=None):
        """未标注视频集合（label=None，仅用于打分）。"""
        items = [(n, None, 0.0) for n in names]
        return VideoDataset(video_dir, items, T=T)


def collate_videos(batch):
    """默认 collate：frames 堆叠为 [B,T,3,H,W]，label/weight 堆叠为 [B]。"""
    frames = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    weights = torch.stack([b[2] for b in batch])
    return frames, labels, weights
