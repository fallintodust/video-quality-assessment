"""帧抽取与预处理（参考 TSN [4]）。

任务书规定：
- 帧抽取：按均匀间隔抽取固定数量 T 帧；帧索引等间距取样（线性插值索引）；
  若视频帧数不足，用最后一帧重复补齐。
- 尺寸统一：每帧缩放到 224x224。
- 数值范围：0~1 归一化 + ImageNet 均值/方差标准化。
"""

import os

import numpy as np
import torch

from .config import Config


def sample_frame_indices(num_frames: int, T: int) -> np.ndarray:
    """等间距抽取 T 帧的索引（线性插值 + 四舍五入）。

    当 num_frames < T 时，clip 到最后一帧 —— 等价于"用最后一帧重复补齐"。
    """
    idx = np.linspace(0, num_frames - 1, T)
    idx = np.round(idx).astype(int)
    return np.clip(idx, 0, num_frames - 1)


def load_cached_frames(path, cache_dir):
    """从预抽帧缓存读取 [T, size, size, 3] uint8 BGR；未命中返回 None。"""
    if not cache_dir:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    npy = os.path.join(cache_dir, stem + ".npy")
    if not os.path.isfile(npy):
        return None
    x = np.load(npy)
    if x.ndim != 4 or x.dtype != np.uint8:
        return None
    return x


def read_video_frames(path, T=8, size=224, mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
    """读取视频并按 TSN 策略抽取 T 帧，返回 [T, 3, size, size] 标准化张量。

    - 命中 Config.FRAME_CACHE（scripts/precache_frames.py 预抽帧）时直接
      读 npy，跳过视频解码；缓存帧数须 >= T，取前 T 帧。
    - 否则 OpenCV 顺序读取，只保留被采样到的帧（避免 seek 不可靠）。
    - 若视频实际帧数少于 T，用最后一帧重复补齐。
    """
    cached = load_cached_frames(path, Config.FRAME_CACHE)
    if cached is not None:
        x = torch.from_numpy(cached[:T]).float().permute(0, 3, 1, 2) / 255.0
        mean_t = torch.tensor(mean, dtype=x.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean_t) / std_t

    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"无法打开视频: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # 帧数元数据不可靠：先顺序读一遍计数，再回卷重读
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        total = n
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    idx = sample_frame_indices(total, T)
    wanted = set(int(i) for i in idx)

    frames = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frame = cv2.resize(frame, (size, size))          # 统一尺寸
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR -> RGB
            frames.append(frame)
    cap.release()

    if not frames:  # 完全读不出帧（损坏文件）
        raise IOError(f"视频无有效帧: {path}")

    while len(frames) < T:  # 帧数不足：最后一帧重复补齐
        frames.append(frames[-1])
    frames = frames[:T]

    x = torch.from_numpy(np.stack(frames)).float().permute(0, 3, 1, 2) / 255.0
    mean_t = torch.tensor(mean, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t
