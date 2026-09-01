"""合成闪烁数据集生成器（开发/自测用）。

生成与任务书失真类型一致的视频：不同程度的典型闪烁（时域一致性损伤），
每个视频附带 MOS 分数（0~100，描述整体时域一致性，越高越好）。

失真类型（严重程度由 severity s 统一控制，s ∈ [0.05, 0.95]）：
  1. 亮度闪烁：帧亮度按 (1 + s*sin(2π f t + φ)) 周期性抖动（f 3~8 Hz）
  2. 帧冻结卡顿：以 s*0.4 的概率冻结当前帧 2 帧（时域不连续）
  3. 时域噪声：每帧叠加 N(0, s*15) 高斯噪声

MOS = round(100 * (1 - s), 1)，即 s=0.05 -> 95 分（几乎无损），s=0.95 -> 5 分（严重闪烁）。

用法：
    python scripts/make_flicker_dataset.py --n 24 --out data/synthetic
产物：
    data/synthetic/videos/video1.mp4 ... videoN.mp4
    data/synthetic/labels.txt   （videoN: MOS，与任务书格式一致）
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from vqa.dataset import natural_key


def base_scene(kind, t, n_frames, w=480, h=270, rng=None):
    """生成第 t 帧的干净画面（不同场景类型，保证内容多样性）。"""
    rng = rng or np.random.default_rng(0)
    if kind == "gradient":
        # 移动的对角渐变 + 移动亮圆
        x = np.linspace(-1, 1, w, dtype=np.float32)
        y = np.linspace(-1, 1, h, dtype=np.float32)
        phase = 2 * np.pi * t / n_frames
        g = (x[None, :] + y[:, None]) / 2 * 0.5 + 0.5
        img = np.stack([g * (0.6 + 0.4 * np.sin(phase)),
                        g * (0.6 + 0.4 * np.sin(phase + 2.1)),
                        g * (0.6 + 0.4 * np.sin(phase + 4.2))], axis=-1)
        cx = int(w * (0.25 + 0.5 * (0.5 + 0.5 * np.sin(phase))))
        cy = int(h * (0.25 + 0.5 * (0.5 + 0.5 * np.cos(phase * 1.3))))
        cv2.circle(img, (cx, cy), 40, (1.0, 0.9, 0.6), -1)
    elif kind == "checker":
        # 漂移棋盘格
        x = np.arange(w)[None, :].astype(np.float32) + 4 * t
        y = np.arange(h)[:, None].astype(np.float32) - 3 * t
        grid = ((x // 32 + y // 32) % 2).astype(np.float32)
        img = np.stack([grid * 0.7, grid * 0.5, grid * 0.9], axis=-1)
    else:  # "noise"：缓慢漂移的平滑噪声纹理
        r = rng.random((h // 8 + 4, w // 8 + 4, 3), dtype=np.float32)
        r = cv2.resize(r, (w, h), interpolation=cv2.INTER_CUBIC)
        shift = int(t * 1.5) % 8
        img = np.roll(r, shift, axis=0) * 0.7 + 0.15
    return np.clip(img, 0, 1)


def render_video(kind, severity, n_frames, w=480, h=270, fps=24, rng=None):
    """渲染一个带闪烁失真的视频帧序列 [n_frames, h, w, 3] uint8。

    severity s: 0 无失真，1 最严重。
    """
    rng = rng or np.random.default_rng(0)
    frames = []
    prev = None
    for t in range(n_frames):
        img = base_scene(kind, t, n_frames, w, h, rng)
        # 1) 亮度闪烁：周期性亮度抖动
        freq = rng.uniform(3, 8)
        phase = rng.uniform(0, 2 * np.pi)
        img = img * (1.0 + severity * 0.5 * np.sin(2 * np.pi * freq * t / fps + phase))
        # 2) 帧冻结卡顿：按概率冻结上一帧（时域断裂）
        if prev is not None and rng.random() < severity * 0.4:
            img = prev
        # 3) 时域噪声
        if severity > 0:
            img = img + rng.normal(0, severity * 15 / 255.0, img.shape)
        img = np.clip(img, 0, 1)
        prev = img
        frames.append((img * 255).astype(np.uint8))
    return frames


def make_dataset(out_dir, n_videos, n_frames=48, fps=24, size=(480, 270), seed=42):
    videos_dir = os.path.join(out_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    kinds = ["gradient", "checker", "noise"]
    labels = []

    for i in range(1, n_videos + 1):
        severity = float(rng.uniform(0.05, 0.95))
        mos = round(100 * (1 - severity), 1)
        kind = kinds[i % len(kinds)]
        frames = render_video(kind, severity, n_frames, w=size[0], h=size[1],
                              fps=fps, rng=np.random.default_rng(seed + i))
        name = f"video{i}.mp4"
        path = os.path.join(videos_dir, name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for fr in frames:
            writer.write(fr)
        writer.release()
        labels.append((name, mos))
        print(f"  {name}: MOS={mos} (severity={severity:.2f}, scene={kind})")

    # 按自然序写 labels.txt（video2 排在 video10 前）
    labels.sort(key=lambda kv: natural_key(kv[0]))
    labels_path = os.path.join(out_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for name, mos in labels:
            f.write(f"{name}: {mos}\n")
    print(f"生成完成: {n_videos} 个视频 -> {videos_dir}")
    print(f"标注文件: {labels_path}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="合成闪烁数据集生成器")
    p.add_argument("--n", type=int, default=24, help="视频数量")
    p.add_argument("--frames", type=int, default=48, help="每个视频帧数")
    p.add_argument("--fps", type=int, default=24, help="帧率")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/synthetic")
    args = p.parse_args()
    make_dataset(args.out, args.n, n_frames=args.frames, fps=args.fps,
                 size=(args.width, args.height), seed=args.seed)


if __name__ == "__main__":
    main()
