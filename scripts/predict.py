"""测试推理：对测试集视频打分并输出 score.txt（任务书第六章）。

输入：若干测试视频（帧数/分辨率不定，程序统一抽 T 帧缩放到 224x224）
输出：score.txt，每行 `videoN: 分数`，按视频编号自然排序
同时统计耗时并按任务书公式估算 100 视频测试的扣分情况。

用法：
    python scripts/predict.py --model runs/semisup/model_best.pt \
        --videos data/test_videos --out score.txt
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vqa.config import Config
from vqa.dataset import VideoDataset, list_videos, natural_key
from vqa.metrics import total_score
from vqa.train_utils import build_model, get_device, score_videos


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="VQA 推理，输出 score.txt")
    p.add_argument("--model", default="runs/semisup/model_best.pt",
                   help="模型权重文件（model.pt / model_best.pt）")
    p.add_argument("--videos", default="data/test_videos", help="测试视频目录")
    p.add_argument("--out", default="score.txt", help="输出文件")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--t", type=int, default=8, help="每视频抽帧数")
    p.add_argument("--precision", type=int, default=2, help="分数小数位")
    p.add_argument("--fp16", action="store_true", help="fp16 推理加速")
    p.add_argument("--frame-cache", default="",
                   help="预抽帧缓存目录（scripts/precache_frames.py 生成）")
    args = p.parse_args()

    Config.FRAME_CACHE = args.frame_cache or None
    device = get_device()
    print(f"设备: {device}  fp16: {args.fp16}")

    names = list_videos(args.videos)
    if not names:
        print(f"错误: {args.videos} 下没有找到视频文件")
        sys.exit(1)
    print(f"测试视频: {len(names)} 个")

    model = build_model(device, T=args.t)
    if not os.path.isfile(args.model):
        print(f"错误: 权重文件不存在 {args.model}")
        sys.exit(1)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    model.extractor.load_state_dict(state["extractor"])
    model.head.load_state_dict(state["head"])
    print(f"已加载权重: {args.model}")

    ds = VideoDataset.unlabeled(args.videos, names, T=args.t)
    t0 = time.time()
    scores = score_videos(model, ds, device, batch_size=args.bs,
                          use_fp16=args.fp16)
    elapsed = time.time() - t0

    with open(args.out, "w", encoding="utf-8") as f:
        for n in sorted(scores, key=natural_key):
            f.write(f"{n}: {scores[n]:.{args.precision}f}\n")

    # 耗时统计（任务书：100 视频超 20 分钟开始扣分）
    per_video = elapsed / len(names)
    minutes_100 = per_video * 100 / 60.0
    print(f"推理耗时: {elapsed:.1f}s ({per_video*1000:.0f}ms/视频)")
    print(f"估算 100 视频测试耗时: {minutes_100:.2f} 分钟 "
          f"({'超时扣分' if minutes_100 > 20 else '无超时扣分'})")
    print(f"结果已写入: {args.out}")
    for line in open(args.out, encoding="utf-8"):
        print("  " + line.rstrip())


if __name__ == "__main__":
    main()
