"""（B）阶段半监督训练入口（任务书第五章 B 部分）。

以 baseline 权重为起点，循环：伪标签生成 -> 验证/微调 -> 早停。

用法：
    python scripts/train_semisup.py --data-dir data/synthetic/videos \
        --labels data/synthetic/labels.txt --baseline runs/baseline --out runs/semisup
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqa.config import Config
from vqa.semisup import run_semisup


def main():
    p = argparse.ArgumentParser(description="半监督伪标签训练")
    p.add_argument("--data-dir", default="data/synthetic/videos", help="视频目录")
    p.add_argument("--labels", default="data/synthetic/labels.txt", help="标注文件")
    p.add_argument("--baseline", default="runs/baseline", help="baseline 权重目录")
    p.add_argument("--out", default="runs/semisup", help="输出目录")
    p.add_argument("--pseudo-rounds", type=int, default=Config.PSEUDO_ROUNDS)
    p.add_argument("--n-runs", type=int, default=Config.N_RUNS, help="每轮独立训练次数")
    p.add_argument("--sub-ratio", type=float, default=Config.SUB_RATIO)
    p.add_argument("--sub-epochs", type=int, default=Config.SUB_EPOCHS)
    p.add_argument("--var-threshold", type=float, default=Config.VAR_THRESHOLD)
    p.add_argument("--pseudo-weight", type=float, default=Config.PSEUDO_WEIGHT)
    p.add_argument("--val-epochs", type=int, default=Config.VAL_EPOCHS)
    p.add_argument("--early-stop", type=int, default=Config.EARLY_STOP)
    p.add_argument("--hide-ratio", type=float, default=Config.HIDE_RATIO,
                   help="把多少比例的标注样本隐藏为'未标注'（模拟半监督场景；"
                        "给定 --val-labels 时忽略）")
    p.add_argument("--val-ratio", type=float, default=Config.VAL_RATIO)
    p.add_argument("--val-labels", default="",
                   help="独立验证标注文件（锁定验证集；目录中不在训练标注里的视频"
                        "视为'未标注'进入伪标签池）")
    p.add_argument("--fp16", action="store_true", help="fp16 混合精度训练")
    p.add_argument("--frame-cache", default="",
                   help="预抽帧缓存目录（scripts/precache_frames.py 生成）")
    p.add_argument("--bs", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=Config.LR)
    p.add_argument("--t", type=int, default=Config.T)
    p.add_argument("--seed", type=int, default=Config.SEED)
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()

    # 阈值/权重/帧缓存写入全局配置（供 semisup 内部使用）
    Config.VAR_THRESHOLD = args.var_threshold
    Config.PSEUDO_WEIGHT = args.pseudo_weight
    Config.FRAME_CACHE = args.frame_cache or None
    run_semisup(args)


if __name__ == "__main__":
    main()
