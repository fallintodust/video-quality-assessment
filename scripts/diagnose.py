"""失真专项诊断：对视频目录输出统一诊断报告（分工集成入口）。

- 每个视频：调用 vqa/diagnosis.py 中已注册的检测器（噪点/闪烁/模糊）+ 可选主模型 MOS
- 输出 report.json（结构化）与 report.txt（可读版），与 score.txt 配套交付

用法：
    python scripts/diagnose.py --videos data/test_videos \
        --model runs/divide_baseline/model_best.pt --fp16 \
        --out-dir runs/diagnose
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqa.config import Config
from vqa.diagnosis import _DETECTORS, diagnose_directory
from vqa.train_utils import build_model, get_device


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    p = argparse.ArgumentParser(description="失真专项诊断（噪点/闪烁/模糊 + 整体 MOS）")
    p.add_argument("--videos", default="data/test_videos", help="视频目录")
    p.add_argument("--model", default="", help="主模型权重（可选，提供后附带整体 MOS）")
    p.add_argument("--out-dir", default="runs/diagnose", help="报告输出目录")
    p.add_argument("--fp16", action="store_true", help="fp16 推理加速")
    p.add_argument("--t", type=int, default=8, help="每视频抽帧数")
    p.add_argument("--frame-cache", default="",
                   help="预抽帧缓存目录（可选，加速）")
    args = p.parse_args()

    Config.FRAME_CACHE = args.frame_cache or None
    device = get_device()
    print(f"设备: {device}  fp16: {args.fp16}")
    print(f"已注册检测器: {[n for n, _ in _DETECTORS] or '（无）'}")
    if not _DETECTORS:
        print("警告: 没有已注册的失真检测器，仅输出整体 MOS（如有）")

    model = None
    if args.model:
        model = build_model(device, T=args.t)
        state = __import__("torch").load(args.model, map_location="cpu",
                                         weights_only=True)
        model.extractor.load_state_dict(state["extractor"])
        model.head.load_state_dict(state["head"])
        print(f"已加载权重: {args.model}")

    results = diagnose_directory(args.videos, T=args.t, model=model,
                                 device=device, use_fp16=args.fp16)
    if not results:
        print(f"错误: {args.videos} 下没有找到视频文件")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "report.json")
    txt_path = os.path.join(args.out_dir, "report.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            mos = f"{r['mos']:.2f}" if r["mos"] is not None else "-"
            issues = "，".join(
                f"{k}:{v['level']}({v['score']:.2f})"
                for k, v in r["issues"].items() if v["score"] == v["score"])
            f.write(f"{r['video']}  总分={mos}  问题[{issues or '无'}]\n")
            print(f"  {r['video']}  总分={mos}  问题[{issues or '无'}]")

    print(f"报告已写入: {json_path} / {txt_path}")


if __name__ == "__main__":
    main()
