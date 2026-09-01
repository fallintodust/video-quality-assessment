"""KoNViD-1k 转换脚本（可选数据集，见 download_konvid.md）。

把 KoNViD-1k 的 CSV（1~5 分 MOS）转换为本项目格式：
    data/konvid/videos/  （软链/复制原视频，重命名 video1.mp4 ...）
    data/konvid/labels.txt （videoN: MOS，映射到 0~100）

用法：
    python scripts/convert_konvid.py --csv KoNViD_1k_attributes.csv \
        --videos KoNViD_1k_videos --out data/konvid
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqa.dataset import natural_key


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="KoNViD_1k_attributes.csv 路径")
    p.add_argument("--videos", required=True, help="KoNViD_1k_videos 目录")
    p.add_argument("--out", default="data/konvid")
    p.add_argument("--copy", action="store_true", help="复制视频而非软链")
    args = p.parse_args()

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["flickr_id"], float(r["MOS"])))

    out_videos = os.path.join(args.out, "videos")
    os.makedirs(out_videos, exist_ok=True)
    labels = []
    for i, (fid, mos5) in enumerate(sorted(rows, key=lambda r: natural_key(r[0])), start=1):
        src = os.path.join(args.videos, f"{fid}.mp4")
        if not os.path.isfile(src):
            print(f"跳过缺失视频 {fid}")
            continue
        dst = os.path.join(out_videos, f"video{i}.mp4")
        if args.copy:
            import shutil
            shutil.copy(src, dst)
        else:
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                import shutil
                shutil.copy(src, dst)  # Windows 无软链权限时回退复制
        mos100 = round((mos5 - 1.0) * 25.0, 1)
        labels.append((f"video{i}.mp4", mos100))

    with open(os.path.join(args.out, "labels.txt"), "w", encoding="utf-8") as f:
        for name, mos in labels:
            f.write(f"{name}: {mos}\n")
    print(f"完成: {len(labels)} 个视频 -> {args.out}")


if __name__ == "__main__":
    main()
