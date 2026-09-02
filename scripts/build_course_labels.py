# -*- coding: utf-8 -*-
"""从 MaxWell 原始标签重建课程标注文件（train_lable_train.txt / train_lable_test.txt）。

背景：
  课程数据集 DIVIDE-MaxWell（https://huggingface.co/datasets/teowu/DIVIDE-MaxWell）
  只提供了 videos.zip（4543 个视频，命名为 0000.mp4 ~ 4542.mp4），标注文件
  train_lable_train.txt / train_lable_test.txt 由课程另行提供。

  本脚本利用原版 MaxWell 数据集仓库（VQAssessment/ExplainableVQA，ACMMM 2023）
  中发布的标注重建课程标注：
  - examplar_data_labels/MaxWell/train_labels.txt （3634 条，编号视频名 + 整体 MOS）
  - examplar_data_labels/MaxWell/test_labels.txt  （909 条）
  - MaxWell_train.csv / MaxWell_val.csv（O / A-* / T-* 多轴 MOS）

  已验证（scripts/ 内自查脚本结论）：
  - examplar 的行序与 CSV 行序一致，且其分值与 CSV 的 O 列满足
    v = 0.929 * O + 0.231（Pearson=1.0000），即 examplar 值就是整体 MOS（O 轴）；
  - train ∪ test 恰好覆盖 0000.mp4 ~ 4542.mp4 全部 4543 个视频，无重叠，
    与 data/divide/videos 中的文件一一对应。

  注意：PLCC / SROCC 对线性变换不敏感，训练用分数的量纲（1~5 或 0~100）
  不影响最终指标。官方标注拿到后可用 --compare 对比校验（若官方标注与
  重建的整体 MOS 秩相关 ≈1 则无需重训）。

用法：
  python scripts/build_course_labels.py                # 生成 data/divide/ 下标注
  python scripts/build_course_labels.py --compare-train 官方_train.txt --compare-test 官方_test.txt
"""
import argparse
import csv
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqa.dataset import parse_labels
from scipy.stats import spearmanr  # noqa: E402

RAW_BASE = ("https://raw.githubusercontent.com/VQAssessment/"
            "ExplainableVQA/master")
SOURCES = {
    "train_labels_examplar.txt": RAW_BASE + "/examplar_data_labels/MaxWell/train_labels.txt",
    "test_labels_examplar.txt": RAW_BASE + "/examplar_data_labels/MaxWell/test_labels.txt",
    "MaxWell_train.csv": RAW_BASE + "/MaxWell_train.csv",
    "MaxWell_val.csv": RAW_BASE + "/MaxWell_val.csv",
}


def ensure_sources(cache_dir):
    """缺失时从 GitHub 下载源文件，返回 {名称: 本地路径}。"""
    paths = {}
    for name, url in SOURCES.items():
        p = os.path.join(cache_dir, name)
        if not os.path.isfile(p):
            print(f"下载 {name} <- {url}")
            urllib.request.urlretrieve(url, p)
        paths[name] = p
    return paths


def load_examplar(path):
    """解析 examplar 标注：[(编号 int, MOS float), ...]，保持文件行序。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            num = int(parts[0].split(".")[0])
            rows.append((num, float(parts[-1])))
    return rows


def load_csv_column(path, col):
    with open(path, encoding="utf-8") as f:
        reader = list(csv.reader(f))
    header = reader[0]
    i = header.index(col)
    return [float(row[i]) for row in reader[1:]]


def build(video_dir, cache_dir):
    src = ensure_sources(cache_dir)
    train_rows = load_examplar(src["train_labels_examplar.txt"])
    test_rows = load_examplar(src["test_labels_examplar.txt"])

    # 行序对齐：CSV 第 i 行 <-> examplar 第 i 行（已用 Pearson=1.0 验证）
    tr_tall = load_csv_column(src["MaxWell_train.csv"], "T-all")
    te_tall = load_csv_column(src["MaxWell_val.csv"], "T-all")
    assert len(tr_tall) == len(train_rows) and len(te_tall) == len(test_rows)

    all_rows = train_rows + test_rows
    tall_by_num = {}
    for rows, tall_col in ((train_rows, tr_tall), (test_rows, te_tall)):
        for (num, _), t in zip(rows, tall_col):
            tall_by_num[num] = t

    # 校验：并集恰好覆盖 0000~4542
    nums = [n for n, _ in all_rows]
    assert len(set(nums)) == len(nums), "标注编号有重复"
    assert set(nums) == set(range(len(nums))), "编号未覆盖 0000~N-1 或越界"

    # 校验：视频目录中文件齐全
    missing = [n for n in nums
               if not os.path.isfile(os.path.join(video_dir, f"{n:04d}.mp4"))]
    if missing:
        print(f"警告：视频目录缺少 {len(missing)} 个文件，如 {missing[:5]}")
    else:
        print(f"视频目录校验通过：{len(nums)} 个视频全部存在")

    def write(path, rows, tall_map=None):
        with open(path, "w", encoding="utf-8") as f:
            for num, mos in rows:
                f.write(f"{num:04d}.mp4: {mos:.3f}\n")
        print(f"写出 {path}（{len(rows)} 条，MOS {min(v for _, v in rows):.3f}~"
              f"{max(v for _, v in rows):.3f}）")

    write(os.path.join(video_dir, "..", "train_lable_train.txt"), train_rows)
    write(os.path.join(video_dir, "..", "train_lable_test.txt"), test_rows)
    # T-all 变体：官方标注拿到后用于对比"课程 MOS 到底基于哪一轴"
    write(os.path.join(video_dir, "..", "train_lable_train_Tall.txt"),
          [(n, tall_by_num[n]) for n, _ in train_rows])
    write(os.path.join(video_dir, "..", "train_lable_test_Tall.txt"),
          [(n, tall_by_num[n]) for n, _ in test_rows])

    # 全量对照表（视频名 / 整体MOS / T-all），供排查用
    csv_path = os.path.join(video_dir, "..", "all_video_labels.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "overall_mos", "T_all"])
        for n, v in sorted(all_rows):
            w.writerow([f"{n:04d}.mp4", f"{v:.3f}", f"{tall_by_num[n]:.4f}"])
    print(f"写出 {csv_path}（全量对照表）")


def compare(rebuilt_path, official_path):
    """官方标注 vs 重建标注：逐视频对齐后算 SROCC/PLCC。"""
    rebuilt = parse_labels(rebuilt_path)
    official = parse_labels(official_path)
    def key(name):
        m = re.search(r"\d+", name)
        return int(m.group()) if m else name

    common = set(rebuilt) & set(official)
    if not common:
        # 命名不同（如 video1 vs 0000.mp4）：按数字编号对齐
        rb = {key(n): v for n, v in rebuilt.items()}
        of = {key(n): v for n, v in official.items()}
        common = set(rb) & set(of)
        xs, ys = ([of[i] for i in sorted(common)],
                  [rb[i] for i in sorted(common)])
        print(f"按数字编号对齐 {len(common)} 个视频")
    else:
        xs = [official[n] for n in sorted(common)]
        ys = [rebuilt[n] for n in sorted(common)]
    srocc, _ = spearmanr(xs, ys)
    m_x, m_y = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - m_x) * (y - m_y) for x, y in zip(xs, ys))
    vx = sum((x - m_x) ** 2 for x in xs)
    vy = sum((y - m_y) ** 2 for y in ys)
    plcc = cov / (vx * vy) ** 0.5
    print(f"对齐 {len(common)} 个视频：SROCC={srocc:.4f} PLCC={plcc:.4f}")
    return srocc, plcc


def main():
    p = argparse.ArgumentParser(description="重建课程标注文件")
    p.add_argument("--video-dir", default="data/divide/videos")
    p.add_argument("--cache-dir", default="data/maxwell")
    p.add_argument("--compare-train", metavar="官方_train.txt",
                   help="只做对比：官方训练标注 vs 重建标注")
    p.add_argument("--compare-test", metavar="官方_test.txt",
                   help="只做对比：官方验证标注 vs 重建标注")
    args = p.parse_args()
    if args.compare_train:
        compare(os.path.join("data/divide", "train_lable_train.txt"),
                args.compare_train)
    if args.compare_test:
        compare(os.path.join("data/divide", "train_lable_test.txt"),
                args.compare_test)
    if not (args.compare_train or args.compare_test):
        build(args.video_dir, args.cache_dir)


if __name__ == "__main__":
    main()
