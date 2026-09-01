# KoNViD-1k 可选数据集（真实 UGC 视频质量数据）

KoNViD-1k（1200 个真实用户生成视频 + MOS，540p，每段 8s）可用于增强模型泛化。
本课程课题的正式数据为课程提供的闪烁数据集，此数据集仅为可选预训练/泛化实验用。

## 下载

- 官方来源（Zenodo，约 30GB zip）：https://doi.org/10.5281/zenodo.3928561
  （KonVid-1k 数据集页：https://database.mmsp-kn.de/konvid-1k-database.html）
- 下载后解压得到 `KoNViD_1k_videos/*.mp4` 与 `KoNViD_1k_attributes.csv`
  （CSV 中的 `MOS` 列为 1~5 分制主观分）

## 转换为本项目格式

```bash
# 将 1~5 分 MOS 线性映射到 0~100：
#   mos100 = (mos - 1) * 25
python scripts/convert_konvid.py --csv KoNViD_1k_attributes.csv \
    --videos KoNViD_1k_videos --out data/konvid
```

产物：`data/konvid/videos/`（软拷贝或符号链接原视频）+ `data/konvid/labels.txt`（`videoN: MOS`）。
随后即可用 `train_baseline.py --data-dir data/konvid/videos --labels data/konvid/labels.txt` 训练。

> 注：KoNViD-1k 是通用自然失真（压缩/传输噪声），不专门针对闪烁；
> 与本课题的时域一致性任务分布有差异，仅建议用于预训练增强，不宜替代课程数据评估。
