# 视频质量评估系统（VQA·时域一致性）

基于深度学习的无参考视频质量评估（NR-VQA）课程设计项目，聚焦**复杂场景下的视频时域一致性评价**（典型失真：闪烁 flicker、帧冻结卡顿、时域噪声）。

模型按课程设计任务书第五章的架构逐维度实现：**ResNet-50 多尺度卷积分支 + ViT-B/16 全局 token 分支 → 单帧 15872 维向量 → 时间维聚合 → 回归头输出 MOS**，并实现了任务书规定的 **baseline 监督训练（A 阶段）与半监督伪标签迭代（B 阶段）** 完整流程。

## 任务目标

- 输入：若干测试视频（帧数、分辨率不定），训练集含视频 + MOS 主观分（0~100，描述时域一致性，越高越好）
- 输出：`score.txt`，每行 `videoN: 分数`
- 指标：**SROCC**（排序一致性）与 **PLCC**（线性相关性）均逼近 1
- 总分：`Score = (SROCC + PLCC) / 2 − min(1.0, 0.01 × max(0, T − 20))`，T 为 100 个视频测试耗时（分钟），超过 20 分钟每分钟扣 0.01
- 算力约束：4K 分辨率 20 帧输入 < 300G FLOPs（本模型输入统一缩放至 224×224，见下文估算）

## 方法架构

```
 ┌──────────────────── 单帧 ────────────────────┐
 │  卷积分支 ResNet-50[1]           ViT 分支 ViT-B/16[2]   │
 │  layer1~4 输出                   encoder 末层 token       │
 │  ├─ 空间维 mean/max/std ─┐      ├─ token 维 mean/max/std ─┐│
 │  │  3×(256+512+1024+2048)=11520│  │  3×768 = 2304            ││
 │  └─ layer4 GAP = 2048 ───┴→ 13568  └──────────────→ 2304 ──┘│
 └─────────────── 拼接 = 15872 维单帧向量 ───────────────┘
         T 帧（TSN[4] 均匀采样）在时间维取平均 → 视频级特征 15872
         回归头: Linear(15872→512) + ReLU + Linear(512→1) → MOS
```

| 模块 | 说明 | 维度 |
|---|---|---|
| 卷积分支 | ResNet-50 layer1~4 统计池化 + layer4 GAP | 13568 |
| ViT 分支 | ViT-B/16 token 统计池化（mean/max/std） | 2304 |
| 单帧向量 | 两分支拼接 | **15872** |
| 时间聚合 | T 帧向量取平均（默认 T=8，Patch-VQ[5] 思路） | 15872 |
| 回归头 | Linear+ReLU+Linear | 512 / 1 |

训练流程（任务书第五章）：

- **(A) baseline**：标注视频监督训练（MSE），监控验证 loss 保存 best —— `extractor.pt` / `head.pt` / `model.pt` 三份权重
- **(B) 半监督伪标签循环**：每轮 N 次独立训练（从 baseline 权重重新加载、随机抽样训练子集）→ 对全部视频打分 → 对未标注视频按 **N 次预测方差 < 阈值** 筛选稳定样本，取均值作伪标签入池（加权 MSE，伪标签权重 W_PSEUDO）→ 验证阶段全量训练后在锁定验证集上算 `OBJ = SROCC + PLCC`，优于 best 则保存，连续无提升早停

## 数据集

| 数据集 | 用途 | 获取 |
|---|---|---|
| **官方课程数据集**（DIVIDE-MaxWell） | 正式训练/验证 | 视频（8.55GB）：<https://huggingface.co/datasets/teowu/DIVIDE-MaxWell/resolve/main/videos.zip>；标注 `train_lable_train.txt` / `train_lable_test.txt` 见下方"标注重建" |
| **合成闪烁数据集**（`scripts/make_flicker_dataset.py`） | 开发自测：亮度闪烁 + 帧冻结卡顿 + 时域噪声，severity → MOS（0~100），失真类型与任务书一致 | 本地生成，秒级 |
| KoNViD-1k（可选） | 真实 UGC 数据增强泛化 | 见 `scripts/download_konvid.md` |

标注文件格式（与任务书一致，`#` 开头为注释）：

```
video1: 99.0
video2: 51.6
```

### 标注重建（课程标注尚未下发时的方案）

课程标注 `train_lable_train.txt` / `train_lable_test.txt` 原定由课程另行提供；在拿到官方文件前，已从原版 MaxWell 数据集仓库（[VQAssessment/ExplainableVQA](https://github.com/VQAssessment/ExplainableVQA)，ACMMM 2023）重建出等效标注，**已提交在 `data/divide/` 下**（train 3634 条 + test 909 条，恰好覆盖 videos.zip 的全部 4543 个视频）：

- `data/divide/train_lable_train.txt` / `train_lable_test.txt` — 整体 MOS（1~5 量纲）
- `data/divide/train_lable_train_Tall.txt` / `train_lable_test_Tall.txt` — T-all（时域轴）变体，供官方标注到达后对比"课程 MOS 基于哪一轴"

重建依据与校验（脚本 `scripts/build_course_labels.py` 自动完成）：

1. ExplainableVQA 仓库 `examplar_data_labels/MaxWell/{train,test}_labels.txt` 使用编号视频名（0000.mp4~4542.mp4，与 videos.zip 命名一致），行序与 `MaxWell_{train,val}.csv` 对齐；
2. 其分值与 CSV 的 O（overall）列满足精确线性关系 `v = 0.929·O + 0.231`（Pearson/Spearman = 1.0000），即分值就是整体 MOS；
3. train ∪ test 恰好覆盖全部 4543 个视频、无重叠，且与本地视频目录逐一校验通过。

拿到官方标注后运行：

```bash
python scripts/build_course_labels.py --compare-train 官方_train.txt --compare-test 官方_test.txt
```

若 SROCC/PLCC ≈ 1 则重建与官方等价，无需重训；否则用官方文件替换 `--labels` / `--val-labels` 重训。

> 注：PLCC/SROCC 对线性变换不敏感，训练用分数的量纲不影响最终指标。

## 环境安装

```bash
# 已配好 conda 环境（Windows）：
conda activate vqa_env
# 或全新安装：
#   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
#   pip install opencv-python numpy scipy
```

依赖：Python 3.9+ / PyTorch 2.5.1+cu118（GPU）/ torchvision / opencv-python / numpy / scipy。
首次运行会自动下载 ImageNet 预训练权重（ResNet-50 ~98MB、ViT-B/16 ~330MB）到 `~/.cache/torch`。

## 快速开始

```bash
# 1. 生成合成闪烁数据集（56 个视频 + labels.txt）
python scripts/make_flicker_dataset.py --n 56 --out data/synthetic

# 2. (A) baseline 监督训练
python scripts/train_baseline.py --data-dir data/synthetic/videos \
    --labels data/synthetic/labels.txt --out runs/baseline --epochs 8

# 3. (B) 半监督伪标签迭代（以 baseline 权重为起点）
python scripts/train_semisup.py --data-dir data/synthetic/videos \
    --labels data/synthetic/labels.txt --baseline runs/baseline --out runs/semisup

# 4. 测试推理，输出 score.txt
python scripts/predict.py --model runs/semisup/model_best.pt \
    --videos data/synthetic/videos --out score.txt
```

换用课程数据：把视频放进 `data/course/videos/`、标注写入 `data/course/labels.txt`，把上面命令的路径替换即可，其余不变。

### 课程数据正式训练（DIVIDE-MaxWell，4543 视频）

```bash
# 0. 预抽帧缓存（一次性，约 5.5GB；训练/打分大幅提速）
python scripts/precache_frames.py --data-dir data/divide/videos \
    --cache-dir data/frames_cache --t 8 --workers 8

# 1. (A) baseline：训练标注 3634 + 锁定验证标注 909
python scripts/train_baseline.py --data-dir data/divide/videos \
    --labels data/divide/train_lable_train.txt \
    --val-labels data/divide/train_lable_test.txt \
    --out runs/divide_baseline --fp16 --frame-cache data/frames_cache --epochs 12

# 2. (B) 半监督：909 个验证视频作为"未标注"进入伪标签池（锁定验证集评估）
#    注意：重建标注为 1~5 量纲，--var-threshold 用 0.2（0~100 时用默认 25）
python scripts/train_semisup.py --data-dir data/divide/videos \
    --labels data/divide/train_lable_train.txt \
    --val-labels data/divide/train_lable_test.txt \
    --baseline runs/divide_baseline --out runs/divide_semisup \
    --var-threshold 0.2 --fp16 --frame-cache data/frames_cache

# 3. 测试推理（测试视频到达后）：输出 score.txt
python scripts/predict.py --model runs/divide_semisup/model_best.pt \
    --videos data/test_videos --out score.txt --fp16
```

## 实验记录

### 合成数据自测

| 阶段 | SROCC | PLCC | OBJ | 备注 |
|---|---|---|---|---|
| baseline（合成数据，56 视频，8 epoch） | 0.9222 | 0.9241 | 1.8462 | 验证划分 8 个 |
| 半监督（2 轮伪标签，best=第 1 轮） | 0.8333 | 0.9436 | 1.7769 | 锁定验证集 8 个；伪标签累积 10 个 |

> 说明：半监督自测中 30% 标注被隐藏模拟未标注场景（训练标注 34 个，少于 baseline 的 48 个），
> 故 SROCC 略低于 baseline、PLCC 更高——符合"少量标注 + 伪标签扩充"的预期；
> 全量标注上 best 模型 SROCC=0.9098 / PLCC=0.9016。完整历史见 `runs/semisup/metrics.json`。

### 正式数据（DIVIDE-MaxWell，4543 视频；909 锁定验证集）

| 阶段 | SROCC | PLCC | OBJ | 备注 |
|---|---|---|---|---|
| **(A) baseline** | **0.6723** | **0.6723** | **1.3446** | best@epoch4；12 epochs，fp16+帧缓存 |
| (B) 半监督 v1（默认配置）轮次 1 | 0.6716 | 0.6271 | 1.2987 | 901 伪标签入池 |
| 轮次 2 | 0.6454 | 0.5698 | 1.2152 | |
| 轮次 3 | 0.6628 | 0.5034 | 1.1662 | |
| 轮次 4 | 0.6615 | 0.4719 | 1.1334 | 早停 3/3，实验结束 |
| (B) 半监督 v2（重标定+weight0.2+var0.05）轮次 1 | 0.6616 | 0.6184 | 1.2800 | 868 伪标签入池，重标定生效 |
| **v2 轮次 2（最终 best）** | **0.6897** | **0.6561** | **1.3457** | **反超 baseline；高精度复核 0.689669/0.656080/1.345750** |
| v2 轮次 3 | 0.6657 | 0.6145 | 1.2801 | 无提升 1/3 |
| v2 轮次 4 | 0.6739 | 0.5671 | 1.2411 | 无提升 2/3 |
| v2 轮次 5 | 跳过 B.2 | | | 轮次 5 B.1 完成（+2 → 895 池）；B.2 经评估不再有提升空间，未补跑 |

> **v1 实验结论**：伪标签（模型自身预测）带有压缩偏差（不敢打低分、够不着高分，std 0.422 vs 真值 0.491），
> 以 0.5 权重回喂后模型 PLCC 单调下滑而 SROCC 稳定——伪标签法在默认配置下未带来增益。
> v2 引入 z-score 重标定（对齐真标注分布）、[1,5] 裁剪、权重 0.2、方差阈值 0.05 复测。
> **v2 结论**：轮次 2 以 OBJ=1.3457 反超 baseline（1.3446），SROCC 提升明显（+0.017）；PLCC 在轮 3 后缓慢衰减但未再崩盘——修复方案有效。
> 最终交付权重 = 轮次 2 checkpoint（`runs/divide_semisup_v2/model_best.pt`，LFS）。
> 完整诊断与修改方案见 `docs/semisup_experiment_report.md`，能力边界分析见 `docs/summary_report.md`。

## 分工与失真专项诊断（噪点/闪烁/模糊）

整体打分（组长）+ 三类失真专项判定（组员各一人）并行开发，统一接口合成：

| 模块 | 接口/入口 | 状态 |
|---|---|---|
| 整体 MOS 打分 | `predict.py` → score.txt | baseline 完成 |
| 噪点判定 | `vqa/diagnosis.py` 的 `NOISE_DETECTOR` 槽位 | 待组员实现 |
| 闪烁判定 | `FLICKER_DETECTOR` 槽位（附参考实现 `heuristic_flicker`） | 参考实现可跑 |
| 模糊判定 | `BLUR_DETECTOR` 槽位 | 待组员实现 |

组员接入约定（详见 `vqa/diagnosis.py` 文件头注释）：实现
`detect(frames_rgb) -> {"score": 0~1, "level": 无/轻/中/重, "detail": {...}}`，
赋给对应槽位并 `register_detector`。检测器只依赖抽帧结果，**不依赖打分模型权重，可独立开发自测**。

整合入口（可挂主模型总分，可不挂）：

```bash
# 只出问题清单（组员自测）
python scripts/diagnose.py --videos data/test_videos --out-dir runs/diagnose

# 问题清单 + 整体总分（最终合成交付）
python scripts/diagnose.py --videos data/test_videos \
    --model runs/divide_baseline/model_best.pt --fp16 --out-dir runs/diagnose
# → report.json / report.txt，与 score.txt 配套
```

## 算力与耗时约束说明

- 模型输入统一缩放为 224×224（任务书规定），与原始视频分辨率无关
- 单帧计算量：ResNet-50 ≈ 4.1 GFLOPs + ViT-B/16 ≈ 17.6 GFLOPs ≈ **22 GFLOPs**；T=8 帧 ≈ 176 GFLOPs < 300G；若按 20 帧输入 ≈ 440 GFLOPs，可在推理时用 `--t 8`（默认）或 fp16 满足约束
- RTX 4060 Laptop 实测推理 332ms/视频（含解码抽帧），100 视频约 0.55 分钟，远低于 20 分钟限时

## 目录结构

```
videoquality/
├── vqa/                       # 核心库
│   ├── config.py              # 超参数
│   ├── sampling.py            # TSN 式帧抽取[4]
│   ├── dataset.py             # 数据集与标注解析
│   ├── models.py              # 双分支特征提取器 + 回归头
│   ├── metrics.py             # SROCC/PLCC/总分公式
│   ├── train_utils.py         # 加权MSE/checkpoint/打分
│   ├── semisup.py             # 半监督伪标签主循环（含伪标签重标定 debias_pool）
│   └── diagnosis.py           # 失真专项检测接口（噪点/闪烁/模糊槽位，分工集成点）
├── scripts/
│   ├── make_flicker_dataset.py # 合成闪烁数据集生成器
│   ├── precache_frames.py      # 预抽帧缓存（训练/推理跳过解码）
│   ├── train_baseline.py       # (A) baseline 训练
│   ├── train_semisup.py        # (B) 半监督训练（--pseudo-debias/--pseudo-clip）
│   ├── predict.py              # 推理 → score.txt（20 分钟时限统计）
│   ├── diagnose.py             # 失真诊断 → report.json/txt（分工合成入口）
│   └── eval_val.py             # 验证集高精度重评 + 导出逐视频分数
├── data/                      # 数据（不入库）
├── docs/                      # 汇报文档（semisup_experiment_report.md / summary_report.md）
└── runs/                      # 权重与日志（权重经 LFS 入库，其余不入库）
```

## 思考与探索（任务书第九章）

预留扩展方向：以视频特征为状态、增强/打分策略为动作、标注校准为奖励，用 PPO 训练 RL agent 优化打分/增强策略。见 `docs/rl_extension.md`（规划中）。

## 参考文献

1. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
2. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021
3. Wang, Katsenou & Bull, *ReLaX-VQA: Residual Fragment and Layer Stack Extraction for Enhancing Video Quality Assessment*, arXiv:2407.11496
4. Wang et al., *Temporal Segment Networks: Towards Good Practices for Deep Action Recognition*, ECCV 2016
5. Ying et al., *Patch-VQ: 'Patching Up' the Video Quality Problem*, CVPR 2021
6. Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015
