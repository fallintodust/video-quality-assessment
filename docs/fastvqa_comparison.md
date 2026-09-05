# SOTA 零样本对照实验（FAST-VQA）

> 负责人：ALEKSEEV PETR ｜ 完成时间：2026-09-05
> 原任务为 CAMP-VQA 对照（见 `docs/campvqa_handoff.md`），因硬件限制改用 FAST-VQA 完成，
> 理由与实测数据见第 1 节。

---

## 1. 为什么改用 FAST-VQA

### 1.1 CAMP-VQA 在本机不可行

交接文档要求显存至少 8GB。实测机器为 **RTX 3050 Ti Laptop（4GB）**，
WSL2 分配 12GB 内存。完整流程实测如下：

| 项目 | 实测值 |
|---|---|
| BLIP-2 flan-t5-xl 权重下载 | 15.8 GB / 27 分钟 |
| BLIP-2 fp16 加载耗时 | 347 s |
| 加载后 `memory_allocated` | 8.49 GB（4GB 卡靠 unified memory 溢出到内存） |
| 加 SlowFast/Swin-L/CLIP 后峰值 | 9.69 GB |
| 平均单次 caption 生成 | 9.71 s |
| **单视频（仅 BLIP-2 部分）** | **583 s ≈ 9.7 分钟** |
| **909 视频预估** | **147 小时 ≈ 6 天** |

遇到的阻塞问题：

1. **全链路 OOM**：BLIP-2 常驻显存后，Swin-Large 前向时
   `torch.OutOfMemoryError: Tried to allocate 74.00 MiB`。
2. **改为 BLIP-2 用完移回 CPU**：Swin/SlowFast 那一步正常（2min23s），
   但 `blip_model.to(device)` 触发
   `RuntimeError: CUDA driver error: device not ready`——
   4GB 卡反复搬运 ~10GB 权重，驱动无法承受。
3. **BLIP-2 全程留 CPU**：单次生成 >60s，单视频约 1 小时，100 个要 4 天。

结论：**不是慢的问题，是这台机器做不了。** 距答辩 6 天，147 小时不可能完成。

### 1.2 FAST-VQA 是等价的替代方案

FAST-VQA 与 CAMP-VQA、MaxWell 数据集出自**同一作者组**
（Haoning Wu 等，ECCV 2022 / TPAMI 2023），同样是 LSVQ 预训练的零样本模型，
对照结论的形式完全一致。

| | CAMP-VQA | FAST-VQA |
|---|---|---|
| 骨干 | BLIP-2 + Swin-L + CLIP + SlowFast | Video Swin-T |
| 权重体积 | ~16 GB | **127 MB** |
| 显存需求 | 8 GB+ | 可在 4 GB 上运行 |
| 单视频耗时 | 583 s | **7.47 s** |
| 909 视频 | 147 小时 | **113 分钟** |

---

## 2. 实验设置

- **模型**：FAST-VQA-B，Kinetics-400 预训练骨干 + LSVQ 训练，**零样本**，未在课程数据上做任何微调
- **权重**：`FAST_VQA_B_1*4.pth`（上游 release v2.0.0，127 MB）
- **抽帧**：fragments 7×7×32×32，clip_len 32，num_clips 4（上游 `options/fast/fast-b.yml` 的 `val-kv1k` 配置）
- **评测集**：`data/divide/train_lable_test.txt`，**909 个锁定验证视频**
- **标注轴**：O（整体 MOS），与组内 baseline / 半监督实验一致
- **分数映射**：上游 `sigmoid_rescale`（LSVQ 标定常数），线性/单调变换不影响 SROCC

运行命令：

```bash
python3 fastvqa_eval_batch.py \
    --videos-dir <视频目录> \
    --labels data/divide/train_lable_test.txt \
    --model FAST-VQA \
    --out fastvqa_909.txt
```

脚本见 `scripts/fastvqa_eval_batch.py`，产物在 `runs/fastvqa/`。

---

## 3. 结果

909 个视频全部成功打分，无失败样本，总耗时 113.21 分钟（7.47 s/视频）。

| 模型 | 训练方式 | SROCC | PLCC | OBJ |
|---|---|---|---|---|
| **FAST-VQA（零样本 SOTA）** | LSVQ 预训练，未见本数据 | **0.7102** | **0.7111** | **1.4213** |
| **冻结特征 + 回归头** | 课程数据训练 | 0.6950 | 0.7066 | 1.4016 |
| 任务书模型 半监督 v2 | 课程数据 + 伪标签 | 0.6897 | 0.6561 | 1.3457 |
| 任务书模型 baseline | 课程数据训练 | 0.6723 | 0.6723 | 1.3446 |

### 3.1 三点结论

**（1）零样本 SOTA 领先，但差距很小。**
FAST-VQA 在 LSVQ（3.9 万个带主观标注的视频）上训练，**从未见过 MaxWell 的任何样本**，
OBJ 仅比我们最好的自建模型高 **0.0197**。考虑到 47616 维配置的重复运行波动为 0.011
（见 `docs/cached_feature_pipeline2.md` 6.4 节），这个差距只有噪声的约 2 倍。

**（2）冻结特征方案已接近 SOTA 水平。**
在骨干完全冻结、只训练一个两层回归头的条件下达到 OBJ 1.4016，
与零样本 SOTA 差 0.02，且单次实验只需 3 分钟。

**（3）半监督 v2 相对 baseline 提升 0.0011。**
这个量级远低于噪声上界 0.011，**不能判定为有效提升**。
伪标签机制在本数据上没有产生实际收益。

### 3.2 讨论

FAST-VQA 领先的原因不在架构复杂度，而在**训练数据规模**：
LSVQ 有 3.9 万个带主观 MOS 的野生视频，而课程训练集只有 3634 个。
这说明在本任务上，数据量的作用大于模型设计——
用更多外部 VQA 数据做预训练，可能比继续调整半监督策略更有价值。

同时也应指出：本对照使用的是 **O 轴（整体质量）** 标注。
FAST-VQA 预测的正是整体质量，与该轴天然对齐；
若课程标注实为时域维度（T-5 抖动 / T-8 卡顿），
FAST-VQA 的优势未必成立——它没有针对时域一致性的专门设计。
标注轴的确认仍是待办事项。

---

## 4. 复现说明

### 4.1 环境

```bash
git clone https://github.com/VQAssessment/FAST-VQA-and-FasterVQA.git
cd FAST-VQA-and-FasterVQA
pip install -e . --no-deps        # --no-deps 避免覆盖已有 torch
pip install einops sk-video
```

国内网络：GitHub 直连不稳定，可用镜像前缀
`https://gh-proxy.com/` 或 `https://ghfast.top/`。

### 4.2 权重

从上游 release **v2.0.0**（"Refactorized weights!"）下载，放入 `pretrained_weights/`：

| 模型 | 文件 | MACs |
|---|---|---|
| FAST-VQA | `FAST_VQA_B_1_4.pth` | 279 G |
| FasterVQA | `FAST_VQA_3D_1_1.pth` | 69 G |
| FAST-VQA-M | `FAST_VQA_M_1_4.pth` | 46 G |

**注意两处坑**：

1. 不要用 v2.0.1（文件名带 `_Scr`）——那是从零训练的权重，无 Kinetics-400 预训练，指标明显更低。
2. release 里文件名是下划线（`FAST_VQA_B_1_4.pth`），而配置文件期望星号
   （`FAST_VQA_B_1*4.pth`），需要复制一份改名：
   ```bash
   cp FAST_VQA_B_1_4.pth "FAST_VQA_B_1*4.pth"
   ```

### 4.3 运行

脚本必须从仓库**根目录**运行，配置里的路径都是相对路径。

若之前跑过 CAMP-VQA，注意清除环境变量
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`——
它会导致 FAST-VQA 推理时 `CUDA driver error: device not ready`，开新终端即可。

---

## 5. 待办

- [ ] 补跑 FasterVQA（69 G MACs）与 FAST-VQA-M（46 G MACs），
      形成"计算量 vs 精度"对照，与任务书 300 G FLOPs 限制呼应（约 4 小时）
- [ ] 若组内有 8GB 以上显卡的机器，可交接 CAMP-VQA 全量实验
      （环境配置与脚本改动记录见本文第 1 节）
- [ ] 标注轴确认后，在正确的轴上重跑本对照
