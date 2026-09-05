# CAMP-VQA 对照实验交接文档（Peter 负责）

> 交接时间：2026-09-05 ｜ 负责人：Peter（ALEKSEEV PETR）｜ 组长：Vzixing ｜ 答辩：2026-09-11
> **English summary**：Run CAMP-VQA (WACV 2026 SOTA, zero-shot) on our 909 validation videos, compute SROCC/PLCC against our labels, and compare with our model (SROCC=0.6897 / PLCC=0.6561 / OBJ=1.3457). Everything you need is in this doc + `docs/campvqa/` scripts. Report results and progress to the group lead.

---

## 1. 任务目标（Goal）

用 **CAMP-VQA（WACV 2026，MaxWell 作者组 SOTA 模型，零样本）** 对我们课程数据的 **909 个锁定验证视频**打分，计算 SROCC / PLCC，与任务书模型对比，把结果写进答辩报告。

**对比基准（我们的最终模型，半监督 v2 best=轮次 2）**：

| 模型 | SROCC | PLCC | OBJ |
|---|---|---|---|
| 任务书模型 baseline | 0.6723 | 0.6723 | 1.3446 |
| **任务书模型 半监督 v2（最终）** | **0.6897** | **0.6561** | **1.3457** |
| CAMP-VQA（待你跑出） | ? | ? | ? |

若 CAMP-VQA 零样本指标明显更高，报告里如实说明"SOTA 大模型更强，但任务书架构是自建全链路课程目标"；若接近或更低，则说明自建模型在课程数据上具有竞争力——两种结果都有价值。

## 2. 环境与代码准备

### 2.1 代码

```bash
git clone https://github.com/xinyiW915/CAMP-VQA.git
cd CAMP-VQA
```

然后把本仓库（video-quality-assessment，`main` 分支）里 `docs/campvqa/` 下两个脚本复制到 `CAMP-VQA/src/`：
- `blip2_smoke_gpu.py` —— 冒烟测试（显存 + 单视频耗时估算，先跑这个）
- `campvqa_eval_batch.py` —— 批量打分（含 8GB 显存错峰策略 + 自动算 SROCC/PLCC）

这两个脚本是组内为本实验写的适配（官方 demo 在 8GB 显存上会 OOM），其余代码用上游原版。

### 2.2 Python 环境

```bash
conda create -n campvqa python=3.10 -y
conda activate campvqa
pip install -r requirements_fixed.txt
# 若 pip 慢（国内网络）：加镜像
# pip install -r requirements_fixed.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

要点：CUDA 版 torch（cu124，驱动 581.80+ 兼容）；显存至少 **8GB**（脚本按 8GB 设计，BLIP-2 用 fp16）。

### 2.3 权重下载（约 16GB，先下载再跑）

| 组件 | 来源 | 说明 |
|---|---|---|
| BLIP-2 flan-t5-xl | HuggingFace `Salesforce/blip2-flan-t5-xl` | ~15GB；国内网络用镜像：`export HF_ENDPOINT=https://hf-mirror.com` |
| Swin-Large | timm 自动下载（走 HF 镜像） | `swin_large_patch4_window7_224` |
| CLIP ViT-B/32 | `clip.load("ViT-B/32")` 自动下载 | ~340MB |
| SlowFast R50 | torch.hub fb CDN 自动下载 | 官方 CDN 实测可直连 |
| CAMP-VQA MLP 权重 | 上游仓库 `model/lsvq_train_camp-vqa_Mlp_byrmse_trained_model_kfold.pth` | 仓库自带（batch 脚本默认路径） |

### 2.4 数据（909 验证视频 + 标注）

- 视频：课程官方 DIVIDE-MaxWell 数据集（与组长拿到的 videos.zip 同一份）。只需要 909 个视频，名单在组长仓库 `data/divide/train_lable_test.txt`（格式 `视频名: MOS`）。
- 获取方式：**向组长要 909 视频子集的网盘链接**，或自行解压课程 videos.zip 后按名单筛选。
- 标注：直接用 `data/divide/train_lable_test.txt`（脚本自动解析并算 SROCC/PLCC）。

## 3. 运行步骤

```bash
cd CAMP-VQA/src

# 第 1 步：冒烟测试（先确认显存装得下 + 测单视频耗时）
python blip2_smoke_gpu.py --load-backbones
# 输出会给出"平均单次生成 X s"和"909 视频需 Y 小时"——把这两个数字报告给组长

# 第 2 步：全量批量打分（每视频约 1~3 分钟，909 个约 8~45 小时，建议挂机过夜）
python campvqa_eval_batch.py \
    --videos-dir <909视频所在目录> \
    --labels <train_lable_test.txt 路径> \
    --out campvqa_val_scores_909.txt
# 脚本最后自动打印: 对比标注（n=909）: SROCC=... PLCC=... OBJ=...
```

建议先用 `--max-n 50` 跑子集验证全链路无误（~1 小时），再开全量挂机。

## 4. 需要汇报的内容

1. **冒烟测试结果**：BLIP-2 单次生成耗时、全链路显存占用（8GB 是否够）、预估总时长
2. **全量结果**：`SROCC / PLCC / OBJ`（909 视频）+ 分数文件
3. **进度节奏**：冒烟测试完成、子集验证完成、全量启动、全量完成——各报一次；中途遇到报错/OOM/下载问题随时找组长
4. 把最终分数文件与指标发到组群，或直接 commit 到 video-quality-assessment 仓库（你有 write 权限，建议放 `runs/campvqa/` 目录，大数据文件走网盘）

## 5. 常见问题预案

| 问题 | 处理 |
|---|---|
| 8GB 显存 OOM | batch 脚本已做 SlowFast/Swin 用完即移回 CPU 的错峰策略；若仍 OOM，把 BLIP-2 换 `load_in_8bit=True`（需要 bitsandbytes） |
| HF 下载慢/断 | `export HF_ENDPOINT=https://hf-mirror.com` 重下；断点续传会自动进行 |
| ffprobe 失败 | 脚本有默认值兜底（1920x1080），不影响继续；少量视频元数据缺失可忽略 |
| 单视频过慢（>5 分钟） | 汇报给组长，讨论抽帧降频方案（改 `extract_clip_embeds` 采样步长）或抽样 300 视频 |

## 6. 时间规划（倒排）

- 9/5：环境 + 权重下载（~16GB，一晚）+ 冒烟测试
- 9/6：子集验证（50 个）→ 全量 909 启动，挂机
- 9/7–9/8：全量出结果 → 汇报 → 组长写进答辩 PPT/报告
- 9/11 答辩前预留缓冲

有任何问题（权重/脚本/网络/报错）随时在组群里问组长。
