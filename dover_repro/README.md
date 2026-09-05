# DOVER 复现（DIVIDE-MaxWell，第3组 VQA 课程设计）

官方仓库 [VQAssessment/DOVER](https://github.com/VQAssessment/DOVER)（ICCV 2023）的 Windows/torch2.5 复现，
包含零样本验证、测试集打分与 DOVER++ 微调训练脚本。

## 零样本复现结果（DIVIDE-MaxWell 官方 909 验证集）

| | SROCC | PLCC | KROCC |
|---|---|---|---|
| 本复现（DOVER.pth 零样本） | 0.7110 | 0.7053 | 0.5218 |
| 官方报告 | 0.7477 | 0.7546 | 0.5510 |

- 预测：`zero_shot_predictions.csv`（909 行，含官方 overall 与课程重建标注对照）
- 已确认：课程重建标注 = 官方 overall 的精确仿射变换（user = 0.929·official + 0.231，Pearson=0.9999998）

## 相对官方仓库的改动（Windows 适配，均不改模型/损失/优化器）

1. `dover/datasets/basic_datasets.py`：decord → `Cv2VideoReader`（cv2 实现，含尾帧校验
   修正个别视频多报 1 帧的问题）
2. `dover/datasets/dover_datasets.py`：解码换 cv2；补上官方 master 漏掉的
   `gt_label_a/gt_label_t` 解包（DOVER++ 三列监督必需）
3. `training_with_divide.py`：移除 thop 导入（仅被注释代码使用）；wandb 用根目录
   `wandb.py` 桩替换（离线 no-op）
4. `divide_repro.yml`：本地数据路径 + `split_seed: -1`（单次训练）+ batch 8

## 用法

```bash
# 零样本打分（输出 CSV + SROCC/PLCC）
python zero_shot_eval.py --fp16

# 任意目录视频打分（课程任务书格式 score.txt，0~100）
python score_test.py --videos <目录> --out score.txt --fp16

# DOVER++ 微调复现（目标 0.8071/0.8126）
python training_with_divide.py -o divide_repro.yml \
    --train train-dividemaxwell --val val-dividemaxwell
```

权重：`pretrained_weights/DOVER.pth`（HF teowu/DOVER，Git LFS 入库）。
官方 3 列标注：`examplar_data_labels/DIVIDE_MaxWell/`（overall/aesthetic/technical）。
