# 下次继续：快速开始指南（给 Claude 的会话恢复提示）

> 更新时间：2026-09-02 18:50 ｜ 用途：重新打开 Claude 后，把本文件内容作为第一条消息发给 Claude 即可快速恢复。

## 一句话背景

VQA 课程设计（第3组），答辩 2026-09-11。任务书架构已实现并自测通过；**课程标注已从原版 MaxWell 仓库重建**并入库；**(A) baseline 已跑完**（SROCC=0.6723 / PLCC=0.6723，best@epoch4，验证集 909）；**(B) 半监督进行中**（9/2 15:10 启动，轮次 1/5 的 B.2 微调阶段；实测 ~0.4-0.9 步/秒，ETA 9/4 下午-晚间）。

## 当前状态快照

- **分支**：`Vzixing`（本地训练相关修改都在这）；`main` 有标注重建提交（0dbdadd）。均已推送 GitHub。
- **标注**（已入库 data/divide/，见 Issue #1）：
  - `train_lable_train.txt` 3634 条 + `train_lable_test.txt` 909 条，整体 MOS（1~5 量纲），覆盖全部 4543 视频
  - 附 `*_Tall.txt` 时域轴变体（官方标注到达后对比课程 MOS 基于哪一轴）
  - 校验脚本：`scripts/build_course_labels.py --compare-train 官方 --compare-test 官方`
- **视频**：`data/divide/videos/`（4543 个，320p~4K 分辨率不一）；F: 盘有 videos.zip 的 5 个 2GB 分卷（FAT32 放不下单文件）
- **帧缓存**：`data/frames_cache/` 已完整生成（4543 个 npy，T=8，约 5.5GB）——训练必须带 `--frame-cache data/frames_cache`
- **协作者**：kun0-0 / fbw1128 / stoneofTa（均 write 权限，已接受）
- **待课程提供**：官方 `train_lable_train.txt` / `train_lable_test.txt`（用于对比校验）、答辩测试视频（放 `data/test_videos/` 后 predict.py 出 score.txt）

## 环境

- Python：`C:\Users\ASUS\anaconda3\envs\vqa_env\python.exe`（conda 不在 bash PATH 里，直接用完整路径）
- GPU：RTX 4060 Laptop 8GB（fp16 训练显存约 6.6GB，bs 默认 4）

## 恢复步骤（按序执行）

```bash
# 0. 检查半监督是否还在跑（9/2 15:10 启动，约 35-50 小时）
tail -5 runs/divide_semisup_console.log

# 1. 若(B)被中断且未完成：重启半监督（baseline 权重已在 runs/divide_baseline）
/c/Users/ASUS/anaconda3/envs/vqa_env/python.exe -u scripts/train_semisup.py \
    --data-dir data/divide/videos \
    --labels data/divide/train_lable_train.txt \
    --val-labels data/divide/train_lable_test.txt \
    --baseline runs/divide_baseline --out runs/divide_semisup \
    --var-threshold 0.2 --fp16 --frame-cache data/frames_cache

# 2. 复核任意 checkpoint 在 909 验证集上的高精度指标（scripts/eval_val.py，本次新增）
/c/Users/ASUS/anaconda3/envs/vqa_env/python.exe -u scripts/eval_val.py \
    --model-dir runs/divide_semisup --data-dir data/divide/videos \
    --labels data/divide/train_lable_test.txt --frame-cache data/frames_cache --fp16 \
    --out runs/divide_semisup/val_scores_best.txt

# 3. 结果记录到 README「实验记录」表，向用户阐释 SROCC/PLCC
# 4. 上传 runs 小文件 + 最终 best 权重(LFS) 到 Vzixing 分支（用户已授权直接执行）
# 5. 写训练总结 + 模型能力边界/研究方向分析（docs/summary_report.md）
```

要点：
- 用 `-u` 关闭输出缓冲（上次裸跑 stdout 缓冲导致日志看不到进度）
- `--var-threshold 0.2` 是 1~5 量纲的阈值（0~100 量纲才是默认 25）
- 若时间不够，可先 `--epochs 6` 出初步结果，答辩前再补满

## 后续待办（TODO.md 也有）

- [ ] baseline / 半监督出结果 → 更新 README 实验记录表
- [ ] 拿到官方标注后跑 `--compare-train/--compare-test` 校验重建标注；不一致则重训
- [ ] 测试视频到达 → `predict.py --fp16` 出 score.txt（注意 20 分钟耗时扣分）
- [ ] 答辩 PPT（架构图、合成数据演示素材、指标曲线）+ 每人实践报告（附 AI 对话记录）
- [ ] 可选：消融实验（仅CNN/仅ViT/双分支）、最终模型用全量 4543 视频再训一轮
