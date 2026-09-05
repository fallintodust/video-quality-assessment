# 下次继续：快速开始指南（给 Claude 的会话恢复提示）

> 更新时间：2026-09-05 12:30 ｜ 用途：重新打开 Claude 后，把本文件内容作为第一条消息发给 Claude 即可快速恢复。

## 一句话背景

VQA 课程设计（第3组），答辩 2026-09-11。(A) baseline 完成（SROCC=0.6723/PLCC=0.6723）；**(B) 半监督 v2 已跑完 4 轮 + 第 5 轮 B.1，best=轮次 2（OBJ=1.3457，反超 baseline）；9/5 中午用户重启计算机时第 5 轮 B.2 被中断，重启后需用断点续训补跑**。

## ⚠️ 当前状态快照（9/5 12:30，重启计算机前）

- **训练进程**：v2 训练（9/3 19:51 启动，原 PID 264332）跑至第 5 轮 B.2 epoch 1（约 12:21 时 step 240/1133），因用户重启计算机被中断。**重启后先确认没有训练进程在跑**（`tasklist | findstr python`），再执行下方补跑命令。
- **已安全落盘的状态**：
  - `runs/divide_semisup_v2/pool.json`：**895 个伪标签**（轮1 868 → 轮2 880 → 轮3 890 → 轮4 893 → 轮5 B.1 +2）
  - `runs/divide_semisup_v2/model_best.pt`：**best = 轮次 2**（SROCC=0.6897 / PLCC=0.6561 / OBJ=1.3457 > baseline 1.3446）
  - `runs/divide_semisup_v2/metrics.json`：手工重建的轮次 1~4 历史（轮2 为精确值，轮1/3/4 为日志 4 位小数）
  - 轮次 3/4 未提升（早停计数 2/3）；轮次 5 B.2 尚未出结果
- **断点续训功能已实现**（`--resume-round N [--skip-b1]`，见 scripts/train_semisup.py + vqa/semisup.py）：加载 pool.json + metrics.json，从第 N 轮继续；`--skip-b1` 跳过该轮 B.1 直接 B.2。重启后须先冒烟测试再上真数据。
- **分支**：`Vzixing`（本地训练相关修改都在这）；`main` 有标注重建提交。均已推送 GitHub。**待推送**：本地提交 `e741117`（断点续训 + 9/5 进度保存，9/5 12:35 时 GitHub 连接被重置未推成功），重启后执行 `git push origin Vzixing`。

## 环境

- Python：`C:\Users\ASUS\anaconda3\envs\vqa_env\python.exe`（conda 不在 bash PATH 里，直接用完整路径）
- GPU：RTX 4060 Laptop 8GB；帧缓存 `data/frames_cache/`（4543 npy，T=8）——训练/评估必须带 `--frame-cache data/frames_cache`

## 重启后恢复步骤（按序执行）

```bash
# 0. 确认 GPU 空闲、无残留训练进程（重启后应干净）
tasklist | findstr python
nvidia-smi

# 1.（可选但建议）续训冒烟测试：合成数据 + 假 pool/metrics，验证 --resume-round/--skip-b1 路径
#    用极小参数 --n-runs 1 --sub-ratio 0.1 --sub-epochs 1 --val-epochs 1 跑 1 轮即可

# 2. 补跑第 5 轮 B.2（约 2.5 小时：10 epochs × 1133 步 @80步/分 + 验证集评测）
/c/Users/ASUS/anaconda3/envs/vqa_env/python.exe -u scripts/train_semisup.py \
    --data-dir data/divide/videos \
    --labels data/divide/train_lable_train.txt \
    --val-labels data/divide/train_lable_test.txt \
    --baseline runs/divide_baseline --out runs/divide_semisup_v2 \
    --var-threshold 0.05 --pseudo-weight 0.2 --pseudo-debias --pseudo-clip \
    --fp16 --frame-cache data/frames_cache \
    --resume-round 5 --skip-b1 >> runs/divide_semisup_v2_console.log 2>&1

# 3. 高精度复核最终 best 权重（909 锁定验证集）
/c/Users/ASUS/anaconda3/envs/vqa_env/python.exe -u scripts/eval_val.py \
    --model-dir runs/divide_semisup_v2 --data-dir data/divide/videos \
    --labels data/divide/train_lable_test.txt --frame-cache data/frames_cache --fp16 \
    --out runs/divide_semisup_v2/val_scores_best.txt

# 4. CAMP-VQA 对比实验（见下节）→ README 实验表 + docs 更新 → push Vzixing（权重走 LFS）
```

要点：
- v2 参数固定为 `--var-threshold 0.05 --pseudo-weight 0.2 --pseudo-debias --pseudo-clip`，续跑命令必须一致
- 若时间紧张，可跳过第 2 步直接进入第 3 步（best 已是轮次 2 权重，第 5 轮只是完整性的补充实验）
- 补跑日志追加到同一 console.log；补跑若 OBJ 超 1.3457 会覆盖 model_best.pt（正常行为）

## CAMP-VQA 复现（WACV 2026 SOTA 对照实验）

- 仓库：`D:\campvqa`；conda 环境 `D:\campvqa\env`（python 3.10 + torch 2.6 cu124）
- 缓存全在 D：`PIP_CACHE_DIR=/d/pip_cache`、`HF_HOME=/d/hf_cache`（BLIP-2 flan-t5-xl 13GB ✓、Swin-Large ✓）、CAMP-VQA checkpoint 在 `D:\campvqa\model`（277M ✓）
- 13056 维特征拼接与官方权重精确对齐已验证（CPU）；**待办**：GPU 上 BLIP-2 smoke test → 909 验证视频批量打分 → SROCC/PLCC 与任务书模型对比写入报告
- 入口：`python src/camp-vqa_demo.py`（in-repo 示例视频 `test_videos/0_16_07_...mp4`）

## 后续待办（TODO.md 也有）

- [ ] 补跑第 5 轮 B.2 → eval_val 复核 → README 实验记录表
- [ ] CAMP-VQA 对照实验（909 验证集 SROCC/PLCC）
- [ ] 拿到官方标注后跑 `--compare-train/--compare-test` 校验重建标注；不一致则重训
- [ ] 测试视频到达 → `predict.py --fp16` 出 score.txt（注意 20 分钟耗时扣分）
- [ ] 答辩 PPT（架构图、合成数据演示素材、指标曲线）+ 每人实践报告（附 AI 对话记录，素材 docs/ai_conversation_log.md）
- [ ] 可选：消融实验（仅CNN/仅ViT/双分支）、最终模型用全量 4543 视频再训一轮
