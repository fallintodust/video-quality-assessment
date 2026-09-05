# 下次继续：快速开始指南（给 Claude 的会话恢复提示）

> 更新时间：2026-09-05 13:00 ｜ 用途：重新打开 Claude 后，把本文件内容作为第一条消息发给 Claude 即可快速恢复。

## 一句话背景

VQA 课程设计（第3组），答辩 2026-09-11。**(A)+(B) 主线已全部完成**：baseline（SROCC=0.6723/PLCC=0.6723）+ 半监督 v2 最终 **SROCC=0.6897/PLCC=0.6561/OBJ=1.3457**（best=轮次 2，反超 baseline，eval_val 已高精度复核）。**DOVER 复现已入库**（`dover_repro/`）：零样本 SROCC=0.7110/PLCC=0.7053（官方 0.7477/0.7546）；14 个训练集外测试视频对比：DOVER 11/11 排序全对 vs 自研 9/11（`docs/test_comparison_report.md`）。**DOVER++ 微调复现待启动**（目标 0.8071/0.8126）。**CAMP-VQA 对照实验已交接组员 Peter**。

## 当前状态快照（9/5 13:00）

- **训练全部结束**：v2 跑完轮次 1~4 + 轮次 5 B.1（池 895）；轮次 5 B.2 经组内决定跳过（轮 3/4 连续无提升）。无训练进程在跑，GPU 空闲。
- **最终交付权重**：`runs/divide_semisup_v2/model_best.pt`（=轮次 2；extractor_best/head_best 配套），逐视频分数 `val_scores_best.txt`（909 条）。指标：SROCC=0.689669 / PLCC=0.656080 / OBJ=1.345750。
- **断点续训功能已入库**（`--resume-round N [--skip-b1]`）——未来重训中断时可复用，不再需要本次补跑。
- **CAMP-VQA 交接 Peter**：`docs/campvqa_handoff.md`（环境/权重/数据/命令/汇报要求）+ `docs/campvqa/`（blip2_smoke_gpu.py、campvqa_eval_batch.py 两个适配脚本）。Peter 需回报：冒烟耗时、子集验证、全量 SROCC/PLCC、进度。收到结果后由组长写进答辩 PPT/报告。
- **分支**：`Vzixing`（训练相关修改）+ `main`（标注重建 + 模型代码合并）。**待推送**：9/5 的本地提交（GitHub 连接被重置），网络恢复后 `git push origin Vzixing main`。

## 环境

- Python：`C:\Users\ASUS\anaconda3\envs\vqa_env\python.exe`（conda 不在 bash PATH 里，直接用完整路径）
- GPU：RTX 4060 Laptop 8GB；帧缓存 `data/frames_cache/`（4543 npy，T=8）——训练/评估必须带 `--frame-cache data/frames_cache`
- CAMP-VQA 本地环境（仅作参照/备用）：`D:\campvqa`（env `D:\campvqa\env`，权重全就绪，HF_HOME=/d/hf_cache）；若 Peter 那边受阻，本机可随时自己跑

## 剩余待办（TODO.md 也有）

- [ ] **推送本地提交**：`git push origin main`（含 dover_repro LFS 权重，网络恢复后执行）
- [ ] DOVER++ 微调复现：`cd dover_repro && python training_with_divide.py -o divide_repro.yml`（10 线性 + 20 端到端，batch 8，约 4~6 小时）
- [ ] 接收 Peter 的 CAMP-VQA 结果 → README 实验表补一行 + 答辩 PPT
- [ ] 拿到官方标注后跑 `--compare-train/--compare-test` 校验重建标注；不一致则重训
- [ ] 测试视频到达 → `predict.py --fp16` 出 score.txt（注意 20 分钟耗时扣分）
- [ ] 答辩 PPT（架构图、合成数据演示素材、指标曲线）+ 每人实践报告（附 AI 对话记录，素材 docs/ai_conversation_log.md）
- [ ] 可选：消融实验（仅CNN/仅ViT/双分支）、最终模型用全量 4543 视频再训一轮
