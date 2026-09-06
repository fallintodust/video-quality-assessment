# TODO（项目待办清单）

> 更新时间：2026-09-04 ｜ 答辩：2026-09-11 ｜ 提交物：系统源代码 + 答辩PPT（含演示视频/图片）+ 每人一份实践报告

## 👥 分工（第3组 VQA）

| 模块 | 负责人 | 状态 |
|---|---|---|
| 整体打分模型（A+B 流程训练、score.txt） | 组长（Vzixing） | baseline 完成 SROCC/PLCC=0.6723；半监督 v2 复测中 |
| 噪点判定 | 组员（认领） | 接口就绪，待实现 |
| 闪烁判定 | 组员（认领） | 接口就绪（附参考实现 `heuristic_flicker`） |
| 模糊判定 | 组员（认领） | 接口就绪，待实现 |

**统一诊断接口**：`vqa/diagnosis.py` + `scripts/diagnose.py`
- 组员接入方式：实现 `detect(frames_rgb) -> {"score": 0~1, "level": 无/轻/中/重, "detail": {...}}`，赋给对应槽位（`NOISE_DETECTOR` / `FLICKER_DETECTOR` / `BLUR_DETECTOR`）并 `register_detector`（详见文件头注释）
- 整合入口：`python scripts/diagnose.py --videos <目录> [--model 权重]` → 输出 `report.json` / `report.txt`（总分 + 各失真问题清单，与 `score.txt` 配套交付）

## 🔴 阻塞项（等课程提供）

- [ ] **官方标注文件**：`train_lable_train.txt` / `train_lable_test.txt`
  - 已从原版 MaxWell 仓库（VQAssessment/ExplainableVQA）重建等效标注（examplar 分值与 CSV O 列 Pearson=1.0，覆盖全部 4543 视频）
  - 官方下发后运行 `scripts/build_course_labels.py --compare-train ... --compare-test ...` 校验；不一致则用官方文件重训
- [ ] **答辩测试视频**：放 `data/test_videos/` 后 `predict.py` 出 `score.txt`（注意 20 分钟耗时扣分）
  - 已用 14 个训练集外视频自测：自研模型 + DOVER 各出一份打分表（`dover_repro/test_comparison/`）

## 🟡 打分训练（主线，A+B 任务书流程）

- [x] (A) baseline：**SROCC=0.6723 / PLCC=0.6723 / OBJ=1.3446**（best@epoch4，909 锁定验证集）
- [x] (B) 半监督 v1（默认配置 var 0.2 / weight 0.5）：全部 4 轮低于 baseline（早停 3/3），结论：伪标签压缩偏差 + 回喂权重过大，详见 `docs/semisup_experiment_report.md`
- [x] (B) 半监督 v2（重标定 + weight 0.2 + var 0.05 + 裁剪）：**轮次 2 best SROCC=0.6897 / PLCC=0.6561 / OBJ=1.3457 反超 baseline**；轮次 5 B.2 组内决定跳过（轮 3/4 连续无提升）
- [x] eval_val 高精度复核最终权重（SROCC=0.689669 / PLCC=0.656080 / OBJ=1.345750）
- [x] DOVER 复现（`dover_repro/`）：零样本 SROCC=0.7110 / PLCC=0.7053；测试集对比 11/11 排序全对（详见 `docs/test_comparison_report.md`）
- [ ] DOVER++ 微调复现（目标 0.8071/0.8126）→ 训练启动后监控
- [ ] CAMP-VQA 零样本对照实验：**已交接 Peter**（docs/campvqa_handoff.md），待其回报结果后补 README 实验表
- [ ] 全部完成后取 OBJ 最优权重 → `predict.py` 出最终 `score.txt`

## 🟢 问题判定（三条支线）

- [ ] 噪点判定实现（组员1）
- [ ] 闪烁判定实现（组员2，替换参考实现 `heuristic_flicker`）
- [ ] 模糊判定实现（组员3）
- [ ] `diagnose.py` 集成联调，报告格式对齐
- 合成数据：闪烁生成器已有（`scripts/make_flicker_dataset.py`，自测 SROCC≈0.92）；噪点/模糊合成生成器由对应组员补充（用于自测与消融）

## 🟢 提交物（9 月 11 日答辩前）

- [ ] **答辩 PPT**：含演示视频/图片（合成失真样例对比、诊断报告示例、指标曲线）
- [ ] **实践报告（每人 1 份）**：附 AI 对话记录；说明模型原理、输入输出、数据处理、评价指标、运行环境、复现结果
- [ ] 代码最终整理：README 结果表更新、`score.txt`/诊断报告样例、清理临时文件

## ⚪ 可选加分项

- [ ] RL 扩展（PPO 打分/增强策略）—— 设计文档见 `docs/rl_extension.md`
- [ ] 注意力可视化（模型看视频时关注哪些帧/区域，答辩演示素材）
- [ ] 多分支对比实验（仅 CNN / 仅 ViT / 双分支）写进报告

## 🤝 协作规范（多人）

- 各自建 feature 分支，改动经 Pull Request 合并 main
- 大数据（视频）不入库，走线下分发或网盘；权重经 LFS 入库；标注 txt、代码、结果 json/txt 可直接入库
- 实验结果统一记录在 README「实验记录」表格
