# TODO（项目待办清单）

> 更新时间：2026-09-02 ｜ 答辩：2026-09-11 ｜ 提交物：系统源代码 + 答辩PPT（含演示视频/图片）+ 每人一份实践报告

## 🔴 阻塞项（等课程提供）

- [x] **拿到课程标注文件**：`train_lable_train.txt`（训练集标注）、`train_lable_test.txt`（验证集标注）
  - 已从原版 MaxWell 仓库（VQAssessment/ExplainableVQA）重建出等效标注并提交到 `data/divide/`（依据：examplar 分值与 O 列 Pearson=1.0，train/test 划分覆盖全部 4543 视频），见 README「标注重建」
  - 官方标注下发后运行 `scripts/build_course_labels.py --compare-train ... --compare-test ...` 校验；不一致则用官方文件重训
- [ ] **确认测试集视频**：答辩前拿到测试视频（放 `data/test_videos/`，`predict.py` 直接出 `score.txt`）

## 🟡 正式训练与调参（重建标注已就绪）

- [x] 正式数据跑 baseline（进行中，分支 Vzixing）：
      `python scripts/train_baseline.py --data-dir data/divide/videos --labels data/divide/train_lable_train.txt --val-labels data/divide/train_lable_test.txt --out runs/divide_baseline --fp16 --epochs 12`
- [ ] 正式数据跑半监督循环（`train_lable_test.txt` 作为锁定验证集；909 个验证视频作为"未标注"进入伪标签池）
- [ ] 调参实验（记录到 README 结果表）：
  - [ ] 伪标签方差阈值 `--var-threshold`（默认 25 是 0~100 量纲；重建标注为 1~5，需改为 ~0.2）
  - [ ] 伪标签权重 `--pseudo-weight`（默认 0.5）
  - [ ] 每轮独立训练次数 `--n-runs`、抽样比例 `--sub-ratio`
  - [ ] 抽帧数 `--t`（8 ↔ 16）与 batch size
- [ ] 生成最终 `score.txt`，核对 SROCC/PLCC/总分（注意 20 分钟耗时扣分规则）

## 🟢 提交物（9 月 11 日答辩前）

- [ ] **答辩 PPT**：含演示视频/图片（建议：合成闪烁样例对比、打分界面/输出示例、指标曲线）
- [ ] **实践报告（每人 1 份）**：
  - [ ] 附上 AI 辅助的对话记录（课程要求）
  - [ ] 说明模型原理、输入输出、数据处理、评价指标、运行环境、复现结果
- [ ] 代码最终整理：README 结果表更新、`score.txt` 样例、清理临时文件

## ⚪ 可选加分项

- [ ] RL 扩展（PPO 打分/增强策略）—— 设计文档见 `docs/rl_extension.md`，可实现 `vqa/rl/` 模块
- [ ] fp16 推理加速（约 2×，进一步压缩耗时）
- [ ] KoNViD-1k 预训练增强泛化（`scripts/download_konvid.md`）
- [ ] 多分支对比实验（仅 CNN / 仅 ViT / 双分支）写进报告

## 🤝 协作规范（多人）

- 各自 fork 或建 feature 分支，改动经 Pull Request 合并
- 大数据（视频/权重）不入库，走线下分发或网盘；标注 txt 可直接入库
- 实验结果统一记录在 README「实验记录」表格
