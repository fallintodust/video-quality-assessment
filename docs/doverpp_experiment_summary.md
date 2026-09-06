# DOVER / DOVER++ 复现实验总结

> 更新时间：2026-09-06 19:20 ｜ 实验机器：RTX 4060 Laptop 8GB ｜ 数据集：DIVIDE-MaxWell（train 3634 / val 909）

## 一、总体结论

| 部分 | 状态 | 结果 |
|---|---|---|
| DOVER 零样本复现 | ✅ 完成 | SROCC=0.7110 / PLCC=0.7053；14 视频测试集 11/11 排序全对 |
| DOVER++ 线性微调（仅头部） | ✅ 完成 | **SROCC=0.7854 / PLCC=0.7905**（s 分支，epoch 3 best） |
| DOVER++ 端到端微调（全参数） | ⏳ 未完成 | e2e epoch 0 两次中断，`resume.pth` 可续跑 |

线性微调 4 个 epoch 即逼近官方全流程数字（0.8071/0.8126），端到端阶段完成后期望追平或超过官方报告。

## 二、DOVER 零样本（已完成）

- 权重：`dover_repro/pretrained_weights/DOVER.pth`（LFS 入库）
- 909 验证集全量：**SROCC=0.7110 / PLCC=0.7053**（官方报告 0.7477/0.7546）
- 14 个训练集外测试视频：**11/11 排序全对**（自研模型 9/11），详见 `docs/test_comparison_report.md`

## 三、DOVER++ 训练时间线（2026-09-06 共 8 次启动）

| 次 | 启动 | 死亡/停止 | 配置 | 结果 |
|---|---|---|---|---|
| 1 | ~1:53 | ~2:12 | batch 8 | 崩溃（原因未记录） |
| 2 | 2:12 | 2:40 | batch 8, workers 8 | epoch 0 训练完成，验证 51% 时 DataLoader worker 崩溃 |
| 3 | 12:16 | 12:43 | batch 8, workers 6 | epoch 0 训练 90%（410/455），**GPU 硬错误**（VALORANT 挤爆显存） |
| 4 | 12:52 | 12:57 | batch 8 | epoch 0 训练 5%，**GPU 硬错误**（同上） |
| 5 | 14:38 | ~14:42 | batch 4 | epoch 0 训练 17%，无 Traceback 死亡 |
| 6 | 14:55 | 14:56 | batch 2 | epoch 0 训练 2%，无 Traceback 死亡 |
| 7 | 15:26 | 18:15 | batch 4, workers 2 | ✅ 线性 4 epoch 全部完成 → e2e epoch 0 跑到 57/909 时 **GPU 硬错误** |
| 8 | 18:16 | 19:12（主动停止） | batch 4, 纯 e2e 7 epochs | 从 `resume.pth` 续跑至 e2e epoch 0 108/909（12%），组长主动停止 |

## 四、线性微调结果（已完成 ✅）

验证集 909 视频，逐轮指标：

| Epoch | SROCC (n 分支) | SROCC (s 分支) | PLCC (s 分支) |
|---|---|---|---|
| 0 | 0.7401 | — | — |
| 1 | 0.7639 | — | — |
| 2 | 0.7785 | 0.7680 | 0.7751 |
| 3 | 0.7664（回落） | **0.7854** | **0.7905** |

epoch 3 s 分支完整指标：SROCC=0.7854 / PLCC=0.7905 / KROCC=0.5915 / RMSE=0.3716

### 参照系对比

| 模型 | SROCC | PLCC |
|---|---|---|
| DOVER 零样本 | 0.7110 | 0.7053 |
| **DOVER++ 线性微调（4 epoch）** | **0.7854** | **0.7905** |
| 官方 DOVER++ 报告（全流程） | 0.8071 | 0.8126 |
| 自研模型（半监督 v2 最优） | 0.6897 | 0.6561 |

## 五、端到端阶段（未完成 ⏳）

- 第 7 次进入 e2e 后（57/909）GPU 硬错误中断；第 8 次从 `resume.pth` 续跑至 108/909 后主动停止。
- **续跑方法**：
  ```
  cd dover_repro
  python training_with_divide.py -o divide_repro.yml --train train-dividemaxwell --val val-dividemaxwell
  ```
  当前 `divide_repro.yml` 已配好：`test_load_path: ./pretrained_weights/resume.pth`、`l_num_epochs: 0`（跳过线性）、`num_epochs: 7`、batch 4、workers 2。
- **速度与时长**：e2e batch 4 ≈ 30s/it × 909 batch ≈ 7.6h/epoch。**7 epochs ≈ 53h，9/11 答辩前来不及，建议 `num_epochs` 砍到 2~3**（每 epoch 结束自动验证+保存，随时可取 best）。
- 线性阶段已收敛的头部权重保留在 `resume.pth` 中，e2e 只需微调全参数，2~3 个 epoch 预期即可超越线性结果。

## 六、GPU 硬错误记录（重要环境风险）

系统事件日志 `nvlddmkm` **Event 153 — "Error occurred on GPUID: 100"** 全天出现 4 次：

| 时间 | 触发场景 |
|---|---|
| 12:45 / 12:58 | VALORANT 与训练抢 8GB 显存，游戏加载内容时挤爆 → CUDA 上下文崩溃（进程无 Traceback 直接死） |
| 18:16 | 无游戏，纯训练长时负载（15:26 起连续 ~3h，线性阶段正常 → e2e 阶段全参数计算量骤增） |

**结论与规避建议**：
1. 8GB 显卡容不下「游戏 + 训练」或「demo UI 服务 + 训练」共存——训练期间关闭 VALORANT、demo 后端（uvicorn）、火绒录屏等 GPU 占用进程；
2. e2e 阶段全参数训练显存/功耗激增，长时跑注意散热（此前峰值 82°C），可考虑降低负载或缩短单段训练时长；
3. 死亡前脚本已实现「失败视频跳过保护」「每轮落盘」「resume.pth 保存」，再遇 GPU 错误可用 `resume.pth` 直接续跑。

## 七、剩余任务

- [ ] **DOVER++ e2e 续跑**（`resume.pth` 就绪，建议 num_epochs 2~3，预计 15~23h，9/8 前启动）
- [ ] e2e 完成后：结果写入 README 实验记录表 + 测试集对比表补 DOVER++ 列
- [ ] 噪点（组员1）、模糊（组员3）正式实现**至今未提交**（截至 9/6 19:00 无分支/PR，仅有组长参考实现）——需催
- [ ] CAMP-VQA 结果等 Peter 回报后补 README
- [ ] 官方标注校验、测试视频 score.txt、答辩 PPT、实践报告

## 八、产物清单

| 文件 | 说明 |
|---|---|
| `dover_repro/pretrained_weights/DOVER.pth` | DOVER 官方权重（LFS） |
| `dover_repro/pretrained_weights/DOVER_head_..._s_finetuned.pth` | 线性微调头部权重（s/n 分支，0.4MB×2，LFS） |
| `dover_repro/pretrained_weights/DOVER_head_..._s_latest.pth` | 线性 3 epoch 完整模型（s 分支 best：0.7854/0.7905，LFS） |
| `dover_repro/pretrained_weights/resume.pth` | e2e 续跑起点（含线性成果 + e2e 部分进度，LFS） |
| `dover_repro/zero_shot_predictions.csv` | 零样本逐视频预测（909 条） |
| `dover_repro/test_comparison/` | 14 视频测试集对比打分表 |
| `dover_repro/train_doverpp{1..8}.log` | 训练日志（仅存于 D:\dover_repro 工作目录，未入库） |

> 注：n 分支完整模型（`n_latest.pth`，SROCC=0.7809）被 s 分支严格占优，未上传 LFS 以节省配额，需要时从 D:\dover_repro 本地取。
