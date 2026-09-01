"""全局配置：超参数集中于此，各脚本通过命令行覆盖常用项。

任务书规定的固定量（特征维度等）直接写死在 models.py 中，
此处只放可调超参数。
"""


class Config:
    # ---- 数据 ----
    T = 8                    # 每个视频抽取帧数（任务书示例 T=8）
    SIZE = 224               # 帧统一空间分辨率 224x224
    MEAN = [0.485, 0.456, 0.406]   # ImageNet 归一化均值
    STD = [0.229, 0.224, 0.225]    # ImageNet 归一化方差

    # ---- 训练（baseline）----
    EPOCHS = 20              # 总 epoch 数
    BATCH_SIZE = 4           # 视频数 / batch（每 batch 含 B*T 帧）
    LR = 1e-4                # Adam 学习率（任务书建议 1e-4 ~ 5e-5）
    WEIGHT_DECAY = 1e-4
    VAL_RATIO = 0.15         # 标注集中留作验证的比例
    SEED = 42

    # ---- 半监督（B 阶段）----
    N_RUNS = 5               # 每轮独立训练-打分次数
    SUB_RATIO = 0.7          # 每次独立训练从当前训练集随机抽样的比例
    SUB_EPOCHS = 3           # 每次独立训练的 epoch 数
    PSEUDO_ROUNDS = 5        # 伪标签生成轮数
    VAR_THRESHOLD = 25.0     # 稳定性筛选阈值：N 次预测方差小于该值才生成伪标签（MOS 量纲）
    PSEUDO_WEIGHT = 0.5      # 伪标签样本的损失权重（真实标签 w=1.0）
    VAL_EPOCHS = 10          # 验证/微调阶段每轮 epoch 数
    EARLY_STOP = 3           # 连续多少轮 OBJ 无提升则早停
    HIDE_RATIO = 0.4         # 半监督模式下，隐藏多少比例的标注样本当作"未标注"（模拟场景）
