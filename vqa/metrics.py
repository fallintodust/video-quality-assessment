"""评价指标（任务书第七章）。

- SROCC：排序一致性（Spearman 秩相关系数）
- PLCC ：线性相关性（Pearson 相关系数）
- 总分：Score = (SROCC + PLCC)/2 - min(1.0, 0.01*max(0, T-20))，T 为 100 个视频的测试耗时（分钟）
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def srocc(y_true, y_pred):
    """Spearman 秩相关系数，[-1, 1]，越接近 1 排序越一致。"""
    if len(y_true) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def plcc(y_true, y_pred):
    """Pearson 线性相关系数，[-1, 1]，越接近 1 线性相关性越强。"""
    if len(y_true) < 2:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def total_score(srocc_val, plcc_val, minutes):
    """任务总分：平均相关分减去超时惩罚。"""
    penalty = min(1.0, 0.01 * max(0.0, minutes - 20.0))
    return (srocc_val + plcc_val) / 2.0 - penalty


def evaluate_metrics(y_true, y_pred):
    """一次算齐 SROCC/PLCC/OBJ。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    s = srocc(y_true, y_pred)
    p = plcc(y_true, y_pred)
    return {"SROCC": s, "PLCC": p, "OBJ": s + p}
