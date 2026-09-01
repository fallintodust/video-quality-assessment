"""双分支视觉特征提取器 + 回归头（任务书第五章架构，维度逐项对齐）。

单帧流程：
    卷积分支  ResNet-50[1] layer1~4 输出 -> 空间维 mean/max/std 统计池化 + layer4 GAP
              = 3*(256+512+1024+2048) + 2048 = 13568 维
    ViT 分支  ViT-B/16[2] encoder 末层 token -> token 维 mean/max/std 统计池化
              = 3*768 = 2304 维
    拼接      -> 单帧向量 15872 维
时间聚合：T 帧向量在时间维取平均 -> 视频级特征（维度不变，15872）
回归头：  Linear(15872->512) + ReLU + Linear(512->1) -> 标量质量分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .config import Config

# ---- 特征维度（与任务书计算一致）----
CNN_FEAT_DIM = 13568
VIT_FEAT_DIM = 2304
FRAME_FEAT_DIM = CNN_FEAT_DIM + VIT_FEAT_DIM   # 15872
HIDDEN_DIM = 512


def spatial_stat_pool(x):
    """空间维 mean/max/std 池化并在通道维拼接：[B,C,H,W] -> [B,3C]"""
    m = x.mean(dim=(2, 3))
    mx = x.amax(dim=(2, 3))
    s = x.std(dim=(2, 3))
    return torch.cat([m, mx, s], dim=1)


def token_stat_pool(x):
    """token 维 mean/max/std 池化并拼接：[B,N,D] -> [B,3D]"""
    return torch.cat([x.mean(dim=1), x.amax(dim=1), x.std(dim=1)], dim=1)


class CNNBranch(nn.Module):
    """ResNet-50 多尺度卷积分支 -> [B, 13568]"""

    LAYERS = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self):
        super().__init__()
        # ImageNet 预训练权重（任务书要求）
        self.backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.acts = {}
        for name in self.LAYERS:
            getattr(self.backbone, name).register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(module, inp, out):
            self.acts[name] = out
        return fn

    def forward(self, x):
        self.acts.clear()
        _ = self.backbone(x)  # 完整前向；各 stage 输出由钩子收集
        parts = [spatial_stat_pool(self.acts[n]) for n in self.LAYERS]
        # 最终层 GAP：layer4 输出 adaptive_avg_pool 到 1x1 后展平
        gap = torch.flatten(F.adaptive_avg_pool2d(self.acts["layer4"], 1), 1)
        parts.append(gap)
        return torch.cat(parts, dim=1)  # [B, 13568]


class ViTBranch(nn.Module):
    """ViT-B/16 全局 token 分支 -> [B, 2304]"""

    def __init__(self):
        super().__init__()
        self.vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        # patch tokens：[B, 196, 768]（torchvision 0.20 的 _process_input 不含 class token）
        tokens = self.vit._process_input(x)
        # 补上 class token 再进 encoder：末层隐状态 [B, 197, 768]
        cls = self.vit.class_token.expand(tokens.shape[0], -1, -1)
        tokens = self.vit.encoder(torch.cat([cls, tokens], dim=1))
        return token_stat_pool(tokens)  # [B, 2304]


class FeatureExtractor(nn.Module):
    """双分支并行提取，拼接为 15872 维单帧向量。"""

    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.vit = ViTBranch()

    def forward(self, x):
        return torch.cat([self.cnn(x), self.vit(x)], dim=1)  # [B, 15872]


class RegressionHead(nn.Module):
    """回归头：15872 -> 512 -> ReLU -> 1"""

    def __init__(self, in_dim=FRAME_FEAT_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


class VQAModel(nn.Module):
    """完整模型：帧 -> 视频。

    输入 [B, T, 3, H, W]，T 帧各自提取 15872 维特征，时间维平均后回归 MOS。
    """

    def __init__(self, T=None):
        super().__init__()
        self.T = T if T is not None else Config.T
        self.extractor = FeatureExtractor()
        self.head = RegressionHead()

    def forward(self, x):
        B, T = x.shape[:2]
        feat = self.extractor(x.reshape(B * T, 3, *x.shape[-2:]))  # [B*T, 15872]
        video_feat = feat.reshape(B, T, -1).mean(dim=1)             # [B, 15872] 时间聚合
        return self.head(video_feat)                                # [B]
