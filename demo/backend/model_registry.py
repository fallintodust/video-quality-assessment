# demo/backend/model_registry.py
"""多模型注册表：统一 predict(video_path) -> float 接口，懒加载 + 缓存。

每个模型附带：id / 名称 / 特点描述（供前端选择卡片展示）/ 量纲说明。
新增模型只需在 MODELS 列表加一条。
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dover_repro"))

import numpy as np
import torch


def _get_device():
    prefer = os.environ.get("DEMO_DEVICE", "cuda")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _BasePredictor:
    """预测器基类：predict(video_path) -> float"""

    def predict(self, video_path):
        raise NotImplementedError


# ---------------------------------------------------------------- 自研模型
class OursPredictor(_BasePredictor):
    """任务书架构：ResNet-50 多尺度 + ViT-B/16 双分支，15872 维 + 回归头。"""

    def __init__(self, ckpt):
        from vqa.train_utils import build_model, score_videos
        from vqa.dataset import VideoDataset

        self.device = _get_device()
        self.model = build_model(self.device, T=8)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.extractor.load_state_dict(state["extractor"])
        self.model.head.load_state_dict(state["head"])
        self.model.eval()
        self._score_videos = score_videos
        self._VideoDataset = VideoDataset

    def predict(self, video_path):
        name = os.path.basename(video_path)
        ds = self._VideoDataset.unlabeled(os.path.dirname(video_path), [name], T=8)
        s = self._score_videos(self.model, ds, self.device, batch_size=4,
                               use_fp16=True)
        return float(s[name])


# ---------------------------------------------------------------- DOVER 系
class _DoverBase(_BasePredictor):
    """DOVER / DOVER++ 共用：双分支（Video Swin GRPB 技术 + ConvNeXt 美学）。"""

    def __init__(self, ckpt, fuse=False, scale100=False):
        import yaml
        from dover.models import DOVER
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

        self.device = _get_device()
        self.fuse = fuse
        self.scale100 = scale100
        self.model = DOVER(**{
            "backbone": {
                "technical": {"type": "swin_tiny_grpb", "checkpoint": True},
                "aesthetic": {"type": "conv_tiny"},
            },
            "backbone_preserve_keys": "technical,aesthetic",
            "divide_head": True,
            "vqa_head": {"in_channels": 768, "hidden_channels": 64},
        }).to(self.device)
        loaded = torch.load(ckpt, map_location="cpu")
        sd = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
        self.model.load_state_dict(sd)
        self.model.eval()
        # 采样器（与官方验证配置一致）
        self.tech = UnifiedFrameSampler(32, 1, 2, 3)      # 3 clips x 32 帧
        self.aest = UnifiedFrameSampler(32 // 32, 32, 2, 1)
        self._decompose = spatial_temporal_view_decomposition
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

    def predict(self, video_path):
        samplers = {"technical": self.tech, "aesthetic": self.aest}
        sample_types = {
            "technical": {"fragments_h": 7, "fragments_w": 7, "fsize_h": 32,
                          "fsize_w": 32, "aligned": 32, "clip_len": 32,
                          "frame_interval": 2, "num_clips": 3},
            "aesthetic": {"size_h": 224, "size_w": 224, "clip_len": 32,
                          "frame_interval": 2, "t_frag": 32, "num_clips": 1},
        }
        video, frame_inds = self._decompose(video_path, sample_types, samplers,
                                            False, False)
        for k, v in video.items():
            v = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
            b, c, t, h, w = v.shape
            nc = sample_types[k]["num_clips"]
            video[k] = (v.reshape(b, c, nc, t // nc, h, w)
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(b * nc, c, t // nc, h, w)).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            scores = self.model(video, reduce_scores=False)
            scores = [np.mean(x.float().cpu().numpy()) for x in scores]
        if self.fuse:
            # 官方融合：technical/aesthetic 标定后加权（0~1）x100
            t, a = ((scores[1] - 0.1107) / 0.07355,
                    (scores[0] + 0.08285) / 0.03774)
            x = t * 0.6104 + a * 0.3896
            s = 1 / (1 + np.exp(-x))
            return float(s * 100.0) if self.scale100 else float(s)
        # 微调模型：两分支分数之和（训练/验证同口径）
        return float(sum(scores))


class DoverPredictor(_DoverBase):
    def __init__(self, ckpt):
        super().__init__(ckpt, fuse=True, scale100=True)


class DoverPlusPlusPredictor(_DoverBase):
    def __init__(self, ckpt):
        super().__init__(ckpt, fuse=False, scale100=False)


# ---------------------------------------------------------------- 注册表
def _find_doverpp_ckpt():
    """自动找 DOVER++ 微调产物（best > latest > 不存在）。"""
    cands = [
        PROJECT_ROOT / "dover_repro" / "pretrained_weights" / "divide_val-dividemaxwell_s_finetuned.pth",
        PROJECT_ROOT / "dover_repro" / "pretrained_weights" / "divide_val-dividemaxwell_s_latest.pth",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


MODELS = [
    {
        "id": "ours",
        "name": "自研模型（任务书架构）",
        "desc": ("ResNet-50 多尺度卷积 + ViT-B/16 全局 token 双分支（15872 维）"
                 "+ 时间平均 + 回归头；(A) baseline + (B) 半监督伪标签训练；"
                 "验证集 SROCC=0.6897 / PLCC=0.6561"),
        "scale": "1~5",
        "ckpt": str(PROJECT_ROOT / "runs" / "divide_semisup_v2" / "model_best.pt"),
        "available": (PROJECT_ROOT / "runs" / "divide_semisup_v2" / "model_best.pt").exists(),
        "builder": lambda ckpt: OursPredictor(ckpt),
    },
    {
        "id": "dover",
        "name": "DOVER（ICCV2023 零样本）",
        "desc": ("双视角评估器：技术分支 Video Swin GRPB（7x7 碎片采样，对闪烁/冻结"
                 "等时域失真敏感）+ 美学分支 ConvNeXt；LSVQ 预训练权重，未使用本课程"
                 "数据；零样本 SROCC=0.7110 / PLCC=0.7053"),
        "scale": "0~100",
        "ckpt": str(PROJECT_ROOT / "dover_repro" / "pretrained_weights" / "DOVER.pth"),
        "available": (PROJECT_ROOT / "dover_repro" / "pretrained_weights" / "DOVER.pth").exists(),
        "builder": lambda ckpt: DoverPredictor(ckpt),
    },
    {
        "id": "doverpp",
        "name": "DOVER++（课程数据微调）",
        "desc": ("在 DOVER 基础上用 DIVIDE-MaxWell 整体+技术+美学三列标注微调"
                 "（PLCC+rank 损失，官方报告 SROCC=0.8071 / PLCC=0.8126）；"
                 "本机复现权重（微调完成后自动可用）"),
        "scale": "1~5（两分支和）",
        "ckpt": str(_find_doverpp_ckpt()) if _find_doverpp_ckpt() else "",
        "available": _find_doverpp_ckpt() is not None,
        "builder": lambda ckpt: DoverPlusPlusPredictor(ckpt),
    },
]

_instances = {}


def get_model(model_id):
    """懒加载并缓存模型实例；不可用抛 KeyError。"""
    if model_id in _instances:
        return _instances[model_id]
    for m in MODELS:
        if m["id"] == model_id:
            if not m["available"] or not m["ckpt"]:
                raise RuntimeError(f"模型 {model_id} 权重不可用")
            inst = m["builder"](m["ckpt"])
            _instances[model_id] = inst
            return inst
    raise KeyError(f"未知模型: {model_id}")


def model_info_list():
    return [
        {"id": m["id"], "name": m["name"], "desc": m["desc"],
         "scale": m["scale"], "available": m["available"],
         "loaded": m["id"] in _instances}
        for m in MODELS
    ]
