from .swin_backbone import SwinTransformer3D as VQABackbone
from .swin_backbone import SwinTransformer2D as IQABackbone
from .head import VQAHead, IQAHead, VARHead
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
# X-CLIP 分支需要 openai/CLIP，本项目只用 FAST-VQA / FasterVQA，用不到它。
# 设为可选导入，避免为了跑推理而额外装一个包。
try:
    from .xclip_backbone import build_x_clip_model
except ImportError:      # clip 未安装
    build_x_clip_model = None
from .evaluator import BaseEvaluator, BaseImageEvaluator, DiViDeAddEvaluator

__all__ = [
    "VQABackbone",
    "IQABackbone",
    "VQAHead",
    "IQAHead",
    "VARHead",
    "BaseEvaluator",
    "BaseImageEvaluator",
    "DiViDeAddEvaluator",
]
