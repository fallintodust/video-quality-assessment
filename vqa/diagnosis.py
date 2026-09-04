"""失真专项检测接口（小组分工集成点）。

组长整合流程：
    scripts/diagnose.py 对每个视频调用已注册的检测器 + 主模型 MOS，
    输出统一诊断报告（report.json / report.txt），
    与 score.txt（整体打分）配套交付。

组员接入约定（噪点 / 闪烁 / 模糊 各一人认领）：
    1. 在本文件对应槽位实现 detect(frames_rgb) -> dict
    2. 返回值格式（三人都一致）：
       {
         "score": float,    # 该失真的程度，统一 0.0~1.0（0=无，1=极重）
         "level": str,      # 分级："无" | "轻" | "中" | "重"
         "detail": dict,    # 可选附加信息（如闪烁频率、噪点方差、模糊半径）
       }
    3. frames_rgb: np.ndarray [T, size, size, 3] uint8 RGB（已抽好 T 帧，见 load_frames_rgb）
    4. 完成后把函数赋给对应全局槽位并注册，例如：
           FLICKER_DETECTOR = my_flicker_detector
           register_detector("闪烁", FLICKER_DETECTOR)
"""

import os

import numpy as np

from .config import Config
from .sampling import load_cached_frames, sample_frame_indices

# ---- 失真类型槽位（组员各自实现后填入） ----
NOISE_DETECTOR = None     # 组员1：噪点检测  detect(frames_rgb) -> dict
FLICKER_DETECTOR = None   # 组员2：闪烁检测  detect(frames_rgb) -> dict
BLUR_DETECTOR = None      # 组员3：模糊检测  detect(frames_rgb) -> dict

# 已注册检测器列表 [(名称, 检测器)]
_DETECTORS = []


def register_detector(name, detector):
    """注册一个失真检测器（组员实现后调用）。"""
    if detector is None:
        return
    _DETECTORS.append((name, detector))


def level_from_score(score, thresholds=(0.25, 0.5, 0.75)):
    """按分数映射分级：无 / 轻 / 中 / 重（阈值组内统一）。"""
    if score < thresholds[0]:
        return "无"
    if score < thresholds[1]:
        return "轻"
    if score < thresholds[2]:
        return "中"
    return "重"


def load_frames_rgb(video_path, T=None, size=224):
    """抽取 T 帧，返回 [T, size, size, 3] uint8 RGB。

    - 命中 Config.FRAME_CACHE 时直接读缓存（缓存为 uint8 BGR，此处转 RGB）
    - 否则 OpenCV 顺序解码（TSN 等间距抽帧，与训练/推理一致）
    """
    T = T if T is not None else Config.T

    cached = load_cached_frames(video_path, Config.FRAME_CACHE)
    if cached is not None:
        x = cached[:T]                      # uint8 BGR [T,size,size,3]
        return np.ascontiguousarray(x[..., ::-1])  # BGR -> RGB

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        total = n
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    idx = sample_frame_indices(total, T)
    wanted = set(int(i) for i in idx)
    frames = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frame = cv2.resize(frame, (size, size))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"视频无有效帧: {video_path}")
    while len(frames) < T:      # 帧数不足：最后一帧重复补齐
        frames.append(frames[-1])
    return np.stack(frames[:T])


def diagnose_video(video_path, T=None, model=None, device=None, use_fp16=False):
    """对单个视频做完整诊断。

    返回：
        {
          "video": 视频文件名,
          "mos": 主模型整体分（未提供 model 时为 None）,
          "issues": {检测器名称: {"score":..., "level":..., "detail":...}},
        }
    未实现的检测器槽位自动跳过。
    """
    from .train_utils import score_videos, get_device
    from .dataset import VideoDataset

    T = T if T is not None else Config.T
    frames = load_frames_rgb(video_path, T=T)

    issues = {}
    for name, det in _DETECTORS:
        try:
            issues[name] = det(frames)
        except Exception as e:  # 单个检测器失败不拖垮整体
            issues[name] = {"score": float("nan"), "level": "未知",
                            "detail": {"error": str(e)}}

    mos = None
    if model is not None:
        device = device or get_device()
        ds = VideoDataset.unlabeled(os.path.dirname(video_path),
                                    [os.path.basename(video_path)], T=T)
        s = score_videos(model, ds, device, batch_size=4, use_fp16=use_fp16)
        mos = float(s[os.path.basename(video_path)])

    return {"video": os.path.basename(video_path), "mos": mos, "issues": issues}


def diagnose_directory(video_dir, T=None, model=None, device=None,
                       use_fp16=False):
    """对目录下全部视频做诊断，返回按自然排序的结果列表。"""
    from .dataset import list_videos, natural_key

    names = list_videos(video_dir)
    out = []
    for n in sorted(names, key=natural_key):
        out.append(diagnose_video(os.path.join(video_dir, n), T=T,
                                  model=model, device=device, use_fp16=use_fp16))
    return out


# ======================================================================
# 参考实现（供组员对照接口格式；正式实现由组员替换对应槽位）
# ======================================================================

def heuristic_flicker(frames_rgb):
    """闪烁检测·参考实现（组员2可整体替换）。

    原理：相邻帧亮度差 + 帧亮度序列波动（帧差能量对亮度闪烁幅度敏感，
    合成闪烁数据集上已验证特征方向有效）。
    """
    gray = frames_rgb.astype(np.float32).mean(axis=3)          # [T,H,W]
    inter = np.abs(np.diff(gray, axis=0))                      # 相邻帧亮度差
    seq = gray.reshape(len(gray), -1).mean(axis=1)             # 每帧平均亮度
    seq_std = float(seq.std())
    # 经验尺度：帧差均值 ~16 或亮度序列 std ~16 视为"重"（合成数据标定，可调）
    score = float(np.clip((inter.mean() + seq_std) / 32.0, 0.0, 1.0))
    return {"score": score, "level": level_from_score(score),
            "detail": {"frame_diff_mean": float(inter.mean()),
                       "luma_seq_std": seq_std}}


# 参考实现默认注册，保证脚本开箱可跑；组员实现后替换即可。
# register_detector("闪烁", heuristic_flicker)
# ============================================================================
# 闪烁检测：接入训练好的时域模型（组员2 / ALEKSEEV PETR）
#
# 把本段追加到 vqa/diagnosis.py 末尾，替换原来的
#     register_detector("闪烁", heuristic_flicker)
# 一行（把那一行删掉或注释掉）。
#
# 设计说明：
#   - 懒加载：只有真正调用检测器时才载入骨干与权重，
#     `import vqa.diagnosis` 本身不会花几秒去建 ResNet。
#   - 有回退：torch 不可用、权重缺失或加载失败时，
#     自动退回启发式实现，diagnose.py 依然能跑通。
#   - 权重路径可用环境变量 FLICKER_CKPT 覆盖。
#
# 实测对比（T-5 验证集 200 视频，见 docs/flicker_detector.md）：
#     模型          SROCC 0.5807  PLCC 0.5976  Score 0.5891
#     启发式 v2     SROCC 0.2937  PLCC 0.2928  Score 0.2932
#     原启发式基线  更低
# ============================================================================

import os as _os
import sys as _sys

_FLICKER_CKPT = _os.environ.get(
    "FLICKER_CKPT",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                  "runs", "t5", "best_r50_mean+std+diff.pt"))

_model_flicker = None          # 懒加载后的实例
_model_failed = False          # 失败过就不再重试


def _get_model_flicker():
    global _model_flicker, _model_failed
    if _model_flicker is not None or _model_failed:
        return _model_flicker
    try:
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        scripts = _os.path.join(root, "scripts")
        if scripts not in _sys.path:
            _sys.path.insert(0, scripts)
        from flicker_detectors import ModelFlicker
        _model_flicker = ModelFlicker(_FLICKER_CKPT, scripts_dir=scripts)
        print(f"[闪烁] 已加载模型: {_os.path.basename(_FLICKER_CKPT)}")
    except Exception as e:
        _model_failed = True
        print(f"[闪烁] 模型加载失败（{e}），回退到启发式实现")
    return _model_flicker


def model_flicker_detector(frames_rgb):
    """闪烁/时域一致性检测（模型版，带启发式回退）。

    frames_rgb: [T, size, size, 3] uint8 RGB，由 load_frames_rgb 提供。

    注意抽帧方式：diagnose.py 用的是全局均匀抽帧（TSN 式），
    而模型是在连续片段上训练的。实测这一"不匹配"反而更好
    （Score 0.6186 对 0.5891），原因是 T-5 的抖动是秒级现象，
    均匀抽帧的时间窗口更长，详见 docs/cached_feature_pipeline.md 6.5 节。
    """
    m = _get_model_flicker()
    if m is None:
        return heuristic_flicker(frames_rgb)
    try:
        return m(frames_rgb)
    except Exception as e:
        print(f"[闪烁] 推理失败（{e}），本条回退到启发式")
        return heuristic_flicker(frames_rgb)


FLICKER_DETECTOR = model_flicker_detector
register_detector("闪烁", FLICKER_DETECTOR)
