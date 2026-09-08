#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise detectors for vqa/diagnosis.py (NOISE_DETECTOR slot).

Mirrors scripts/flicker_detectors.py. Both implementations return the
interface agreed in the group:
    detect(frames_rgb) -> {"score": 0.0..1.0, "level": str, "detail": dict}
where score 0 = no noise, 1 = severe noise.

  heuristic_noise  - pure numpy, no model, milliseconds per video.
                     Spatial high-frequency energy + a robust estimate of
                     temporal noise sigma from gated frame differences
                     (see the function docstring).
  ModelNoise       - wraps a head trained on the MaxWell T-3 (noiseless) axis
                     via the cached-feature pipeline (extract_feats.py +
                     train_head.py); needs torch and a checkpoint.

Label direction: T-3 "noiseless" is high when the video is clean, and the
group interface wants high = severe distortion, so the prediction is
inverted (like ModelFlicker inverts T-5 "stable/shaky").

To train the T-3 checkpoint the group uses the same pipeline as the T-5
head, e.g.:
    python3 make_labels.py --col T-3 --train-txt data/divide/train_lable_train.txt \
        --test-txt data/divide/train_lable_test.txt --out data/divide
    python3 train_head.py --feats ~/trkv/feats \
        --train-labels data/divide/t3_train_lable_train.txt \
        --val-labels   data/divide/t3_train_lable_test.txt \
        --agg mean+std+diff --branch r50 --out runs/t3
"""
import os
import sys

import numpy as np


def level_from_score(score, thresholds=(0.25, 0.5, 0.75)):
    if score < thresholds[0]:
        return "无"
    if score < thresholds[1]:
        return "轻"
    if score < thresholds[2]:
        return "中"
    return "重"


# ---------------------------------------------------------------- heuristic

def _luma(frames_rgb):
    """[T,H,W,3] uint8 -> [T,H,W] float32 luminance (Rec. 601)."""
    f = frames_rgb.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _box_blur(x, k=5):
    """Separable box blur on the last two axes, numpy-only (edge-replicated).

    The same effect as cv2.blur for our purposes; windowed sums come from a
    cumulative sum along each axis, so the cost is O(HW) per axis.
    """
    x = np.asarray(x, dtype=np.float32)
    r = k // 2
    y = x
    for axis in (len(x.shape) - 2, len(x.shape) - 1):
        widths = [(0, 0)] * x.ndim
        widths[axis] = (r, r)
        padded = np.pad(y, widths, mode="edge")        # length N = n + 2r
        # prefix-sum with a leading 0 so a window [i, i+k) is c[i+k] - c[i]
        c = np.concatenate(
            [np.zeros(padded.shape[:axis] + (1,) + padded.shape[axis + 1:]),
             np.cumsum(padded, axis=axis)], axis=axis)
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(k, None)
        win = c[tuple(sl)].copy()                      # c[i+k], length N-k+1 = n
        sl[axis] = slice(None, -k)
        win -= c[tuple(sl)]                            # minus c[i]
        y = win / float(k)
    return y


def _robust_sigma(x):
    """MAD-based scale: 1.4826 * median(|x - median|), robust to outliers."""
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def heuristic_noise(frames_rgb, clip_len=None):
    """Temporal-noise / grain estimate without a model.

    Key observation, the mirror image of the flicker member's trick: flicker
    pumps the whole frame and is caught in the frame-MEAN luma; temporal
    noise is i.i.d. per pixel and per frame, so it survives *gating* while
    texture and motion do not.

      - spatial part: the per-frame high-frequency residual res = luma - box
        blur(luma). Texture and edges have large, STABLE residuals; noise
        adds a small amount on top. res alone cannot separate grain from
        texture, which is why it only feeds the detail fields below.
      - temporal part (the actual signal): for each pair of consecutive
        frames, take the frame difference d and keep only pixels that are
        smooth in BOTH frames (|res| < SMOOTH_GATE). Static texture cancels
        in d and is smooth -> excluded, so it contributes nothing; a moving
        edge shifts every frame, so few pixels are smooth on both sides and
        motion contributes little; noise does not cancel and is captured.
        The MAD scale of the gated difference is a robust estimate of
        sqrt(2) * sigma_noise (difference of two independent draws).

    Score calibration (grey levels on a 0..255 LUMA scale): sigma ~ 3/6/9
    map to the group thresholds 0.25/0.5/0.75 through score = sigma / 12.
    Note the estimate lives in luma space - the Rec. 601 weighting compresses
    per-channel RGB sigma by ~0.67 - and gets conservative (truncated) when
    noise is heavy, because the smooth gate |res| < SMOOTH_GATE excludes an
    increasing share of pixels. Both effects are expected and tunable.

    clip_len: if the frames were sampled as K clips of L consecutive frames,
              pass L so nothing is measured across clip borders (mirrors
              heuristic_flicker_v2).
    """
    g = _luma(frames_rgb)                       # [T,H,W]
    T = g.shape[0]
    if T < 2:
        return {"score": 0.0, "level": "无",
                "detail": {"error": "need at least 2 frames for temporal noise"}}

    # constant noise-sigma calibration (grey levels on a 0..255 luma scale)
    scale = 12.0
    gate = 8.0                                  # |res| below this counts as smooth

    def one(block):
        res = block - _box_blur(block)          # [L,H,W] high-frequency residual
        hf_t = np.sqrt(np.mean(res.reshape(block.shape[0], -1) ** 2, axis=1))
        # per-transition robust noise estimate over smooth-in-both pixels
        sm = np.abs(res) < gate
        sig, cov = [], []
        for t in range(block.shape[0] - 1):
            m = sm[t] & sm[t + 1]
            cov.append(float(m.mean()))
            if m.sum() < 50:                    # too few smooth pixels: skip
                continue
            d = block[t + 1] - block[t]
            sel = d[m]
            sig.append(_robust_sigma(sel))
        if not sig:
            return (0.0, float(hf_t.mean()), float(hf_t.std()),
                    float(np.mean(cov) if cov else 0.0),
                    float("nan"), float("nan"))
        sigma_t = np.mean(sig) / np.sqrt(2.0)   # undo the sqrt(2) of differencing
        return (float(sigma_t), float(hf_t.mean()), float(hf_t.std()),
                float(np.mean(cov)), float(np.min(sig)), float(np.max(sig)))

    if clip_len and T % clip_len == 0 and T > clip_len:
        blocks = g.reshape(T // clip_len, clip_len, *g.shape[1:])
        vals = [one(b) for b in blocks]
        sigma = float(np.mean([v[0] for v in vals]))
        hf_mean = float(np.mean([v[1] for v in vals]))
        hf_std = float(np.mean([v[2] for v in vals]))
        cov_mean = float(np.mean([v[3] for v in vals]))
        sig_min = float(np.min([v[4] for v in vals]))
        sig_max = float(np.max([v[5] for v in vals]))
    else:
        sigma, hf_mean, hf_std, cov_mean, sig_min, sig_max = one(g)

    score = float(np.clip(sigma / scale, 0.0, 1.0))
    return {"score": score, "level": level_from_score(score),
            "detail": {"noise_sigma": sigma,         # 时域噪声 sigma(灰阶)
                       "hf_energy_mean": hf_mean,    # 空域高频能量(RMS,帧平均)
                       "hf_energy_std": hf_std,      # 高频能量的帧间波动(时域稳定)
                       "smooth_coverage": cov_mean,  # 参与估计的平滑像素占比
                       "sigma_min": sig_min, "sigma_max": sig_max}}


# ---------------------------------------------------------------- model

class ModelNoise:
    """Wraps a head trained on the T-3 (noiseless) axis as a detector.

    The head predicts noiselessness, where a HIGH value means clean. The
    group interface wants the opposite (high = severe distortion), so the
    prediction is inverted and mapped to 0..1 (same scheme as ModelFlicker:
    lo/hi are derived as y_mu +- 2.5 y_sd from the checkpoint, so any label
    scale of the trained axis works).

    The checkpoint is produced by train_head.py; its `args` store branch,
    aggregation modes and clip_len, so the detector always feeds the head
    exactly as it was trained.
    """

    def __init__(self, ckpt_path, scripts_dir=None, lo=None, hi=None):
        import torch
        sys.path.insert(0, scripts_dir or os.path.dirname(os.path.abspath(__file__)))
        from extract_feats import Extractor, IMAGENET_MEAN, IMAGENET_STD
        from train_head import Head, aggregate, CONV_DIM

        self.torch = torch
        self._agg, self._CONV = aggregate, CONV_DIM
        self._mean, self._std = IMAGENET_MEAN, IMAGENET_STD

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ta = ck["args"]
        self.modes = ta["agg"].split("+")
        self.branch = ta["branch"]
        self.clip_len = ta["clip_len"]
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        self.extractor = Extractor(use_vit=False).to(self.dev)
        self.head = Head(len(ck["mu"])).to(self.dev)
        self.head.load_state_dict(ck["state"])
        self.head.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.y_mu, self.y_sd = ck["y_mu"], ck["y_sd"]

        # Range used to map the prediction into 0..1 (see ModelFlicker for the
        # rationale: y_mu +- 2.5 y_sd adapts to the axis's label scale).
        if lo is None or hi is None:
            lo = self.y_mu - 2.5 * self.y_sd
            hi = self.y_mu + 2.5 * self.y_sd
        self.lo, self.hi = float(lo), float(hi)

    def features(self, frames_rgb):
        torch = self.torch
        x = torch.from_numpy(frames_rgb.copy()).permute(0, 3, 1, 2).float().div_(255.)
        x = (x - torch.tensor(self._mean).view(1, 3, 1, 1)) / \
            torch.tensor(self._std).view(1, 3, 1, 1)
        outs = []
        with torch.no_grad():
            for j in range(0, x.shape[0], 16):
                with torch.autocast(self.dev, dtype=torch.float16,
                                    enabled=(self.dev == "cuda")):
                    outs.append(self.extractor(x[j:j + 16].to(self.dev)).float())
        return torch.cat(outs).cpu().numpy().astype(np.float32)

    def predict_from_features(self, seq):
        """seq: [T, D] per-frame features (may come straight from the cache)."""
        if self.branch == "r50":
            seq = seq[:, :self._CONV]
        elif self.branch == "vit":
            seq = seq[:, self._CONV:]
        v = self._agg(seq, self.modes, self.clip_len)
        v = (v - self.mu) / self.sd
        with self.torch.no_grad():
            p = self.head(self.torch.from_numpy(v[None]).float().to(self.dev)).item()
        return p * self.y_sd + self.y_mu

    def __call__(self, frames_rgb):
        pred = self.predict_from_features(self.features(frames_rgb))
        noiselessness = (pred - self.lo) / (self.hi - self.lo)   # 0..1, high = clean
        score = float(np.clip(1.0 - noiselessness, 0.0, 1.0))    # invert
        return {"score": score, "level": level_from_score(score),
                "detail": {"noiseless_pred": float(pred)}}


# ---------------------------------------------------------------- module-level detect
# `detect` is what vqa/diagnosis.py imports at module load (guarded, see the
# header of diagnosis.py): once this file exposes it, the NOISE_DETECTOR slot
# stops being a placeholder without any further edits there. Model is loaded
# lazily on first call and cached; failures fall back to heuristic_noise.

_NOISE_CKPT = os.environ.get(
    "NOISE_CKPT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "runs", "t3", "best_r50_mean+std+diff.pt"))

_model_noise = None          # 懒加载后的实例
_model_failed = False        # 失败过就不再重试


def _get_model_noise():
    global _model_noise, _model_failed
    if _model_noise is not None or _model_failed:
        return _model_noise
    try:
        _model_noise = ModelNoise(_NOISE_CKPT, scripts_dir=os.path.dirname(
            os.path.abspath(__file__)))
        print(f"[噪点] 已加载模型: {os.path.basename(_NOISE_CKPT)}")
    except Exception as e:
        _model_failed = True
        print(f"[噪点] 模型加载失败（{e}），回退到启发式实现")
    return _model_noise


def detect(frames_rgb):
    """噪点检测（模型版，带启发式回退），契约同 vqa/diagnosis.py 文件头。

    frames_rgb: [T, size, size, 3] uint8 RGB，由 load_frames_rgb 提供。

    注意抽帧方式：diagnose.py 用的是全局均匀抽帧（TSN 式），而 T-3 模型头在
    连续片段特征上训练；与闪烁槽相同的取舍，详见 docs/cached_feature_pipeline.md。
    """
    m = _get_model_noise()
    if m is None:
        return heuristic_noise(frames_rgb)
    try:
        return m(frames_rgb)
    except Exception as e:
        print(f"[噪点] 推理失败（{e}），本条回退到启发式")
        return heuristic_noise(frames_rgb)


# ---------------------------------------------------------------- registration

def install(slot="NOISE_DETECTOR", detector=None):
    """Plug a detector into vqa/diagnosis.py.

    NOTE: vqa/diagnosis.py already imports `detect` above at module load and
    registers the slot itself, so do NOT call install() in that setup or the
    "噪点" entry would be registered twice. install() exists for standalone
    / eval use and for parity with scripts/flicker_detectors.py.
    """
    from vqa import diagnosis
    detector = detector or heuristic_noise
    setattr(diagnosis, slot, detector)
    diagnosis.register_detector("噪点", detector)
    return detector
