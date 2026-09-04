#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flicker detectors for vqa/diagnosis.py (FLICKER_DETECTOR slot).

Two implementations, both returning the interface agreed in the group:
    detect(frames_rgb) -> {"score": 0.0..1.0, "level": str, "detail": dict}
where score 0 = no flicker, 1 = severe flicker.

  heuristic_flicker_v2  - pure numpy, no model, milliseconds per video.
                          Luminance frame differences + periodicity of the
                          luminance sequence (FFT) + temporal spread.
  ModelFlicker          - wraps the trained T-5 head; needs torch and a
                          checkpoint, ~1.4 s per video.

Use eval_flicker.py to measure both against the T-5 labels and decide which
one goes into the slot.
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


def _detrend(seq):
    """Remove mean and linear trend, so a slow fade is not read as flicker."""
    n = len(seq)
    if n < 3:
        return np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32)
    slope, intercept = np.polyfit(t, seq, 1)
    return seq - (slope * t + intercept)


def _periodicity(seq):
    """Share of AC power in the strongest frequency bin, 0..1.

    Flicker is periodic and concentrates its spectrum in one bin. Motion or a
    lighting change spreads energy broadly, giving a low value.
    """
    n = len(seq)
    if n < 4:
        return 0.0
    power = np.abs(np.fft.rfft(_detrend(seq))) ** 2
    ac = power[1:]
    tot = ac.sum()
    return float(ac.max() / tot) if tot > 1e-8 else 0.0


def heuristic_flicker_v2(frames_rgb, clip_len=None):
    """Flicker detection without a model.

    The key observation is that flicker and motion look very different in the
    per-frame MEAN luminance:

      - flicker pumps the whole frame brighter and darker, so the frame mean
        oscillates;
      - motion moves content around but barely changes the frame mean, even
        though per-pixel differences are large.

    So the primary signal is the spread of the detrended per-frame mean
    luminance, not the raw frame difference. Detrending removes slow fades;
    the periodicity term adds confidence when the oscillation is regular.
    `frame_diff_mean` is reported in detail for reference but carries little
    weight, precisely because motion dominates it.

    clip_len: if the frames were sampled as K clips of L consecutive frames,
              pass L so nothing is measured across clip borders.
    """
    g = _luma(frames_rgb)                           # [T,H,W]
    T = g.shape[0]

    def one(block):
        seq = block.reshape(block.shape[0], -1).mean(-1)   # per-frame mean luma
        res = _detrend(seq)
        return (float(np.abs(res).mean()),                 # pumping amplitude
                _periodicity(seq),
                float(np.abs(np.diff(block, axis=0)).mean()))

    if clip_len and T % clip_len == 0 and T > clip_len:
        blocks = g.reshape(T // clip_len, clip_len, *g.shape[1:])
        vals = [one(b) for b in blocks]
        pump = float(np.mean([v[0] for v in vals]))
        period = float(np.mean([v[1] for v in vals]))
        d_mean = float(np.mean([v[2] for v in vals]))
    else:
        pump, period, d_mean = one(g)

    # empirical scale: a detrended swing of ~8 grey levels is already severe.
    # Calibrate on the synthetic flicker set if the group changes the levels.
    raw = (pump / 8.0) * (0.4 + 0.6 * period)
    score = float(np.clip(raw, 0.0, 1.0))
    return {"score": score, "level": level_from_score(score),
            "detail": {"luma_pump": pump,
                       "periodicity": period,
                       "frame_diff_mean": d_mean}}


# ---------------------------------------------------------------- model

class ModelFlicker:
    """Wraps the trained T-5 head as a detector.

    The head predicts the stable/shaky axis, where a HIGH value means stable.
    The group interface wants the opposite (high = severe distortion), so the
    prediction is inverted and mapped to 0..1.

    Note on sampling: the head was trained on K clips of L consecutive frames.
    Feeding it uniformly spread frames changes what the diff features mean, so
    for a fair result the frames must be sampled the same way as in training
    (extract_feats.clip_indices).
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

        self.extractor = Extractor(use_vit=self.branch != "r50").to(self.dev)
        self.head = Head(len(ck["mu"])).to(self.dev)
        self.head.load_state_dict(ck["state"])
        self.head.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.y_mu, self.y_sd = ck["y_mu"], ck["y_sd"]

        # Range used to map the prediction into 0..1. The head stores the mean
        # and std of its training labels, not their min/max, so the range is
        # derived as mu +- 2.5 sd. That covers ~99% of a roughly normal label
        # distribution and, crucially, adapts to whatever scale the axis used:
        # 0..100 for the T-* axes, 1..5 for the O axis. Hard-coding 0..100
        # would squash every O-axis prediction into "severe".
        # SROCC/PLCC are unaffected either way (monotone transform), but the
        # 0..1 score and the 无/轻/中/重 levels are not.
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
        stability = (pred - self.lo) / (self.hi - self.lo)      # 0..1, high = stable
        score = float(np.clip(1.0 - stability, 0.0, 1.0))       # invert
        return {"score": score, "level": level_from_score(score),
                "detail": {"stability_pred": float(pred)}}


# ---------------------------------------------------------------- registration

def install(slot="FLICKER_DETECTOR", detector=None):
    """Plug a detector into vqa/diagnosis.py."""
    from vqa import diagnosis
    detector = detector or heuristic_flicker_v2
    setattr(diagnosis, slot, detector)
    diagnosis.register_detector("闪烁", detector)
    return detector
