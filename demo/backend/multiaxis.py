# demo/backend/multiaxis.py
"""Multi-axis diagnosis: one extractor, four regression heads, one heuristic.

A single score answers one question. This module runs four measurements plus
a flicker detector on the same clip:

    overall     head trained on the O axis (MaxWell overall MOS)
    shake       head trained on T-5 (stable/shaky)   - 4 variants selectable
    stutter     head trained on T-8 (fluent/choppy)
    temporal    head trained on tcons (frame-level quality regressed out)
    flicker     heuristic v2 - no model, no weights

All heads read the SAME cached feature pass, so the extra measurements cost
almost nothing on top of the first one. The extractor is built with ViT so
that r50-only heads can simply slice the first 13568 dimensions.

Why flicker is a heuristic and not a head: MaxWell has no flicker axis, so no
head can be trained for it. On an injected-flicker benchmark (n=600) the
heuristic scores 0.8554 against 0.5965 for the T-5 head - see
docs/flicker_detector.md.

Optional visuals (frames + plots) are returned as base64 PNG/JPEG so the
frontend can show them without extra round trips. They are off by default:
32 frames plus two plots is roughly 400 KB per video, which is wasteful in
batch mode.
"""
import base64
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from extract_feats import (Extractor, clip_indices,  # noqa: E402
                           IMAGENET_MEAN, IMAGENET_STD)
from train_head import Head, aggregate, CONV_DIM  # noqa: E402
from flicker_detectors import heuristic_flicker_v2, level_from_score  # noqa: E402

# sampling for the heads: exactly what they were trained on
N_CLIPS, CLIP_LEN, SIZE = 4, 8, 224

# sampling for the flicker heuristic: denser, because 4x8 covers only ~30% of
# a 106-frame clip and an 8-frame window is shorter than one period of a 3 Hz
# flicker. The heuristic runs no network, so this costs only decoding.
FLICKER_MAX_FRAMES = 256
FLICKER_CLIPS, FLICKER_CLIP_LEN = 8, 32

RUNS = PROJECT_ROOT / "runs"

# The four measurements. `default` is what the UI selects unless told otherwise;
# `variants` are the ablation checkpoints, exposed only where a choice is useful.
AXES = [
    {
        "id": "overall",
        "name": "整体质量",
        "desc": "画面综合主观质量：清晰度、噪点、曝光、构图",
        "dir": "o",
        "default": "best_all_mean+std+diff.pt",
        "scale": "1~5",
    },
    {
        "id": "shake",
        "name": "抖动",
        "desc": "摄像机抖动，秒级现象（MaxWell T-5 轴）",
        "dir": "t5",
        "default": "best_r50_mean+std+diff.pt",
        "scale": "0~100",
    },
    {
        "id": "stutter",
        "name": "卡顿",
        "desc": "播放不连续、帧序断裂（MaxWell T-8 轴）",
        "dir": "t8",
        "default": "best_all_mean+std+diff.pt",
        "scale": "0~100",
    },
    {
        "id": "temporal",
        "name": "纯时域",
        "desc": "抖动+卡顿中扣除单帧画质可解释的部分（残差目标）",
        "dir": "tcons",
        "default": "best_r50_mean+std+diff.pt",
        "scale": "0~100",
    },
]

# Measured Score = (SROCC+PLCC)/2 on the 909-video validation split, so the
# UI can show what each checkpoint actually achieved instead of a bare filename.
KNOWN_SCORE = {
    ("overall", "best_all_mean.pt"): 0.7008,
    ("overall", "best_all_mean+std+diff.pt"): 0.6857,
    ("shake", "best_all_mean.pt"): 0.6060,
    ("shake", "best_all_mean+std+diff.pt"): 0.6811,
    ("shake", "best_r50_mean+std+diff.pt"): 0.6834,
    ("shake", "best_vit_mean+std+diff.pt"): 0.6327,
    ("stutter", "best_all_mean.pt"): 0.4357,
    ("stutter", "best_all_mean+std+diff.pt"): 0.5006,
    ("temporal", "best_r50_mean.pt"): 0.2560,
    ("temporal", "best_r50_mean+std+diff.pt"): 0.4239,
}

BRANCH_LABEL = {"r50": "仅 ResNet-50", "vit": "仅 ViT-B/16", "all": "双分支"}


def _label(axis_id, filename):
    """Readable option label: branch, aggregation, and the measured Score."""
    stem = filename[:-3] if filename.endswith(".pt") else filename
    stem = stem[5:] if stem.startswith("best_") else stem
    branch, _, agg = stem.partition("_")
    txt = f"{BRANCH_LABEL.get(branch, branch)} + {agg}"
    if agg == "mean":
        txt += "（任务书方案）"
    sc = KNOWN_SCORE.get((axis_id, filename))
    return f"{txt}　Score {sc:.4f}" if sc is not None else txt


_extractor = None
_heads = {}                      # (axis_id, filename) -> loaded head dict
_lock = threading.Lock()


def _device():
    prefer = os.environ.get("DEMO_DEVICE", "cuda")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_extractor():
    global _extractor
    with _lock:
        if _extractor is None:
            _extractor = Extractor(use_vit=True).to(_device())
        return _extractor


def _ckpt_path(axis, filename=None):
    return RUNS / axis["dir"] / (filename or axis["default"])


def _get_head(axis, filename=None):
    """Lazy-load one head; cached by (axis, filename)."""
    fn = filename or axis["default"]
    key = (axis["id"], fn)
    with _lock:
        if key in _heads:
            return _heads[key]
        path = _ckpt_path(axis, fn)
        if not path.exists():
            raise FileNotFoundError(str(path))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ta = ck["args"]
        h = Head(len(ck["mu"])).to(_device())
        h.load_state_dict(ck["state"])
        h.eval()
        entry = {"head": h, "modes": ta["agg"].split("+"), "branch": ta["branch"],
                 "mu": ck["mu"], "sd": ck["sd"],
                 "y_mu": ck["y_mu"], "y_sd": ck["y_sd"], "file": fn}
        _heads[key] = entry
        return entry


def available():
    """What the frontend can offer: every axis with every checkpoint on disk.

    Each axis lists all of its trained variants (branch x aggregation), so the
    UI can offer one row per axis with the options as radio buttons - the same
    ablation that produced the numbers in the report.
    """
    out = []
    for a in AXES:
        d = RUNS / a["dir"]
        files = sorted(p.name for p in d.glob("*.pt")) if d.exists() else []
        # preferred default first, the rest after it
        if a["default"] in files:
            files = [a["default"]] + [f for f in files if f != a["default"]]
        out.append({
            "id": a["id"], "name": a["name"], "desc": a["desc"],
            "scale": a["scale"], "dir": a["dir"],
            "default": a["default"] if a["default"] in files
                       else (files[0] if files else None),
            "available": bool(files),
            "variants": [{"file": f, "label": _label(a["id"], f),
                          "available": True} for f in files],
        })
    return out


# ---------------------------------------------------------------- frames

def _cv2_frame_count(path):
    """OpenCV 解码时的帧数（CAP_PROP_FRAME_COUNT 不可信时边读边数）。"""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"无法打开视频: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.release()
    return n


def _cv2_read(path, wanted):
    """OpenCV 逐帧解码并返回 wanted 索引对应帧 [U,224,224,3] uint8 RGB。

    用于 decord 缺失的机器（Windows 无官方 decord 轮子）；
    采样索引语义与 decord 路径一致，仅解码器不同。
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"无法打开视频: {path}")
    want_set = set(int(i) for i in wanted)
    frames = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n in want_set:
            frame = cv2.resize(frame, (SIZE, SIZE))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n += 1
    cap.release()
    if not frames:
        raise IOError(f"视频无有效帧: {path}")
    while len(frames) < len(wanted):     # 帧数不足：末帧补齐
        frames.append(frames[-1])
    return np.stack(frames)


def _read_both(path):
    """Decode once for both consumers.

    The heads need exactly clip_indices(n, 4, 8) - that is what they were
    trained on. The flicker heuristic wants far denser coverage: 4x8 spans only
    ~30% of a 106-frame clip, and an 8-frame window is shorter than one period
    of a 3 Hz flicker.

    Decoding twice cost a full extra pass over the file, so instead we take the
    union of both index sets in a single get_batch and map each consumer back
    to its own frames.

    解码后端：优先 decord（与训练端一致）；decord 缺失时回退 OpenCV
    （逐帧解码 + BGR→RGB，索引语义不变）。
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        VideoReader = None

    if VideoReader is not None:
        vr = VideoReader(path, ctx=cpu(0), width=SIZE, height=SIZE, num_threads=2)
        n = len(vr)
    else:
        n = _cv2_frame_count(path)

    head_idx = clip_indices(n, N_CLIPS, CLIP_LEN)
    if n <= FLICKER_MAX_FRAMES:
        flick_idx = np.arange(n)
        flick_clip, note = n, f"全部 {n} 帧（100% 覆盖）"
    else:
        flick_idx = clip_indices(n, FLICKER_CLIPS, FLICKER_CLIP_LEN)
        cov = FLICKER_CLIPS * FLICKER_CLIP_LEN / n * 100
        flick_clip = FLICKER_CLIP_LEN
        note = f"{FLICKER_CLIPS}x{FLICKER_CLIP_LEN} / {n} 帧（{cov:.0f}% 覆盖）"

    union = np.unique(np.concatenate([head_idx, flick_idx]))
    if VideoReader is not None:
        batch = vr.get_batch(union).asnumpy()
    else:
        batch = _cv2_read(path, union)
    pos = {int(v): i for i, v in enumerate(union)}

    raw = batch[[pos[int(i)] for i in head_idx]]
    fraw = batch[[pos[int(i)] for i in flick_idx]]
    return raw, fraw, flick_clip, note, n


def _normalise(raw):
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.)
    return (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
        torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def _features(x, chunk=16):
    dev = _device()
    ex = _get_extractor()
    outs = []
    with torch.no_grad():
        for j in range(0, x.shape[0], chunk):
            with torch.autocast(dev.type, dtype=torch.float16,
                                enabled=(dev.type == "cuda")):
                outs.append(ex(x[j:j + chunk].to(dev)).float())
    return torch.cat(outs).cpu().numpy().astype(np.float32)


def _head_score(entry, seq):
    s = seq
    if entry["branch"] == "r50":
        s = s[:, :CONV_DIM]
    elif entry["branch"] == "vit":
        s = s[:, CONV_DIM:]
    v = aggregate(s, entry["modes"], CLIP_LEN)
    v = (v - entry["mu"]) / entry["sd"]
    with torch.no_grad():
        p = entry["head"](torch.from_numpy(v[None]).float().to(_device())).item()
    return p * entry["y_sd"] + entry["y_mu"]


def _severity(pred, y_mu, y_sd):
    """Heads predict goodness; the panel shows how bad, 0..1.

    The checkpoint stores mean and std of the training labels, not min/max, so
    the range is taken as mu +- 2.5 sd. That adapts to whatever scale the axis
    used (0..100 for T-*, 1..5 for O); hard-coding 0..100 would squash every
    O-axis prediction into the top severity band.
    """
    lo, hi = y_mu - 2.5 * y_sd, y_mu + 2.5 * y_sd
    return float(np.clip(1.0 - (pred - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------- visuals

def _b64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_frames(raw, quality=80):
    """Sampled frames as base64 JPEG, in order."""
    import cv2
    out = []
    for i in range(raw.shape[0]):
        bgr = cv2.cvtColor(raw[i], cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out.append(base64.b64encode(buf.tobytes()).decode())
    return out


def _plots(seq, fraw):
    """Same two panels as scripts/app.py - default matplotlib palette."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K, L = N_CLIPS, CLIP_LEN
    d = seq.reshape(K, L, -1)
    feat_step = np.abs(np.diff(d, axis=1)).mean(-1)          # [K, L-1]

    g = 0.299 * fraw[..., 0] + 0.587 * fraw[..., 1] + 0.114 * fraw[..., 2]
    luma = g.reshape(len(fraw), -1).mean(-1)

    fig1, ax = plt.subplots(figsize=(5.2, 2.6), dpi=90)
    for k in range(K):
        ax.plot(range(1, L), feat_step[k], marker="o", ms=3, label=f"clip {k+1}")
    ax.set_title("Feature instability")
    ax.set_xlabel("frame step within clip")
    ax.set_ylabel("mean |change|")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.2, 2.6), dpi=90)
    ax2.plot(luma - luma.mean(), lw=1)
    ax2.set_title("Frame brightness (centred) - oscillation = flicker")
    ax2.set_xlabel("sampled frame")
    ax2.set_ylabel("luma - mean")
    ax2.grid(alpha=.3)
    fig2.tight_layout()

    return {"instability": _b64_png(fig1), "luma": _b64_png(fig2)}


# ---------------------------------------------------------------- entry point

def extra_detectors():
    """Detectors registered by teammates in vqa.diagnosis, minus our own.

    Flicker is excluded: this module already runs heuristic_flicker_v2 with its
    own denser sampling, and registering it twice would score it twice.
    """
    try:
        from vqa.diagnosis import _DETECTORS
    except Exception:
        return []
    return [name for name, _ in _DETECTORS if name != "闪烁"]


def _run_extra(raw, wanted):
    """Run the selected teammate detectors on the frames we already decoded."""
    if not wanted:
        return {}
    try:
        from vqa.diagnosis import _DETECTORS
    except Exception:
        return {}
    out = {}
    for name, det in _DETECTORS:
        if name not in wanted or name == "闪烁":
            continue
        try:
            r = det(raw)
            out[name] = {"score": round(float(r["score"]), 3),
                         "level": r.get("level", "未知")}
        except Exception as e:
            out[name] = {"score": 0.0, "level": "未知", "error": str(e)}
    return out


def analyse(video_path, variants=None, with_visuals=False, extras=None):
    """Run every available measurement on one video.

    variants     : {axis_id: checkpoint filename}, missing axes use the default
    with_visuals : also return sampled frames and the two plots as base64
    extras       : names of vqa.diagnosis detectors to also run (noise, blur)
    """
    variants = variants or {}
    raw, fraw, fclip, fnote, n_total = _read_both(video_path)
    seq = _features(_normalise(raw))

    measurements = {}
    for a in AXES:
        fn = variants.get(a["id"])
        try:
            entry = _get_head(a, fn)
        except FileNotFoundError:
            continue
        pred = _head_score(entry, seq)
        sev = _severity(pred, entry["y_mu"], entry["y_sd"])
        measurements[a["id"]] = {
            "name": a["name"], "desc": a["desc"], "scale": a["scale"],
            "prediction": round(pred, 2),
            "severity": round(sev, 3),
            "level": level_from_score(sev),
            "checkpoint": entry["file"],
        }

    fl = heuristic_flicker_v2(fraw, clip_len=fclip)
    measurements["flicker"] = {
        "name": "闪烁", "desc": "亮度周期性抖动（启发式，无需权重）",
        "scale": "0~1",
        "prediction": None,
        "severity": round(fl["score"], 3),
        "level": fl["level"],
        "checkpoint": "heuristic_v2",
        "detail": {k: round(v, 3) for k, v in fl["detail"].items()},
        "sampling": fnote,
    }

    extra = _run_extra(raw, extras)
    for name, r in extra.items():
        measurements[name] = {
            "name": name, "desc": "组员实现的检测器（vqa.diagnosis）",
            "scale": "0~1", "prediction": None,
            "severity": r["score"], "level": r["level"],
            "checkpoint": "vqa.diagnosis",
        }

    out = {
        "video": os.path.basename(video_path),
        "num_frames": n_total,
        "sampling": f"{N_CLIPS}x{CLIP_LEN} 连续帧",
        "measurements": measurements,
    }
    if with_visuals:
        out["frames"] = _b64_frames(raw)
        out["plots"] = _plots(seq, fraw)
    return out
