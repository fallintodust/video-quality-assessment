# demo/backend/fastvqa_predictor.py
"""FAST-VQA / FasterVQA inference for the demo.

The upstream package is vendored under third_party/fastvqa (models and
datasets only, ~250 KB of code). The X-CLIP branch there needs openai/CLIP;
we do not use it, so that import was made optional.

Sampling follows the upstream val configs: fragments 7x7x32x32, clip_len 32.
FAST-VQA-B takes 4 clips, FasterVQA takes 1 clip with 3D fragment sampling.
Scores are mapped to 0..100 with the upstream sigmoid constants, which were
calibrated on LSVQ - a monotone transform, so SROCC is unaffected.

Measured on the 909-video validation split (docs/fastvqa_comparison.md):

    FAST-VQA-B   279 G MACs   SROCC 0.7102  PLCC 0.7111  OBJ 1.4213  7.47 s
    FasterVQA     69 G MACs   SROCC 0.6713  PLCC 0.6770  OBJ 1.3483  1.56 s
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

# upstream sigmoid_rescale constants (mean, std) per model
MEAN_STDS = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FAST-VQA": (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006),
}

OPTS = {
    "FasterVQA": "f3dvqa-b.yml",
    "FAST-VQA": "fast-b.yml",
    "FAST-VQA-M": "fast-m.yml",
}


def options_dir():
    return THIRD_PARTY / "fastvqa_options" / "fast"


def weights_dir():
    return PROJECT_ROOT / "pretrained_weights"


def _resolve_ckpt(raw_path):
    """Configs point at ./pretrained_weights/... relative to the upstream repo."""
    name = os.path.basename(raw_path)
    p = weights_dir() / name
    if p.exists():
        return p
    # release files use underscores where the configs use a star
    alt = weights_dir() / name.replace("*", "_")
    return alt if alt.exists() else p


def is_available(model_name):
    return (options_dir() / OPTS[model_name]).exists() and \
        _resolve_ckpt(yaml.safe_load(
            open(options_dir() / OPTS[model_name],
                 encoding="utf-8"))["test_load_path"]).exists()


class FastVQAPredictor:
    """One model of the FAST-VQA family, loaded lazily and kept warm."""

    def __init__(self, model_name="FAST-VQA", device=None):
        from fastvqa.datasets import (FragmentSampleFrames, SampleFrames,
                                      get_spatial_fragments)
        from fastvqa.models import DiViDeAddEvaluator

        self._FragmentSampleFrames = FragmentSampleFrames
        self._SampleFrames = SampleFrames
        self._fragments = get_spatial_fragments

        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        opt_path = options_dir() / OPTS[model_name]
        with open(opt_path, "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)

        ckpt = _resolve_ckpt(opt["test_load_path"])
        if not ckpt.exists():
            raise FileNotFoundError(f"权重缺失: {ckpt}")

        self.model = DiViDeAddEvaluator(**opt["model"]["args"]).to(self.device)
        state = torch.load(ckpt, map_location=self.device)["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.samplers = self._build_samplers(opt)

    def _build_samplers(self, opt):
        """Configs differ in which val split they define; any val-* carries the
        same fragment settings, so take the first one present."""
        data = opt["data"]
        split = "val-kv1k" if "val-kv1k" in data else \
            next((k for k in data if k.startswith("val")), None)
        if split is None:
            raise RuntimeError(f"配置里没有 val 分组: {list(data)}")
        t_opt = data[split]["args"]
        out = {}
        for stype, sargs in t_opt["sample_types"].items():
            # t_frag may sit at args level or inside the sample type; upstream
            # vqa.py only checks the args level, which silently drops the 3D
            # fragment sampling FasterVQA was trained with.
            t_frag = sargs.get("t_frag", t_opt.get("t_frag", 1))
            if t_frag > 1:
                sampler = self._FragmentSampleFrames(
                    fsize_t=sargs["clip_len"] // t_frag,
                    fragments_t=t_frag,
                    num_clips=sargs.get("num_clips", 1))
            else:
                sampler = self._SampleFrames(
                    clip_len=sargs["clip_len"],
                    num_clips=sargs.get("num_clips", 1))
            out[stype] = (sampler, sargs)
        return out

    def _sigmoid_rescale(self, score):
        mean, std = MEAN_STDS[self.model_name]
        return float(1 / (1 + np.exp(-(score - mean) / std)))

    def predict(self, video_path):
        try:
            import decord
            vr = decord.VideoReader(str(video_path))
        except ImportError:
            # decord 无 Windows 轮子：OpenCV 逐帧解码回退（索引语义与 decord 一致）
            import cv2

            class _CVVideoReader:
                def __init__(self, path):
                    self._cap = cv2.VideoCapture(str(path))
                    self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

                def __len__(self):
                    return self._n

                def __getitem__(self, i):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                    ok, fr = self._cap.read()
                    if not ok:
                        fr = np.zeros((224, 224, 3), np.uint8)
                    return torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

            vr = _CVVideoReader(video_path)
        vsamples = {}
        for stype, (sampler, sargs) in self.samplers.items():
            frames = sampler(len(vr))
            frame_dict = {i: vr[i] for i in np.unique(frames)}
            video = torch.stack([frame_dict[i] for i in frames], 0)
            video = video.permute(3, 0, 1, 2)
            sampled = self._fragments(video, **sargs)
            mean = torch.FloatTensor([123.675, 116.28, 103.53])
            std = torch.FloatTensor([58.395, 57.12, 57.375])
            sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std)
            sampled = sampled.permute(3, 0, 1, 2)
            n_clips = sargs.get("num_clips", 1)
            sampled = sampled.reshape(sampled.shape[0], n_clips, -1,
                                      *sampled.shape[2:]).transpose(0, 1)
            vsamples[stype] = sampled.to(self.device)
        with torch.no_grad():
            raw = self.model(vsamples).mean().item()
        return self._sigmoid_rescale(raw) * 100.0
