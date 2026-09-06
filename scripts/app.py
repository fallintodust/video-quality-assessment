#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio UI: multi-measure video quality diagnosis.

A single score answers only one question. This panel runs four measurements on
the same clip, because they detect different things:

    overall quality     head trained on the O axis (MaxWell overall MOS)
    camera shake        head trained on T-5 (stable/shaky)
    stutter             head trained on T-8 (fluent/choppy)
    brightness flicker  heuristic v2 - no model, no weights

Why the flicker slot is a heuristic and not a head: MaxWell has no flicker
axis, so no head can be trained for it. On an injected-flicker benchmark
(n=600) the heuristic scores 0.8554 against 0.5965 for the T-5 head - see
docs/flicker_detector.md.

All heads share one feature extraction pass, so extra measurements are nearly
free.

Usage:
    python3 scripts/app.py
    python3 scripts/app.py --runs runs --port 7860
"""
import argparse
import glob
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_feats import (Extractor, clip_indices,  # noqa: E402
                           IMAGENET_MEAN, IMAGENET_STD)
from train_head import Head, aggregate, CONV_DIM  # noqa: E402
from flicker_detectors import heuristic_flicker_v2, level_from_score  # noqa: E402

N_CLIPS, CLIP_LEN, SIZE = 4, 8, 224

# Each measurement maps to one runs/<sub>/ directory. Preferred checkpoints are
# tried in order; whatever else is in that directory becomes a selectable
# alternative in the UI.
MEASURES = [
    ("Overall quality", "o",
     ["best_all_mean.pt", "best_r50_mean.pt", "best_all_mean+std+diff.pt"],
     "General picture quality: sharpness, noise, exposure, composition"),
    ("Camera shake", "t5",
     ["best_r50_mean+std+diff.pt", "best_all_mean+std+diff.pt"],
     "Handheld wobble. A seconds-scale phenomenon"),
    ("Stutter", "t8",
     ["best_all_mean+std+diff.pt", "best_r50_mean+std+diff.pt"],
     "Broken playback continuity, frame-scale"),
    ("Temporal residual", "tcons",
     ["best_r50_mean+std+diff.pt", "best_all_mean+std+diff.pt"],
     "Shake+stutter with frame-level quality regressed out"),
]

STATE = {"heads": [], "extractor": None, "dev": "cpu"}


def list_ckpts(runs_dir, sub, preferred):
    """All checkpoints in runs/<sub>/, preferred ones first."""
    found = sorted(glob.glob(os.path.join(runs_dir, sub, "*.pt")))
    names = [os.path.basename(p) for p in found]
    ordered = [n for n in preferred if n in names]
    ordered += [n for n in names if n not in ordered]
    return [os.path.join(runs_dir, sub, n) for n in ordered]


def load_head(path):
    dev = STATE["dev"]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = ck["args"]
    h = Head(len(ck["mu"])).to(dev)
    h.load_state_dict(ck["state"])
    h.eval()
    return {"head": h, "modes": ta["agg"].split("+"), "branch": ta["branch"],
            "mu": ck["mu"], "sd": ck["sd"],
            "y_mu": ck["y_mu"], "y_sd": ck["y_sd"],
            "path": path, "agg": ta["agg"]}


def scan(runs_dir):
    """What is available on disk, per measurement."""
    STATE["runs_dir"] = runs_dir
    STATE["choices"] = {}
    for title, sub, pref, _ in MEASURES:
        STATE["choices"][title] = list_ckpts(runs_dir, sub, pref)
    return STATE["choices"]


def ensure_extractor():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["dev"] = dev
    if STATE["extractor"] is None:
        STATE["extractor"] = Extractor(use_vit=True).to(dev)
    return dev


def apply_selection(*paths):
    """Load exactly the checkpoints picked in the UI. Empty = measurement off."""
    dev = ensure_extractor()
    heads, lines = [], []
    for (title, sub, _, desc), path in zip(MEASURES, paths):
        if not path:
            lines.append(f"  {title:20s}  (off)")
            continue
        try:
            e = load_head(path)
        except Exception as exc:
            lines.append(f"  {title:20s}  FAILED: {exc}")
            continue
        e.update(title=title, desc=desc)
        heads.append(e)
        lines.append(f"  {title:20s}  {os.path.basename(path)}"
                     f"   [{e['agg']} / {e['branch']}]")
    STATE["heads"] = heads
    return (f"device: {dev}   |   {len(heads)} head(s) + heuristic flicker\n"
            + "\n".join(lines))


def read_frames(path):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), width=SIZE, height=SIZE, num_threads=1)
    n_total = len(vr)
    idx = clip_indices(n_total, N_CLIPS, CLIP_LEN)
    raw = vr.get_batch(idx).asnumpy()
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.)
    x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
        torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return raw, x, n_total


def extract(x, chunk=16):
    dev = STATE["dev"]
    outs = []
    with torch.no_grad():
        for j in range(0, x.shape[0], chunk):
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                outs.append(STATE["extractor"](x[j:j + chunk].to(dev)).float())
    return torch.cat(outs).cpu().numpy().astype(np.float32)


def head_score(entry, seq):
    s = seq
    if entry["branch"] == "r50":
        s = s[:, :CONV_DIM]
    elif entry["branch"] == "vit":
        s = s[:, CONV_DIM:]
    v = aggregate(s, entry["modes"], CLIP_LEN)
    v = (v - entry["mu"]) / entry["sd"]
    with torch.no_grad():
        p = entry["head"](torch.from_numpy(v[None]).float().to(STATE["dev"])).item()
    return p * entry["y_sd"] + entry["y_mu"]


def to_severity(pred, y_mu, y_sd):
    """Heads predict 'goodness'; the panel shows 'how bad', 0..1."""
    lo, hi = y_mu - 2.5 * y_sd, y_mu + 2.5 * y_sd
    return float(np.clip(1.0 - (pred - lo) / (hi - lo), 0.0, 1.0))


def make_plot(seq, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K, L = N_CLIPS, CLIP_LEN
    d = seq.reshape(K, L, -1)
    feat_step = np.abs(np.diff(d, axis=1)).mean(-1)                 # [K, L-1]

    g = 0.299 * raw[..., 0] + 0.587 * raw[..., 1] + 0.114 * raw[..., 2]
    luma = g.reshape(K, L, -1).mean(-1)                             # [K, L]

    fig, ax = plt.subplots(1, 2, figsize=(10, 2.8), dpi=110)
    for k in range(K):
        ax[0].plot(range(1, L), feat_step[k], marker="o", ms=3, label=f"clip {k+1}")
        ax[1].plot(range(L), luma[k] - luma[k].mean(), marker="o", ms=3)
    ax[0].set_title("Feature instability")
    ax[0].set_xlabel("frame step within clip")
    ax[0].set_ylabel("mean |change|")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=.3)
    ax[1].set_title("Frame brightness (centred) - sawtooth = flicker")
    ax[1].set_xlabel("frame in clip")
    ax[1].set_ylabel("luma - mean")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def analyse(video, show_frames):
    if not video:
        return "Upload a video first.", None, None
    if not STATE["heads"]:
        return ("No checkpoints loaded. Pick them above and press Load, "
                "or check that runs/ contains .pt files."), None, None
    t0 = time.time()
    try:
        raw, x, n_total = read_frames(video)
    except Exception as e:
        return f"Could not read the video: {e}", None, None

    seq = extract(x)
    rows = []
    for e in STATE["heads"]:
        pred = head_score(e, seq)
        sev = to_severity(pred, e["y_mu"], e["y_sd"])
        rows.append((e["title"], e["desc"], pred, sev, level_from_score(sev)))

    fl = heuristic_flicker_v2(raw, clip_len=CLIP_LEN)
    rows.append(("Brightness flicker",
                 "Periodic luminance pumping (heuristic, no weights)",
                 None, fl["score"], fl["level"]))
    dt = time.time() - t0

    md = ["| Measurement | Prediction | Severity | What it detects |",
          "|---|---|---|---|"]
    for title, desc, pred, sev, lvl in rows:
        val = f"{pred:.1f}" if pred is not None else "-"
        md.append(f"| **{title}** | {val} | **{lvl}** ({sev:.2f}) | {desc} |")
    md += ["",
           f"Flicker detail: luma pumping {fl['detail']['luma_pump']:.2f} | "
           f"periodicity {fl['detail']['periodicity']:.2f} | "
           f"frame diff {fl['detail']['frame_diff_mean']:.2f}",
           "",
           f"{n_total} frames in file | sampled {N_CLIPS}x{CLIP_LEN} "
           f"consecutive | {dt:.2f}s",
           "",
           "> Prediction is the raw axis output (higher = better). "
           "Severity is 0..1 distortion strength (higher = worse)."]

    gallery = [raw[i] for i in range(raw.shape[0])] if show_frames else None
    return "\n".join(md), gallery, make_plot(seq, raw)


def batch(folder):
    import gradio as gr
    if not folder or not os.path.isdir(folder):
        return "Not a directory.", None
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    if not files:
        return f"No videos in {folder}", None

    t0, rows = time.time(), []
    for name in gr.Progress().tqdm(files):
        stem = os.path.splitext(name)[0]
        try:
            raw, x, _ = read_frames(os.path.join(folder, name))
            seq = extract(x)
            vals = [head_score(e, seq) for e in STATE["heads"]]
            fl = heuristic_flicker_v2(raw, clip_len=CLIP_LEN)["score"]
            rows.append((stem, vals, fl))
        except Exception:
            rows.append((stem, [float("nan")] * len(STATE["heads"]), float("nan")))
    el = time.time() - t0

    out = os.path.join(tempfile.gettempdir(), "diagnosis.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("video," + ",".join(e["title"].split()[0]
                                     for e in STATE["heads"]) + ",flicker\n")
        for stem, vals, fl in rows:
            fh.write(f"{stem}," + ",".join(f"{v:.2f}" for v in vals)
                     + f",{fl:.3f}\n")

    est = el / len(files) * 100 / 60
    txt = (f"Scored **{len(files)}** videos in **{el/60:.2f} min** "
           f"({el/len(files):.2f} s/video)\n\n"
           f"- extrapolated to 100 videos: {est:.2f} min\n"
           f"- time penalty per the assignment formula: "
           f"{min(1.0, 0.01*max(0.0, est-20)):.3f}")
    return txt, out


def build_ui(runs_dir):
    import gradio as gr
    choices = scan(runs_dir)

    with gr.Blocks(title="VQA diagnosis") as demo:
        gr.Markdown(
            "# Video quality diagnosis\n"
            "One score answers one question. This panel runs several "
            "measurements on the same clip, because they detect different "
            "things.\n\n"
            "Flicker uses a heuristic rather than a trained head: MaxWell has "
            "no flicker axis, so nothing can be trained for it. On an "
            "injected-flicker benchmark (n=600) the heuristic scores 0.8554 "
            "against 0.5965 for the T-5 head - see `docs/flicker_detector.md`.")

        with gr.Accordion("Models", open=True):
            gr.Markdown(
                "Pick a checkpoint per measurement, or clear one to switch it "
                "off. Every head reads the same cached features, so extra "
                "measurements cost almost nothing.")
            dropdowns = []
            with gr.Row():
                for title, sub, _, desc in MEASURES:
                    opts = choices.get(title, [])
                    labels = [os.path.basename(p) for p in opts]
                    dd = gr.Dropdown(
                        choices=list(zip(labels, opts)) if opts else [],
                        value=opts[0] if opts else None,
                        label=f"{title}  (runs/{sub}/)",
                        info=desc, interactive=True)
                    dropdowns.append(dd)
            with gr.Row():
                load_btn = gr.Button("Load selected", variant="primary")
                rescan_btn = gr.Button("Rescan runs/")
            info = gr.Textbox(label="Loaded", lines=6, interactive=False,
                              value=apply_selection(
                                  *[c[0] if c else None
                                    for c in (choices.get(t[0], [])
                                              for t in MEASURES)]))
            load_btn.click(apply_selection, dropdowns, [info])

            def rescan():
                ch = scan(runs_dir)
                ups = []
                for title, _, _, _ in MEASURES:
                    opts = ch.get(title, [])
                    labels = [os.path.basename(p) for p in opts]
                    ups.append(gr.update(
                        choices=list(zip(labels, opts)) if opts else [],
                        value=opts[0] if opts else None))
                return ups + [f"Rescanned {runs_dir}/ - press Load selected."]
            rescan_btn.click(rescan, None, dropdowns + [info])

        with gr.Tab("Single video"):
            with gr.Row():
                with gr.Column():
                    vid = gr.Video(label="Video")
                    show = gr.Checkbox(True, label="show sampled frames")
                    btn = gr.Button("Analyse", variant="primary")
                with gr.Column():
                    out = gr.Markdown()
            plot = gr.Plot(label="Temporal signals")
            gal = gr.Gallery(label="Sampled frames", columns=8, height=200)
            btn.click(analyse, [vid, show], [out, gal, plot])

        with gr.Tab("Batch"):
            gr.Markdown("Run every measurement over a folder and export a CSV.")
            folder = gr.Textbox(label="Folder with videos",
                                placeholder="/home/peter/trkv/data")
            bbtn = gr.Button("Run", variant="primary")
            bout = gr.Markdown()
            bfile = gr.File(label="diagnosis.csv")
            bbtn.click(batch, [folder], [bout, bfile])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_ui(args.runs).launch(server_name="0.0.0.0", server_port=args.port,
                               share=args.share)


if __name__ == "__main__":
    main()
