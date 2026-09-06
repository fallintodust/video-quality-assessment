#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio web UI for the video quality assessment model.

Two tabs:
  Single video - upload a clip, get a score, see the sampled frames and a
                 per-frame instability curve (what the model reacts to)
  Batch        - point at a folder, get score.txt plus the timing needed for
                 the assignment's scoring formula

Usage:
    python3 scripts/app.py
    python3 scripts/app.py --ckpt runs/t5/best_all_mean.pt --share
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
from extract_feats import Extractor, clip_indices, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from train_head import Head, aggregate, CONV_DIM  # noqa: E402

STATE = {}          # loaded model + settings, filled by load_checkpoint()


# ---------------------------------------------------------------- model

def find_checkpoints(root="runs"):
    return sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))


def load_checkpoint(path):
    """(Re)load a head checkpoint and, if needed, the backbones."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ta = ck["args"]
    branch = ta["branch"]
    use_vit = branch != "r50"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # the extractor is expensive to build, so only rebuild it when the
    # branch configuration actually changes
    if STATE.get("use_vit") != use_vit or "extractor" not in STATE:
        STATE["extractor"] = Extractor(use_vit=use_vit).to(dev)
        STATE["use_vit"] = use_vit

    head = Head(len(ck["mu"])).to(dev)
    head.load_state_dict(ck["state"])
    head.eval()

    STATE.update(
        head=head, dev=dev, branch=branch,
        modes=ta["agg"].split("+"), clip_len=ta["clip_len"],
        n_clips=ta.get("clips", 4), size=ta.get("size", 224),
        mu=ck["mu"], sd=ck["sd"], y_mu=ck["y_mu"], y_sd=ck["y_sd"],
        path=path,
    )
    return (f"loaded: {os.path.basename(path)}\n"
            f"aggregation {ta['agg']} | branch {branch} | "
            f"{STATE['n_clips']} clips x {STATE['clip_len']} frames | {dev}")


def read_frames(path):
    from decord import VideoReader, cpu
    s = STATE["size"]
    vr = VideoReader(path, ctx=cpu(0), width=s, height=s, num_threads=1)
    n_total = len(vr)
    idx = clip_indices(n_total, STATE["n_clips"], STATE["clip_len"])
    raw = vr.get_batch(idx).asnumpy()                       # [T,H,W,3] uint8
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
    seq = torch.cat(outs).cpu().numpy().astype(np.float32)
    if STATE["branch"] == "r50":
        seq = seq[:, :CONV_DIM]
    elif STATE["branch"] == "vit":
        seq = seq[:, CONV_DIM:]
    return seq


def score_from_seq(seq):
    v = aggregate(seq, STATE["modes"], STATE["clip_len"])
    v = (v - STATE["mu"]) / STATE["sd"]
    with torch.no_grad():
        p = STATE["head"](torch.from_numpy(v[None]).float().to(STATE["dev"])).item()
    return p * STATE["y_sd"] + STATE["y_mu"]


def instability_plot(seq):
    """Mean absolute change between adjacent frames, per clip.

    This is the raw signal the 'diff' aggregation is built on: tall bars mean
    the content changes sharply from one frame to the next.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = STATE["clip_len"]
    K = seq.shape[0] // L
    d = seq.reshape(K, L, -1)
    per_step = np.abs(np.diff(d, axis=1)).mean(-1)          # [K, L-1]

    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=110)
    for k in range(K):
        ax.plot(range(1, L), per_step[k], marker="o", ms=3, label=f"clip {k+1}")
    ax.set_xlabel("frame step within clip")
    ax.set_ylabel("mean |change|")
    ax.set_title("Frame-to-frame instability")
    ax.legend(fontsize=7, ncol=K)
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- handlers

def run_single(video, show_frames):
    if not video:
        return "upload a video first", None, None
    if "head" not in STATE:
        return "no checkpoint loaded", None, None
    t0 = time.time()
    try:
        raw, x, n_total = read_frames(video)
    except Exception as e:
        return f"could not read the video: {e}", None, None
    seq = extract(x)
    score = score_from_seq(seq)
    dt = time.time() - t0

    txt = (f"## score: {score:.2f}\n\n"
           f"- frames in the file: {n_total}\n"
           f"- sampled: {STATE['n_clips']} clips x {STATE['clip_len']} frames\n"
           f"- elapsed: {dt:.2f} s\n"
           f"- checkpoint: {os.path.basename(STATE['path'])}")
    gallery = [raw[i] for i in range(raw.shape[0])] if show_frames else None
    return txt, gallery, instability_plot(seq)


def run_batch(folder, progress=None):
    import gradio as gr
    if not folder or not os.path.isdir(folder):
        return "not a folder", None
    if "head" not in STATE:
        return "no checkpoint loaded", None
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    if not files:
        return f"no videos in {folder}", None

    t0, rows, failed = time.time(), [], 0
    it = gr.Progress().tqdm(files) if progress is not False else files
    for name in it:
        stem = os.path.splitext(name)[0]
        try:
            _, x, _ = read_frames(os.path.join(folder, name))
            rows.append((stem, score_from_seq(extract(x))))
        except Exception:
            rows.append((stem, STATE["y_mu"]))
            failed += 1
    elapsed = time.time() - t0

    out = os.path.join(tempfile.gettempdir(), "score.txt")
    with open(out, "w", encoding="utf-8") as fh:
        for n, s in rows:
            fh.write(f"{n}: {s:.1f}\n")

    mins = elapsed / 60
    est100 = elapsed / len(files) * 100 / 60
    txt = (f"scored **{len(files)}** videos in **{mins:.2f} min** "
           f"({elapsed/len(files):.2f} s/video)\n\n"
           f"- failed: {failed}\n"
           f"- extrapolated to 100 videos: {est100:.2f} min\n"
           f"- time penalty at that rate: "
           f"{min(1.0, 0.01*max(0.0, est100-20)):.3f}")
    return txt, out


# ---------------------------------------------------------------- ui

def build_ui(ckpts, initial):
    import gradio as gr
    with gr.Blocks(title="VQA - temporal consistency") as demo:
        gr.Markdown("# Video Quality Assessment\n"
                    "Temporal consistency scoring (course project, group 3)")

        with gr.Row():
            ck = gr.Dropdown(ckpts, value=initial, label="checkpoint", scale=3)
            load_btn = gr.Button("load", scale=1)
        info = gr.Textbox(label="model", lines=3, interactive=False)
        load_btn.click(load_checkpoint, [ck], [info])

        with gr.Tab("Single video"):
            with gr.Row():
                with gr.Column():
                    vid = gr.Video(label="video")
                    show = gr.Checkbox(True, label="show sampled frames")
                    btn = gr.Button("score", variant="primary")
                with gr.Column():
                    out = gr.Markdown()
            plot = gr.Plot(label="frame-to-frame instability")
            gal = gr.Gallery(label="sampled frames", columns=8, height=200)
            btn.click(run_single, [vid, show], [out, gal, plot])

        with gr.Tab("Batch"):
            gr.Markdown("Score every video in a folder and write `score.txt`.")
            folder = gr.Textbox(label="folder with videos",
                                placeholder="/home/peter/trkv/data")
            bbtn = gr.Button("run", variant="primary")
            bout = gr.Markdown()
            bfile = gr.File(label="score.txt")
            bbtn.click(run_batch, [folder], [bout, bfile])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="checkpoint to preload")
    ap.add_argument("--runs", default="runs", help="where to look for *.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio link")
    args = ap.parse_args()

    ckpts = find_checkpoints(args.runs)
    if args.ckpt and args.ckpt not in ckpts:
        ckpts.insert(0, args.ckpt)
    if not ckpts:
        raise SystemExit(f"no *.pt found under {args.runs}/ - train a head first")

    initial = args.ckpt or ckpts[0]
    print(load_checkpoint(initial))

    build_ui(ckpts, initial).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
