#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache per-frame features.

One pass over the dataset: decode frames -> run them through a frozen
ResNet-50 and ViT -> save a [T, D] matrix as .npy (float16).

Features are stored PER FRAME, not averaged. Every later experiment with
temporal aggregation then runs on the cache in seconds, with no re-decoding.

Frame selection: K clips of L CONSECUTIVE frames (rather than T frames spread
uniformly across the whole video). Two reasons:
  1) flicker is an inter-frame effect; between frames 17 positions apart the
     difference reflects scene motion, not flicker;
  2) K seek operations instead of T, which is noticeably faster on h264.

Usage:
    python3 extract_feats.py --videos ./data --out ./feats
    python3 extract_feats.py --videos ./data --out ./feats --backbone r50   # no ViT
    python3 extract_feats.py --flops                                        # measure only
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- sampling

def clip_indices(n_frames, n_clips, clip_len):
    """K clips of L consecutive frames, spread uniformly along the timeline."""
    need = n_clips * clip_len
    if n_frames <= need:
        idx = np.arange(n_frames)
        idx = np.pad(idx, (0, need - n_frames), mode="edge")  # pad with last frame
        return idx
    starts = np.linspace(0, n_frames - clip_len, n_clips).round().astype(int)
    return np.concatenate([np.arange(s, s + clip_len) for s in starts])


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, files, videos_dir, out_dir, n_clips, clip_len, size):
        self.files = files
        self.videos_dir = videos_dir
        self.out_dir = out_dir
        self.n_clips, self.clip_len, self.size = n_clips, clip_len, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from decord import VideoReader, cpu
        name = self.files[i]
        stem = os.path.splitext(name)[0]
        path = os.path.join(self.videos_dir, name)
        try:
            # the decoder does the resize itself - much cheaper than pulling the
            # full-resolution frame into memory and scaling it in torch
            vr = VideoReader(path, ctx=cpu(0), width=self.size, height=self.size,
                             num_threads=1)
            idx = clip_indices(len(vr), self.n_clips, self.clip_len)
            frames = vr.get_batch(idx).asnumpy()          # [T, H, W, 3] uint8
        except Exception as e:
            print(f"  !! {name}: {e}")
            return stem, torch.zeros(1)                    # error marker
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.)
        x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return stem, x


# ---------------------------------------------------------------- extractor

def stat_pool(fm):
    """[B,C,H,W] -> [B,3C]: mean / max / std over the spatial dimensions."""
    b, c = fm.shape[:2]
    flat = fm.flatten(2)                                   # [B,C,HW]
    return torch.cat([flat.mean(-1), flat.max(-1).values, flat.std(-1)], dim=1)


class Extractor(nn.Module):
    """ResNet-50 (multi-scale) plus, optionally, ViT (token stats)."""

    def __init__(self, use_vit=True, vit_name="vit_base_patch16_224"):
        super().__init__()
        import torchvision
        r50 = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.layer1, self.layer2 = r50.layer1, r50.layer2
        self.layer3, self.layer4 = r50.layer3, r50.layer4
        self.conv_dim = 3 * (256 + 512 + 1024 + 2048) + 2048   # 13568

        self.vit = None
        self.vit_dim = 0
        if use_vit:
            import timm
            self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)
            d = self.vit.embed_dim
            self.vit_dim = 3 * d
        self.out_dim = self.conv_dim + self.vit_dim
        self.eval()

    def forward(self, x):                                  # x: [B,3,H,W]
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        gap = torch.nn.functional.adaptive_avg_pool2d(c4, 1).flatten(1)
        feats = [stat_pool(c1), stat_pool(c2), stat_pool(c3), stat_pool(c4), gap]
        if self.vit is not None:
            t = self.vit.forward_features(x)               # [B,N,D]
            feats += [t.mean(1), t.max(1).values, t.std(1)]
        return torch.cat(feats, dim=1)


# ---------------------------------------------------------------- FLOPs

def report_flops(use_vit, n_frames):
    try:
        from thop import profile
    except ImportError:
        print("thop is not installed: pip install thop")
        return
    m = Extractor(use_vit=use_vit)
    macs, _ = profile(m, inputs=(torch.randn(1, 3, 224, 224),), verbose=False)
    print(f"\nper frame : {macs/1e9:.2f} GMACs")
    print(f"{n_frames:2d} frames  : {macs*n_frames/1e9:.1f} GMACs   "
          f"(assignment limit is 300G at 20 frames: "
          f"{macs*20/1e9:.1f}G -> {'OK' if macs*20/1e9 < 300 else 'EXCEEDED'})")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="./data")
    ap.add_argument("--out", default="./feats")
    ap.add_argument("--clips", type=int, default=4, help="number of clips K")
    ap.add_argument("--clip-len", type=int, default=8, help="frames per clip L")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--backbone", choices=["r50", "r50+vit"], default="r50+vit")
    ap.add_argument("--chunk", type=int, default=16, help="frames per forward (VRAM)")
    ap.add_argument("--workers", type=int, default=6, help="decoding processes")
    ap.add_argument("--limit", type=int, default=0, help="process only N (smoke test)")
    ap.add_argument("--flops", action="store_true", help="only measure FLOPs")
    args = ap.parse_args()

    use_vit = args.backbone == "r50+vit"
    T = args.clips * args.clip_len

    if args.flops:
        report_flops(use_vit, T)
        return

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.videos) if f.lower().endswith(".mp4"))
    done = {f[:-4] for f in os.listdir(args.out) if f.endswith(".npy")}
    todo = [f for f in files if os.path.splitext(f)[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(files)} total, {len(done)} done, {len(todo)} to process")
    if not todo:
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Extractor(use_vit=use_vit).to(dev)
    print(f"feature dimension: {model.out_dim}")
    print(f"frames per video: {T} ({args.clips} clips x {args.clip_len})")
    est = len(todo) * T * model.out_dim * 2 / 1024**3
    print(f"the cache will take roughly {est:.1f} GB")

    ds = VideoDataset(todo, args.videos, args.out, args.clips, args.clip_len, args.size)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=args.workers, prefetch_factor=2)

    t0 = time.time()
    n_ok = n_err = 0
    with torch.no_grad():
        for i, (stem, x) in enumerate(dl, 1):
            if x.ndim == 1:                                 # error marker
                n_err += 1
                continue
            x = x.to(dev, non_blocking=True)
            outs = []
            for j in range(0, x.shape[0], args.chunk):      # split to fit VRAM
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    outs.append(model(x[j:j + args.chunk]).float())
            feat = torch.cat(outs).cpu().numpy().astype(np.float16)
            np.save(os.path.join(args.out, f"{stem}.npy"), feat)
            n_ok += 1
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {el/i:.2f} s/video, "
                      f"~{(len(todo)-i)*el/i/60:.0f} min left", flush=True)

    print(f"\ndone: {n_ok}, failed: {n_err}, elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
