"""CAMP-VQA 批量评估（对照实验用）：零样本 LSVQ 预训练模型对任意视频目录打分。

与 camp-vqa_demo.py 同一推理链，但做了两处适配：
  1. 8GB 显存管理：各骨干顺序加载/释放（SlowFast -> Swin-Large -> CLIP+BLIP2），
     BLIP-2 用 fp16 加载（原 demo 直接 fp32 在 8GB 上会 OOM）
  2. 逐视频循环输出全部分数，可选与标注对比计算 SROCC/PLCC

用法（在本目录 src/ 下运行）：
    python campvqa_eval_batch.py \
        --videos-dir D:/videoquality/data/divide/videos \
        --labels D:/videoquality/data/divide/train_lable_test.txt \
        --max-n 100 --out D:/videoquality/runs/campvqa_val_scores.txt
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from extractor.extract_frag import VideoDataset_feature
from extractor.extract_clip_embeds import extract_features_clip_embed
from extractor.extract_slowfast_clip import SlowFast, extract_features_slowfast_pool
from extractor.extract_swint_clip import SwinT, extract_features_swint_pool
from model_finetune import fix_state_dict
from model_regression_lsvq import Mlp, preprocess_data


def get_transform(resize):
    return transforms.Compose([transforms.Resize([resize, resize]),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                    std=[0.225, 0.225, 0.225])])


def get_video_metadata(video_path):
    cmd = (f'ffprobe -v error -select_streams v:0 -show_entries '
           f'stream=width,height,nb_frames,r_frame_rate,bit_rate,'
           f'bits_per_raw_sample,pix_fmt -of json "{video_path}"')
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, check=True)
        info = json.loads(result.stdout)
    except Exception as e:
        print(f"ffprobe 失败 {video_path}: {e}", file=sys.stderr)
        return 1920, 1080, 0, 8, 30.0
    s = info["streams"][0]
    width = int(s["width"])
    height = int(s["height"])
    bitrate = int(s.get("bit_rate") or 0)
    bitdepth = int(s.get("bits_per_raw_sample") or 8)
    num, den = (s.get("r_frame_rate") or "30/1").split("/")
    framerate = float(num) / float(den)
    return width, height, bitrate, bitdepth, framerate


def main():
    p = argparse.ArgumentParser(description="CAMP-VQA 批量零样本评估")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--labels", default="", help="标注文件（可选，格式 name: score）")
    p.add_argument("--max-n", type=int, default=0, help="最多评估的视频数（0=全部）")
    p.add_argument("--out", default="campvqa_scores.txt")
    p.add_argument("--model-path", default="../model/lsvq_train_camp-vqa_Mlp_byrmse_trained_model_kfold.pth",
                   help="LSVQ 预训练 MLP 权重（零样本）")
    p.add_argument("--prompt-path", default="./config/prompts.json")
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--target-size", type=int, default=224)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 收集视频列表与元数据 ----
    all_names = sorted(n for n in os.listdir(args.videos_dir)
                       if n.lower().endswith((".mp4", ".avi", ".mkv", ".mov")))
    if args.max_n:
        all_names = all_names[: args.max_n]
    print(f"评估视频数: {len(all_names)}")

    rows = []
    for n in tqdm(all_names, desc="ffprobe 元数据"):
        w, h, br, bd, fr = get_video_metadata(os.path.join(args.videos_dir, n))
        rows.append({"vid": os.path.splitext(n)[0], "test_video_path": n,
                     "width": w, "height": h, "bitrate": br, "bitdepth": bd,
                     "framerate": fr, "prediction_mode": 50.0})
    test_df = pd.DataFrame(rows)

    prompts = json.load(open(args.prompt_path, encoding="utf-8"))
    resize_transform = get_transform(args.resize)
    top_n = int(args.target_size / args.patch_size) ** 2

    dataset = VideoDataset_feature(test_df, args.videos_dir, "test",
                                   resize_transform, args.resize,
                                   args.patch_size, args.target_size, top_n)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=False, num_workers=0)

    # ---- 骨干加载（BLIP-2 fp16 常驻 GPU；SlowFast/Swin 按需在 GPU/CPU 间切换） ----
    model_slowfast = SlowFast().to(device).eval()
    model_swint = SwinT(model_name="swin_large_patch4_window7_224",
                        global_pool="avg", pretrained=True).to(device).eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    blip_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", use_fast=True)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16).to(device).eval()
    model_mlp = Mlp(input_features=13056, out_features=1,
                    drop_rate=0.1, act_layer=torch.nn.GELU).to(device).eval()
    state = torch.load(args.model_path, map_location=device)
    model_mlp.load_state_dict(fix_state_dict(state))
    print("backbones + MLP 加载完成")

    # 分数集合（Swin/SlowFast/CLIP 特征），逐视频处理
    scores = {}
    with torch.no_grad():
        for i, (video_segments, video_res_frag_all, video_frag_all,
                video_name, frames_info, metadata) in enumerate(
                tqdm(data_loader, desc="CAMP-VQA 评估")):
            vid = str(test_df.iloc[i]["vid"])

            # SlowFast 特征
            _, _, sf = extract_features_slowfast_pool(video_segments, model_slowfast, device)
            _, _, sf_res = extract_features_slowfast_pool(video_res_frag_all, model_slowfast, device)
            _, _, sf_frag = extract_features_slowfast_pool(video_frag_all, model_slowfast, device)
            sf_feats = torch.cat((sf.mean(0), sf_res.mean(0), sf_frag.mean(0)), 0)

            # Swin-Large 特征
            sw = extract_features_swint_pool(video_segments, model_swint, device)
            sw_res = extract_features_swint_pool(video_res_frag_all, model_swint, device)
            sw_frag = extract_features_swint_pool(video_frag_all, model_swint, device)
            sw_feats = torch.cat((sw.mean(0), sw_res.mean(0), sw_frag.mean(0)), 0)

            # 腾显存给 BLIP-2
            model_slowfast.to("cpu")
            model_swint.to("cpu")
            torch.cuda.empty_cache()

            # CLIP + BLIP2 语义特征
            image_emb, quality_emb, artifact_emb = extract_features_clip_embed(
                frames_info, metadata, clip_model, clip_preprocess,
                blip_processor, blip_model, prompts, device)
            clip_feats = torch.cat((image_emb.mean(0), quality_emb.mean(0),
                                    artifact_emb.mean(0)), 0)

            vqa_feats = torch.cat((sf_feats, sw_feats, clip_feats), 0)
            feat_tensor, _ = preprocess_data(vqa_feats, None)
            feat_tensor = feat_tensor.unsqueeze(0) if feat_tensor.dim() == 1 else feat_tensor

            with torch.amp.autocast(device_type="cuda"):
                scores[vid] = float(model_mlp(feat_tensor).item())

            # 恢复
            model_slowfast.to(device)
            model_swint.to(device)

    # ---- 输出 ----
    with open(args.out, "w", encoding="utf-8") as f:
        for vid in sorted(scores):
            f.write(f"{vid}: {scores[vid]:.4f}\n")
    print(f"分数已写入: {args.out}（{len(scores)} 条）")

    if args.labels:
        from scipy.stats import pearsonr, spearmanr
        labels = {}
        for line in open(args.labels, encoding="utf-8"):
            parts = line.strip().rsplit(":", 1)
            if len(parts) == 2:
                labels[parts[0].strip().replace(".mp4", "")] = float(parts[1])
        common = [v for v in scores if v in labels]
        yt = [labels[v] for v in common]
        yp = [scores[v] for v in common]
        if len(common) >= 8:
            s = spearmanr(yt, yp).statistic
            pl = pearsonr(yt, yp).statistic
            print(f"对比标注（n={len(common)}）: SROCC={s:.4f}  PLCC={pl:.4f}  OBJ={s+pl:.4f}")


if __name__ == "__main__":
    main()
