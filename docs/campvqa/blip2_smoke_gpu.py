"""BLIP-2 GPU 冒烟测试（v2 训练结束后 GPU 空出时运行）。

目标：
  1. 测 BLIP-2 flan-t5-xl fp16 加载后的显存占用（8GB 卡是否装得下全链路）
  2. 测单次 caption 生成耗时（估算 909 视频全量评估的总时长）
  3. 若 fp16 超显存，给出可退路：bitsandbytes 8-bit / device_map offload

用法（src/ 下）：
    python blip2_smoke_gpu.py [--num-gens 6] [--load-backbones]
"""

import argparse
import json
import time

import torch

torch.cuda.reset_peak_memory_stats()


def mem():
    return torch.cuda.memory_allocated() / 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-gens", type=int, default=6,
                   help="生成次数（测平均耗时）")
    p.add_argument("--load-backbones", action="store_true",
                   help="同时加载 SlowFast/Swin/CLIP，测联合显存")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"设备: {device} 总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import clip
    from PIL import Image
    import numpy as np

    print("[1] 加载 BLIP-2 fp16 ...")
    t0 = time.time()
    blip_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", use_fast=True)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16).to(device).eval()
    print(f"    BLIP-2 加载 {time.time()-t0:.1f}s，显存占用 {mem():.2f}GB")

    if args.load_backbones:
        print("[2] 额外加载 SlowFast/Swin/CLIP ...")
        from extractor.extract_slowfast_clip import SlowFast
        from extractor.extract_swint_clip import SwinT
        model_slowfast = SlowFast().to(device).eval()
        model_swint = SwinT(model_name="swin_large_patch4_window7_224",
                            global_pool="avg", pretrained=True).to(device).eval()
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        print(f"    联合显存占用 {mem():.2f}GB（8GB 卡此为峰值）")
        # 模拟批量脚本的错峰策略
        model_slowfast.to("cpu"); model_swint.to("cpu"); torch.cuda.empty_cache()
        print(f"    SF/Swin 移回 CPU 后 {mem():.2f}GB（BLIP-2+CLIP 常驻）")

    print(f"[3] 生成 {args.num_gens} 次 caption（随机图，测耗时）...")
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    prompt = ("Quality: fair. Resolution: SD. Bitrate: high. Bit depth: 8. "
              "Frame rate: standard.\n\nRate the quality of this video frame.")
    times = []
    with torch.no_grad():
        for i in range(args.num_gens):
            inputs = blip_processor(img, prompt, return_tensors="pt").to(device)
            t1 = time.time()
            out = blip_model.generate(**inputs, max_new_tokens=50)
            torch.cuda.synchronize()
            dt = time.time() - t1
            times.append(dt)
            cap = blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"    gen {i+1}: {dt:.2f}s -> {cap[:60]}")
    avg = sum(times) / len(times)
    print(f"平均单次生成: {avg:.2f}s | 显存 {mem():.2f}GB | 峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
    per_video = avg * 3 * 20  # 每视频约 20 帧 x 3 次生成（官方默认抽帧步长）
    print(f"估算: 仅 BLIP-2 部分 {per_video:.0f}s/视频；909 视频需 {per_video*909/3600:.1f} 小时")


if __name__ == "__main__":
    main()
