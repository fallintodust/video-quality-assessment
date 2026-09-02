# -*- coding: utf-8 -*-
"""把视频目录中的所有视频按 TSN 策略预抽帧并缓存到磁盘（uint8 .npy）。

正式数据（4543 个视频，分辨率 320p~4K 不等）逐 epoch 实时解码开销过大，
预抽帧一次后训练/打分的每样本成本从"解码视频"降为"读 1.2MB npy"。

缓存布局：<cache-dir>/<视频名去扩展名>.npy，形状 [T, 224, 224, 3]（uint8）。
训练时通过 --frame-cache 指定同一目录即可自动命中（见 vqa.sampling）。

用法：
    python scripts/precache_frames.py --data-dir data/divide/videos \
        --cache-dir data/frames_cache --t 8 --workers 8
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vqa.dataset import VIDEO_EXTS, list_videos, natural_key


def decode_to_cache(args):
    """单进程任务：解码一个视频并写 npy。返回 (name, ok, err)。"""
    name, video_dir, cache_dir, T, size = args
    out_path = os.path.join(cache_dir, os.path.splitext(name)[0] + ".npy")
    if os.path.isfile(out_path):
        return name, True, None
    try:
        import cv2
        from vqa.sampling import sample_frame_indices

        cap = cv2.VideoCapture(os.path.join(video_dir, name))
        if not cap.isOpened():
            cap.release()
            return name, False, "无法打开"
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # 帧数元数据不可靠：先顺序读一遍计数，再回卷重读
            n = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            total = n
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = sample_frame_indices(total, T) if total > 0 else None
        wanted = set(int(i) for i in idx) if idx is not None else None

        frames = []
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames.append(cv2.resize(frame, (size, size)))
        cap.release()
        if not frames:
            return name, False, "无有效帧"
        while len(frames) < T:
            frames.append(frames[-1])
        x = np.stack(frames[:T])  # [T, size, size, 3] uint8 BGR
        np.save(out_path, x)
        return name, True, None
    except Exception as e:  # noqa: BLE001
        return name, False, str(e)


def main():
    p = argparse.ArgumentParser(description="预抽帧缓存")
    p.add_argument("--data-dir", default="data/divide/videos")
    p.add_argument("--cache-dir", default="data/frames_cache")
    p.add_argument("--t", type=int, default=8)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    names = list_videos(args.data_dir)
    print(f"视频总数: {len(names)}")
    os.makedirs(args.cache_dir, exist_ok=True)
    done = {os.path.splitext(n)[0] for n in os.listdir(args.cache_dir)
            if n.endswith(".npy")}
    todo = [n for n in names if os.path.splitext(n)[0] not in done]
    print(f"已有缓存: {len(done)} | 待处理: {len(todo)}")

    ok = fail = 0
    failed = []
    tasks = [(n, args.data_dir, args.cache_dir, args.t, args.size)
             for n in todo]
    if args.workers > 1 and tasks:
        with Pool(args.workers) as pool:
            for i, (name, s, err) in enumerate(
                    pool.imap_unordered(decode_to_cache, tasks), 1):
                if s:
                    ok += 1
                else:
                    fail += 1
                    failed.append((name, err))
                if i % 200 == 0:
                    print(f"  进度 {i}/{len(tasks)} 成功 {ok} 失败 {fail}")
    else:
        for i, task in enumerate(tasks, 1):
            name, s, err = decode_to_cache(task)
            if s:
                ok += 1
            else:
                fail += 1
                failed.append((name, err))
            if i % 200 == 0:
                print(f"  进度 {i}/{len(tasks)} 成功 {ok} 失败 {fail}")

    print(f"\n完成: 成功 {ok} 失败 {fail}（总计 {len(names)}）")
    if failed:
        with open(os.path.join(args.cache_dir, "failed.txt"), "w",
                  encoding="utf-8") as f:
            for name, err in failed:
                f.write(f"{name}\t{err}\n")
        print(f"失败清单已写 {args.cache_dir}/failed.txt（前 5 条）:")
        for name, err in failed[:5]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
