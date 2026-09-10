#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""答辩演示一键启动：检查权重 → 启动后端 → 自动打开浏览器。

双击 start_defense.bat 调用本脚本（勿直接 python 运行，路径由 bat 定位）。
"""
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent.parent
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"

# (说明, 路径, 缺失严重度: 0=不挡启动)
WEIGHTS = [
    ("自研模型(半监督v2)", ROOT / "runs" / "divide_semisup_v2" / "model_best.pt", 0),
    ("DOVER 零样本", ROOT / "dover_repro" / "pretrained_weights" / "DOVER.pth", 0),
    ("DOVER++ 微调", ROOT / "dover_repro" / "pretrained_weights" /
     "DOVER_head_train-dividemaxwell_0_val-dividemaxwell_s_latest.pth", 0),
    ("四维诊断", ROOT / "runs" / "o" / "best_all_mean+std+diff.pt", 0),
    ("FAST-VQA 零样本", ROOT / "pretrained_weights" / "FAST_VQA_B_1_4.pth", 0),
]


def port_in_use():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def wait_health(timeout=120):
    """轮询 /api/health，就绪返回 True。首次懒加载不触发模型加载，后端起得快。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL + "/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    print("=" * 52)
    print("  VQA 视频质量评估 - 答辩演示一键启动")
    print("=" * 52)

    # 1. 权重检查
    print("\n[1/4] 权重检查：")
    for name, path, _ in WEIGHTS:
        print(f"    [{'OK' if path.exists() else '缺'}] {name}")

    # 2. 端口检查
    print("\n[2/4] 端口检查：")
    if port_in_use():
        print(f"    端口 {PORT} 已有服务在运行，直接打开浏览器")
        webbrowser.open(URL)
        return 0

    # 3. 启动后端
    print("\n[3/4] 启动后端服务...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "demo.backend.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT),
    )

    # 4. 等就绪 → 自动打开浏览器
    print(f"[4/4] 等待服务就绪（最多 120 秒）...")
    if not wait_health():
        print(f"\n[错误] 后端 {120} 秒未就绪。排查：")
        print("   1. 上方后端日志是否有报错（依赖缺失/权重损坏）")
        print("   2. 确认 8000 端口未被其他程序占用")
        proc.terminate()
        return 1

    print(f"\n[OK] 后端就绪，打开浏览器 {URL}")
    webbrowser.open(URL)
    print("\n演示结束请回到本窗口按 Ctrl+C 停止服务")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
