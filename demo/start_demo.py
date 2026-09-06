#!/usr/bin/env python
# demo/start_demo.py
import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 50)
    print("  NR-VQA 视频质量评估 - Demo启动")
    print("=" * 50)
    print()
    
    # 切换到项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 检查Python
    print("[1/3] 检查Python环境...")
    try:
        subprocess.run(["python", "--version"], check=True, capture_output=True)
    except:
        print("❌ 未找到Python，请安装Python 3.8+")
        sys.exit(1)
    
    # 检查模型
    print("[2/3] 检查模型文件...")
    model_path = Path("runs/baseline/model_123.pt")
    if not model_path.exists():
        print(f"⚠️ 警告: 未找到模型文件 {model_path}")
        print("    请将训练好的模型重命名为 model_123.pt 放入该目录")
        print("    或修改 demo/backend/config.py 中的模型路径")
        print()
        input("    按回车键继续（使用随机权重演示）...")
    
    # 启动服务
    print("[3/3] 启动服务...")
    print()
    print("📡 服务地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/api/docs")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 50)
    print()
    
    try:
        subprocess.run([
            "python", "-m", "uvicorn",
            "demo.backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n服务已停止")

if __name__ == "__main__":
    main()