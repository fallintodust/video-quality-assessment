# demo/tests/test_inference.py
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from demo.backend.inference import VQAInference

def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("测试1: 模型加载")
    print("=" * 50)
    
    model_path = Path("runs/baseline/model_123.pt")
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("   请先训练模型或创建演示模型")
        return False
    
    try:
        model = VQAInference(model_path)
        print("✅ 模型加载成功")
        print(f"   设备: {model.device}")
        print(f"   模型名称: {model.model_name}")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_inference():
    """测试推理"""
    print("\n" + "=" * 50)
    print("测试2: 模型推理")
    print("=" * 50)
    
    model_path = Path("runs/baseline/model_123.pt")
    
    if not model_path.exists():
        print("❌ 模型文件不存在，跳过测试")
        return False
    
    try:
        model = VQAInference(model_path)
        
        # 创建模拟视频帧
        fake_frames = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        mos = model.predict_from_numpy(fake_frames)
        
        print(f"✅ 推理测试成功")
        print(f"   模拟MOS分数: {mos:.4f}")
        print(f"   分数范围: 1.0 - 5.0")
        return True
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

if __name__ == "__main__":
    print("\n🧪 NR-VQA 推理测试")
    print()
    
    test1 = test_model_loading()
    test2 = test_inference()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("✅ 所有测试通过！")
        print("\n可以启动UI了:")
        print("   python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000")
    else:
        print("❌ 部分测试失败")
    print("=" * 50)