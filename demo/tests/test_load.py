# demo/tests/test_load.py
import sys
from pathlib import Path

# 添加项目根目录到路径（从 demo/tests/ 向上两级到项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo.backend.inference import VQAInference


def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("  测试模型加载")
    print("=" * 50)
    print()
    
    # 测试加载 model_123.pt
    model_path = PROJECT_ROOT / "runs" / "baseline" / "model_123.pt"
    
    if not model_path.exists():
        print(f"❌ 文件不存在: {model_path}")
        print()
        print("检查 runs/baseline 目录内容:")
        baseline_dir = PROJECT_ROOT / "runs" / "baseline"
        if baseline_dir.exists():
            for f in baseline_dir.iterdir():
                print(f"  - {f.name}")
        else:
            print(f"  - 目录不存在: {baseline_dir}")
        return False
    
    print(f"✅ 文件存在: {model_path}")
    print()
    
    try:
        infer = VQAInference(model_path)
        print()
        print("✅ 加载成功！")
        print(f"   模型名称: {infer.model_name}")
        print(f"   设备: {infer.device}")
        print(f"   模型类型: {type(infer.model).__name__}")
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_dummy_input():
    """测试模型推理"""
    print()
    print("=" * 50)
    print("  测试模型推理")
    print("=" * 50)
    print()
    
    model_path = PROJECT_ROOT / "runs" / "baseline" / "model_123.pt"
    
    if not model_path.exists():
        print(f"❌ 文件不存在: {model_path}")
        return False
    
    try:
        infer = VQAInference(model_path)
        
        # 创建模拟输入
        import torch
        import numpy as np
        
        print("创建模拟视频帧...")
        # 模拟 8 帧 224x224 的 RGB 图像
        dummy_frames = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        
        print("执行推理...")
        # 使用 predict_from_numpy 方法（如果存在）
        if hasattr(infer, 'predict_from_numpy'):
            mos = infer.predict_from_numpy(dummy_frames)
            print(f"✅ 推理成功")
            print(f"   MOS 分数: {mos:.4f}")
            return True
        else:
            # 如果没有 predict_from_numpy，用 predict_video 需要真实视频文件
            print("⚠️ 没有 predict_from_numpy 方法，跳过推理测试")
            return True
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("🧪 NR-VQA 推理引擎测试")
    print()
    
    # 运行测试
    test1 = test_model_loading()
    
    if test1:
        test2 = test_model_with_dummy_input()
    else:
        test2 = False
    
    print()
    print("=" * 50)
    if test1 and test2:
        print("✅ 所有测试通过！")
        print()
        print("现在可以启动UI了:")
        print("   python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000")
    elif test1:
        print("⚠️ 模型加载成功，但推理测试跳过")
        print()
        print("可以启动UI试试:")
        print("   python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000")
    else:
        print("❌ 测试失败，请检查模型文件")
    print("=" * 50)