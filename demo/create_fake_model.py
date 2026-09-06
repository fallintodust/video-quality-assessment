# create_clean_model.py
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vqa.models import VQAModel

def create_clean_model():
    """创建一个干净的 state_dict 模型"""
    
    # 创建输出目录
    output_dir = Path("runs/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建模型
    print("构建模型结构...")
    model = VQAModel(T=8)
    model.eval()
    
    # 保存为 state_dict（不包含自定义类）
    model_path = output_dir / "model_123.pt"
    torch.save(model.state_dict(), model_path)
    
    print(f"✅ 干净模型已创建: {model_path}")
    print(f"   模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"   格式: state_dict (可被任何 VQAModel 加载)")
    
    # 测试加载
    print("\n测试加载...")
    test_model = VQAModel(T=8)
    test_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    test_model.eval()
    print("✅ 模型加载测试通过")
    
    # 测试前向传播
    print("\n测试前向传播...")
    test_input = torch.randn(1, 8, 3, 224, 224)
    with torch.no_grad():
        output = test_model(test_input)
        print(f"✅ 前向传播测试通过")
        print(f"   MOS值: {output.item():.4f}")
    
    return model_path

if __name__ == "__main__":
    print("="*50)
    print("  创建干净模型 (state_dict)")
    print("="*50)
    print()
    
    try:
        model_path = create_clean_model()
        print()
        print("="*50)
        print("🎉 模型创建成功！")
        print()
        print("现在可以启动UI了:")
        print("   python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000")
        print("="*50)
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()