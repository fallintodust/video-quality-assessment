# demo/backend/config.py
from pathlib import Path

class DemoConfig:
    """演示配置"""
    
    # 模型路径（优先查找顺序）
    MODEL_PATHS = [
        Path("runs/baseline/model_123.pt"),      # 你指定的命名
        Path("runs/baseline/model_best.pt"),      # 原训练脚本命名
        Path("runs/baseline/model.pt"),           # 通用命名
        Path("runs/model_123.pt"),                # 备用位置
    ]
    
    # 推理参数
    NUM_FRAMES = 8          # 抽帧数（可根据需要调整）
    BATCH_SIZE = 1          # 推理批次大小
    
    # API配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # 前端配置
    FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
    MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
    
    # 视频格式支持
    SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]