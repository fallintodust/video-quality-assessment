# demo/backend/inference.py
import sys
from pathlib import Path
from typing import Dict, Optional, Union

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import cv2
from tqdm import tqdm

# 导入模型类（用于加载 state_dict）
from vqa.models import VQAModel
from vqa.sampling import read_video_frames
from vqa.train_utils import get_device


class VQAInference:
    """视频质量评估推理引擎"""
    
    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None):
        self.device = device or get_device()
        self.model_path = Path(model_path)
        self.model = None
        self.model_name = self.model_path.stem
        
        self._load_model()
        
    def _load_model(self):
        """加载模型权重 - 支持多种格式"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        try:
            print(f"📥 加载模型: {self.model_path}")
            
            # 尝试加载
            loaded = torch.load(self.model_path, map_location=self.device)
            
            # 判断加载的是什么类型
            if isinstance(loaded, dict):
                # 这是 state_dict，需要先创建模型再加载权重
                print("   检测到 state_dict，创建模型实例...")
                self.model = VQAModel(T=8)
                self.model.load_state_dict(loaded)
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✅ 模型加载成功 (设备: {self.device})")
                
            elif hasattr(loaded, 'eval'):
                # 这是完整的模型对象
                self.model = loaded
                self.model.eval()
                print(f"✅ 模型加载成功 (设备: {self.device})")
                
            else:
                raise RuntimeError(f"无法识别的模型格式: {type(loaded)}")
            
            # 打印模型信息
            params = sum(p.numel() for p in self.model.parameters())
            print(f"   参数数量: {params/1e6:.2f}M")
            
        except Exception as e:
            # 如果加载失败，尝试另一种方式
            if "Can't get attribute" in str(e):
                print(f"⚠️ 完整模型加载失败，尝试作为 state_dict 加载...")
                try:
                    # 重新加载为 state_dict
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    if isinstance(state_dict, dict):
                        self.model = VQAModel(T=8)
                        self.model.load_state_dict(state_dict)
                        self.model = self.model.to(self.device)
                        self.model.eval()
                        print(f"✅ 模型作为 state_dict 加载成功")
                        params = sum(p.numel() for p in self.model.parameters())
                        print(f"   参数数量: {params/1e6:.2f}M")
                        return
                except Exception as e2:
                    raise RuntimeError(f"模型加载失败: {e2}")
            raise RuntimeError(f"模型加载失败: {e}")
    
    def predict_video(self, video_path: Union[str, Path]) -> Dict:
        """预测单个视频的MOS分数"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            return {
                'status': 'error',
                'error': f'文件不存在: {video_path}'
            }
        
        try:
            # 1. 使用 read_video_frames 提取帧并预处理
            frames_tensor = read_video_frames(
                str(video_path), 
                T=8,
                size=224
            )
            
            # 2. 添加 batch 维度: [T, 3, H, W] -> [1, T, 3, H, W]
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            
            # 3. 推理
            with torch.no_grad():
                output = self.model(frames_tensor)
                mos = output.item()
            
            # 4. 分数映射到1-5
            mos_score = max(1.0, min(5.0, mos))
            
            return {
                'status': 'success',
                'mos_score': round(mos_score, 4),
                'video_name': video_path.name,
                'num_frames': 8,
                'model_name': self.model_name
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'video_name': video_path.name
            }
    
    def predict_batch(self, video_paths: list) -> list:
        """批量预测"""
        results = []
        for path in tqdm(video_paths, desc="评估视频"):
            result = self.predict_video(path)
            results.append(result)
        return results