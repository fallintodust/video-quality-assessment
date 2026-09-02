# demo/backend/main.py
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
from typing import List

from demo.backend.config import DemoConfig
from demo.backend.inference import VQAInference
from demo.backend.schemas import PredictResponse, HealthResponse, BatchPredictResponse

# 创建FastAPI应用
app = FastAPI(
    title="NR-VQA 视频质量评估系统",
    description="基于深度学习的无参考视频质量评估 - 课程设计项目",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
frontend_dir = DemoConfig.FRONTEND_DIR
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")

# 全局模型实例
model = None
config = DemoConfig()

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model
    
    # 尝试多个路径加载模型
    for model_path in config.MODEL_PATHS:
        if model_path.exists():
            try:
                model = VQAInference(model_path)
                return
            except Exception as e:
                print(f"⚠️ 加载模型失败 ({model_path}): {e}")
    
    print("❌ 未找到任何可用的模型文件")
    print("   请将训练好的模型放在以下位置之一:")
    for path in config.MODEL_PATHS:
        print(f"   - {path}")

@app.get("/")
async def serve_index():
    """首页"""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"error": "前端页面不存在，请检查 frontend/index.html"}
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_name": model.model_name if model else None,
        "device": model.device.type if model else None
    }

@app.post("/api/predict", response_model=PredictResponse)
async def predict_video(
    file: UploadFile = File(..., description="上传的视频文件")
):
    """视频质量评估"""
    if not model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 检查文件类型
    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件类型: {file.content_type}，请上传视频文件"
        )
    
    # 检查文件大小
    if file.size and file.size > DemoConfig.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，最大支持 {DemoConfig.MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )
    
    # 保存临时文件
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 执行推理
        result = model.predict_video(tmp_path)
        
        if result['status'] == 'error':
            return JSONResponse(
                status_code=500,
                content={
                    'status': 'error',
                    'message': result.get('error', '推理失败')
                }
            )
        
        return {
            'status': 'success',
            'mos_score': result['mos_score'],
            'video_name': result['video_name'],
            'num_frames': result.get('num_frames'),
            'model_name': result.get('model_name')
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': str(e)}
        )
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/api/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """批量视频评估"""
    if not model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    results = []
    for file in files:
        # 单个预测
        result = await predict_video(file)
        results.append(result)
    
    # 计算统计信息
    valid_results = [r for r in results if r.get('mos_score') is not None]
    if valid_results:
        scores = [r['mos_score'] for r in valid_results]
        stats = {
            'count': len(valid_results),
            'average': round(sum(scores) / len(scores), 4),
            'max': round(max(scores), 4),
            'min': round(min(scores), 4)
        }
    else:
        stats = None
    
    return {
        'status': 'success',
        'results': results,
        'statistics': stats
    }

@app.get("/api/model/info")
async def get_model_info():
    """获取模型信息"""
    if not model:
        return JSONResponse(
            status_code=503,
            content={'status': 'error', 'message': '模型未加载'}
        )
    
    return {
        'status': 'success',
        'model_name': model.model_name,
        'model_path': str(model.model_path),
        'device': model.device.type,
        'is_loaded': True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "demo.backend.main:app",
        host=DemoConfig.API_HOST,
        port=DemoConfig.API_PORT,
        reload=True
    )