# demo/backend/main.py
"""NR-VQA 演示后端：多模型选择打分 + 失真问题反馈 + 任务书格式 score.txt。"""
import os
import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from demo.backend.config import DemoConfig
from demo.backend import model_registry

app = FastAPI(
    title="NR-VQA 视频质量评估系统（多模型）",
    description="自研任务书架构 / DOVER / DOVER++ 多模型打分 + 失真诊断",
    version="2.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

frontend_dir = DemoConfig.FRONTEND_DIR
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")

# ---- 打分历史（score.txt，任务书格式）----
_lock = threading.Lock()
_score_history = []   # [(video_name, model_id, score), ...]


@app.get("/")
async def serve_index():
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/health")
async def health():
    infos = model_registry.model_info_list()
    return {
        "status": "healthy",
        "models": infos,
        "device": os.environ.get("DEMO_DEVICE", "cuda"),
    }


@app.get("/api/models")
async def list_models():
    return {"status": "success", "models": model_registry.model_info_list()}


def _save_tmp(file: UploadFile) -> str:
    if not (file.content_type.startswith("video/")
            or Path(file.filename).suffix.lower() in DemoConfig.SUPPORTED_VIDEO_EXTENSIONS):
        raise HTTPException(400, detail=f"不支持的文件类型: {file.content_type}")
    suffix = Path(file.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    return tmp.name


@app.post("/api/predict")
async def predict_video(model_id: str, file: UploadFile = File(...)):
    """指定模型打分 + 失真问题反馈。"""
    tmp_path = _save_tmp(file)
    try:
        predictor = model_registry.get_model(model_id)
        score = predictor.predict(tmp_path)

        # 失真问题反馈（闪烁/噪点/模糊，vqa.diagnosis 已注册的检测器）
        from vqa.diagnosis import diagnose_video
        diag = diagnose_video(tmp_path, model=None)

        with _lock:
            n = len(_score_history) + 1
            _score_history.append((f"video{n}", model_id, score))

        return {
            "status": "success",
            "video_name": file.filename,
            "score": round(score, 4),
            "model_id": model_id,
            "model_name": next(m["name"] for m in model_registry.MODELS
                               if m["id"] == model_id),
            "scale": next(m["scale"] for m in model_registry.MODELS
                          if m["id"] == model_id),
            "score_txt_line": f"video{n}: {round(score, 2)}",
            "issues": diag["issues"],
        }
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(404, detail=f"模型不存在: {model_id}")
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"推理失败: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/score/txt")
async def score_txt(model_id: str):
    """任务书格式 score.txt：videoN: 分数（按评估顺序）。"""
    lines = [f"{name}: {round(score, 2)}" for name, mid, score in _score_history
             if mid == model_id]
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""))


@app.post("/api/score/reset")
async def score_reset():
    with _lock:
        _score_history.clear()
    return {"status": "success", "message": "已清空 score.txt 记录"}


@app.get("/api/score/history")
async def score_history():
    with _lock:
        return {"status": "success", "history": [
            {"video": n, "model_id": m, "score": s}
            for n, m, s in _score_history]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo.backend.main:app",
                host=DemoConfig.API_HOST, port=DemoConfig.API_PORT, reload=False)
