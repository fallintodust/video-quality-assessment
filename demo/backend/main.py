# demo/backend/main.py
"""NR-VQA 演示后端：多模型选择打分 + 失真问题反馈 + 任务书格式 score.txt。"""
import json
import os
import sys
import shutil
import tempfile
import threading
import time
import uuid
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

# ---- 视频暂存区（重新打分用）：上传打分成功后保留一份副本，返回 video_id ----
VIDEO_TTL_SECONDS = 30 * 60    # 无使用自动过期时间
VIDEO_STORE_MAX = 60           # 最多保留条数，超出逐出最旧
VIDEO_STORE_GRACE = 60         # 逐出时对 60 秒内新建条目让路（可能正在推理）
_video_dir = Path(tempfile.gettempdir()) / "nr_vqa_rescore"
_video_store = {}              # video_id -> {"path","name","expires","created"}


def _purge_expired_videos():
    """清理过期条目并删除文件（须持有 _lock）。"""
    now = time.time()
    expired = [vid for vid, info in _video_store.items() if info["expires"] <= now]
    for vid in expired:
        info = _video_store.pop(vid, None)
        if info and os.path.exists(info["path"]):
            os.unlink(info["path"])


def _retain_video(tmp_path: str, name: str) -> str:
    """打分成功后把临时视频移入暂存目录；返回 video_id，失败返回 ''（不影响打分）。"""
    try:
        _video_dir.mkdir(parents=True, exist_ok=True)
        with _lock:
            _purge_expired_videos()
            now = time.time()
            # 超出上限时逐出最旧条目（给新暂存让出空间，跳过宽限期内的）
            for vid, info in sorted(_video_store.items(),
                                    key=lambda kv: kv[1]["created"]):
                if len(_video_store) < VIDEO_STORE_MAX:
                    break
                if info["created"] > now - VIDEO_STORE_GRACE:
                    break
                _video_store.pop(vid, None)
                if os.path.exists(info["path"]):
                    os.unlink(info["path"])
            vid = uuid.uuid4().hex[:12]
            dest = _video_dir / f"{vid}{Path(tmp_path).suffix}"
            shutil.move(tmp_path, dest)
            _video_store[vid] = {
                "path": str(dest), "name": name,
                "expires": time.time() + VIDEO_TTL_SECONDS,
                "created": time.time(),
            }
        return vid
    except Exception:
        return ""


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


def _predict_with_errors(model_id: str, video_path: str, video_name: str) -> dict:
    """打分 + 失真问题反馈 + 记录 history，返回统一响应体；模型/推理错误转 HTTPException。

    与 /api/predict、/api/predict/{video_id} 共用，保证两个入口响应结构一致。
    """
    try:
        predictor = model_registry.get_model(model_id)
        score = predictor.predict(video_path)

        # 失真问题反馈（闪烁/噪点/模糊，vqa.diagnosis 已注册的检测器）
        from vqa.diagnosis import diagnose_video
        diag = diagnose_video(video_path, model=None)

        with _lock:
            n = len(_score_history) + 1
            _score_history.append((f"video{n}", model_id, score))

        return {
            "status": "success",
            "video_name": video_name,
            "score": round(score, 4),
            "model_id": model_id,
            "model_name": next(m["name"] for m in model_registry.MODELS
                               if m["id"] == model_id),
            "scale": next(m["scale"] for m in model_registry.MODELS
                          if m["id"] == model_id),
            "score_txt_line": f"video{n}: {round((score - 1.0) * 25.0, 2)}",
            "issues": diag["issues"],
        }
    except KeyError as e:
        raise HTTPException(404, detail=f"模型不存在: {model_id}")
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"推理失败: {e}")


@app.post("/api/predict")
def predict_video(model_id: str, file: UploadFile = File(...)):
    """指定模型打分 + 失真问题反馈（同步 def → FastAPI 线程池，不阻塞事件循环）。

    打分成功后把视频暂存到服务端并返回 video_id，前端“重新打分”可免重传复用。
    """
    tmp_path = _save_tmp(file)
    try:
        result = _predict_with_errors(model_id, tmp_path, file.filename)
        result["video_id"] = _retain_video(tmp_path, file.filename)
    finally:
        # _retain_video 成功后已把文件移走；失败/异常时删除残留临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return result


@app.post("/api/predict/{video_id}")
def rescore_video(video_id: str, model_id: str):
    """重新打分：复用服务端暂存视频，无需重新上传（响应字段与 /api/predict 一致）。"""
    with _lock:
        info = _video_store.get(video_id)
        if info is None:
            raise HTTPException(404, detail="暂存视频不存在或已过期，请重新上传后再打分")
        info["expires"] = time.time() + VIDEO_TTL_SECONDS   # 使用中顺延过期时间
    result = _predict_with_errors(model_id, info["path"], info["name"])
    result["video_id"] = video_id
    return result


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

# ---------------------------------------------------------------- 多维诊断
# 一个分数只能回答一个问题。以下三个接口同时给出四项回归测量
# （整体质量 / 抖动 / 卡顿 / 纯时域）加启发式闪烁检测。
# 四个回归头共用同一次特征提取，额外维度几乎不增加耗时。

from . import multiaxis   # noqa: E402


@app.get("/api/diagnose/axes")
def diagnose_axes():
    """四个测量维度及各自可选的权重变体（供前端渲染选择器）。

    抖动维度提供 4 个变体（r50 / 双分支 / 仅 ViT / 仅 mean），
    用于现场演示分支与时间聚合的消融结果。
    """
    return {"status": "success", "axes": multiaxis.available(),
            "extras": multiaxis.extra_detectors()}


def _parse_variants(raw: str):
    """variants 参数：JSON 形式的 {轴 id: 权重文件名}，留空则各轴用默认权重。"""
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return {k: v for k, v in d.items() if v}
    except Exception:
        raise HTTPException(400, detail="variants 需为 JSON 对象")


def _parse_extras(raw: str):
    """extras 参数：逗号分隔的检测器名称，如 "噪点,模糊"。"""
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else []


def _record_score(video_name: str, model_id: str, score: float) -> str:
    """写入 score.txt 历史；同一文件用不同模型评估会各留一条记录，互不覆盖。

    任务书量纲：score.txt 为 0~100（模型输出 1~5 → 线性转 0~100，排序不变）。
    """
    s100 = (score - 1.0) * 25.0
    with _lock:
        n = len(_score_history) + 1
        _score_history.append((f"video{n}", model_id, s100))
    return f"video{n}: {round(s100, 2)}"


@app.post("/api/diagnose")
def diagnose_upload(file: UploadFile = File(...),
                    variants: str = None,
                    with_visuals: bool = False,
                    extras: str = None):
    """多维诊断：上传视频 -> 四项回归测量 + 闪烁检测。

    variants：JSON，如 {"shake": "best_all_mean.pt"}，留空各轴用默认权重
    with_visuals：额外返回抽取的帧与两张图（base64），约 400 KB，批量时建议关闭
    """
    tmp_path = _save_tmp(file)
    keep = False
    try:
        result = multiaxis.analyse(tmp_path, _parse_variants(variants),
                                   with_visuals, _parse_extras(extras))
        # analyse 看到的是临时文件名，前端要显示用户上传时的原名
        result["video"] = file.filename
        ov = result["measurements"].get("overall")
        if ov:
            result["score_txt_line"] = _record_score(
                file.filename, "multiaxis", ov["prediction"])
        result["status"] = "success"
        result["video_id"] = _retain_video(tmp_path, file.filename)
        keep = bool(result["video_id"])
        return result
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"权重缺失：{e}")
    except Exception as e:
        raise HTTPException(500, detail=f"诊断失败：{e}")
    finally:
        # _retain_video 成功时已移走文件；失败或异常时清理残留
        if not keep and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/diagnose/{video_id}")
def diagnose_again(video_id: str, variants: str = None,
                   with_visuals: bool = False, extras: str = None):
    """换权重重新诊断：复用服务端暂存视频，无需重新上传。"""
    v = _parse_variants(variants)
    with _lock:
        info = _video_store.get(video_id)
        if info is None:
            raise HTTPException(404, detail="暂存视频不存在或已过期，请重新上传")
        info["expires"] = time.time() + VIDEO_TTL_SECONDS
    try:
        result = multiaxis.analyse(info["path"], v, with_visuals,
                                   _parse_extras(extras))
        result["video"] = info["name"]
        ov = result["measurements"].get("overall")
        if ov:
            result["score_txt_line"] = _record_score(
                info["name"], "multiaxis", ov["prediction"])
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"权重缺失：{e}")
    except Exception as e:
        raise HTTPException(500, detail=f"诊断失败：{e}")
    result["status"] = "success"
    result["video_id"] = video_id
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo.backend.main:app",
                host=DemoConfig.API_HOST, port=DemoConfig.API_PORT, reload=False)
