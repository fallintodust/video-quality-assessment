# demo/backend/schemas.py
from pydantic import BaseModel
from typing import Optional, List

class PredictResponse(BaseModel):
    status: str
    mos_score: Optional[float] = None
    video_name: Optional[str] = None
    num_frames: Optional[int] = None
    model_name: Optional[str] = None
    message: Optional[str] = None

class BatchPredictResponse(BaseModel):
    status: str
    results: List[dict]
    statistics: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None