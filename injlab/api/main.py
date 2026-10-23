"""FastAPI 服务：把检测器和实验管理暴露成 HTTP 接口。

跑起来：
    pip install fastapi uvicorn
    python -m injlab serve            # 另开终端
    open http://127.0.0.1:8000/docs   # 交互式文档，答辩演示很好用

注意：这里的 /detect 返回**逐块分数**（chunks），
Streamlit 页面靠它做"命中片段高亮"，这是安全分析员真正需要的东西。
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        f"缺少依赖: {e.name}\n请先执行: pip install fastapi uvicorn pydantic"
    ) from e

from ..chunking import chunk_text, max_chunk_score
from ..config import Config
from ..db.store import Store
from ..models.stub import KeywordDetector

app = FastAPI(title="injlab", version="0.1.0",
              description="Prompt-injection detector robustness lab")
STATE: dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    cfg = Config()
    det = KeywordDetector().fit([], [], fpr_target=0.05)
    STATE["cfg"] = cfg
    STATE["detector"] = det
    STATE["store"] = Store(cfg.db_path)
    STATE["store"].init()


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    chunk_size: int = Field(256, ge=16, le=2048)
    chunk_stride: int = Field(128, ge=1, le=2048)


class DetectResponse(BaseModel):
    score: float
    pred: str
    max_chunk_score: float
    chunks: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "detector": STATE["detector"].name}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    det = STATE["detector"]
    whole = float(det.predict_proba([req.text])[0])
    chunks = [
        {"start": s, "end": e, "score": float(det.predict_proba([c])[0])}
        for s, e, c in chunk_text(req.text, req.chunk_size, req.chunk_stride)
    ]
    top = max((c["score"] for c in chunks), default=0.0)
    score = max(whole, top)
    return DetectResponse(
        score=score,
        pred="malicious" if score > req.threshold else "benign",
        max_chunk_score=top,
        chunks=chunks,
    )


@app.get("/audit/verify")
def audit_verify() -> dict[str, Any]:
    ok, msg = STATE["store"].verify_chain()
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "detail": msg}


@app.get("/stats")
def stats() -> dict[str, int]:
    return STATE["store"].counts()
