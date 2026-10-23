"""FastAPI 服务：把检测器和实验管理暴露成 HTTP 接口。

跑起来（三种方式任选）：
    python -m injlab serve            # 推荐：命令行服务
    python injlab/api/main.py         # 直接跑也行（VS Code 右上角三角就是这种）
    open http://127.0.0.1:8000/docs   # 交互式文档，答辩演示很好用

注意：这里的 /detect 返回**逐块分数**（chunks），
Streamlit 页面靠它做"命中片段高亮"，这是安全分析员真正需要的东西。
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# 允许 `python injlab/api/main.py` 这种直接运行方式：
# 直接把文件当脚本跑时，import 路径里没有项目根，这里补上。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        f"缺少依赖: {e.name}\n"
        f"当前解释器: {sys.executable}\n"
        f"请先激活虚拟环境并安装: source .venv/bin/activate && "
        f"pip install -r requirements.txt"
    ) from e

from injlab.chunking import chunk_text
from injlab.config import Config
from injlab.db.store import Store
from injlab.models.stub import KeywordDetector

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    """启动时装配检测器与数据库，关闭时释放连接（FastAPI 0.93+ 推荐写法）。"""
    cfg = Config()
    det = KeywordDetector().fit([], [], fpr_target=0.05)
    STATE["cfg"] = cfg
    STATE["detector"] = det
    STATE["store"] = Store(cfg.db_path)
    STATE["store"].init()
    yield
    STATE["store"].close()


app = FastAPI(title="injlab", version="0.1.0",
              description="Prompt-injection detector robustness lab",
              lifespan=lifespan)


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


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
