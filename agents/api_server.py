# agents/api_server.py
from typing import Any
import math

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .coordinator import process_event


# ────────── FastAPI plumbing ──────────
app    = FastAPI(title="Crisis-311")
router = APIRouter()


# ────────── request schema ──────────
class EventReq(BaseModel):
    text:       str = Field(..., examples=["tweet_content"])
    radar_path: str = Field(..., examples=["/path/to/tor/file.nc"])


# ────────── helper: scrub NaN / ±Inf ──────────
def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ────────── endpoint ──────────
@router.post("/event", summary="Run full pipeline and return merged report")
async def handle(ev: EventReq):
    """
    Invokes the LangGraph pipeline and returns a dict with keys  
    `radar`, `tweet`, `resource`, and `summary`.
    """
    raw_report = process_event(ev.text, ev.radar_path)
    return JSONResponse(content=_sanitize(raw_report))


# **register the router *after* all routes are defined**
app.include_router(router)


# ────────── local dev entry-point ──────────
if __name__ == "__main__":                       # → python -m agents.api_server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
