# agents/radar_agent.py
from __future__ import annotations

import math, torch, xarray as xr, numpy as np
from pathlib import Path
from typing import ClassVar

from pydantic          import BaseModel, Field, ConfigDict
from langchain.tools   import BaseTool

from .utils            import load_radar

_RADAR, _NORM = load_radar()

_VARS = [
    "DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH",
    "range_folded_mask"          # kept for completeness
]


# ─────────────────────────────────  pydantic args
class _RadarArgs(BaseModel):
    model_config = ConfigDict(extra="allow")        # allow extra keys
    file_path: str = Field(
        ...,
        description="Absolute path to a mini-NEXRAD *.nc file",
    )


# ─────────────────────────────────  pre-processing
def _prep(ncfile: Path) -> torch.Tensor:
    """
    Load <ncfile> and build a tensor shaped (1, 14, 120, 240).

    * 7 variables × 2 elevation sweeps → 14 channels
    * NaN / ±Inf → 0
    * Values are divided by Normalisation constants **without** collapsing
      everything to ~0 so the net can generate a non-zero probability.
    """
    ds   = xr.open_dataset(ncfile)
    arrs = []

    for var in _VARS:
        a = ds[var].values.astype("float32")           # (T, 120, 240, 2)
        if var != "range_folded_mask":
            a /= float(_NORM.get(var, 1.0))            # robust: default 1.0
        arrs.append(a)

    ds.close()

    vol = np.stack(arrs, axis=0)                       # (7, T, 120, 240, 2)
    vol = vol[:, -1]                                   # latest timestep
    vol = np.transpose(vol, (3, 0, 1, 2))              # (2, 7, 120, 240)
    vol = vol.reshape(14, 120, 240)                    # 14 channels

    # replace NaN/Inf → 0, but **NO hard clipping to 0** anymore
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(vol).unsqueeze(0)          # (1, 14, 120, 240)


# ─────────────────────────────────  LangChain tool
class RadarTool(BaseTool):
    name: str = "radar_detector"
    description: str = (
        "Loads a mini-NEXRAD *.nc volume, runs TinyTorCNN, "
        "and returns the tornado probability."
    )

    args_schema: ClassVar[type[_RadarArgs]] = _RadarArgs   # explicit type

    # sync interface used by LangGraph
    def _run(self, file_path: str, **__) -> dict:
        try:
            x = _prep(Path(file_path))
            with torch.no_grad():
                prob_tensor = _RADAR(x)
                # Probabilities are already sigmoid-ed in the net;
                # fallback to sigmoid here if they are logits
                if prob_tensor.max() > 1.0:
                    prob_tensor = torch.sigmoid(prob_tensor)
                prob = float(prob_tensor.squeeze().max().clamp(0.0, 1.0))
                if not math.isfinite(prob):
                    prob = 0.0
        except Exception as exc:                        # final safety-net
            print(f"[radar] ERROR: {exc}")
            prob = 0.0

        return {"file": str(file_path), "tornado_prob": prob}

    async def _arun(self, *a, **k):
        raise NotImplementedError("RadarTool supports only synchronous calls.")
