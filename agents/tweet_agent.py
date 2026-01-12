"""
agents/tweet_agent.py
─────────────────────
Return tweet-classification **plus** the *.nc* path so the RadarTool
receives the field it needs.
"""
from typing import ClassVar
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
import torch, torch.nn.functional as F

from .utils import TOKENIZER, load_tweet_bin, load_tweet_multi

_BIN  = load_tweet_bin()
_MLT  = load_tweet_multi()
_MLT_ID2LBL = {0: "damage", 1: "other", 2: "request", 3: "warning"}


class _TweetArgs(BaseModel):
    model_config = ConfigDict(extra="allow")      # accept file_path, radar_path…
    text: str = Field(..., description="Tweet text to classify")


class TweetTool(BaseTool):
    name: str = "tweet_classifier"
    description: str = (
        "Classifies a tweet for tornado relevance. "
        "Outputs binary on/off probability and 4-class probs "
        "for damage / request / warning / other."
    )
    args_schema: ClassVar[type[BaseModel]] = _TweetArgs

    # ------------------------------------------------------------------
    def _run(self, text: str, **extra) -> dict:
        # grab the .nc path (may be given as file_path or radar_path)
        file_path: str | None = (
            extra.get("file_path") or extra.get("radar_path")
        )

        tok = TOKENIZER(text, return_tensors="pt",
                        truncation=True, max_length=128)
        with torch.no_grad():
            p_bin = F.softmax(_BIN(**tok).logits, -1)[0]
            p_mlt = F.softmax(_MLT(**tok).logits, -1)[0]

        out = {
            "binary": {"on": float(p_bin[1]), "off": float(p_bin[0])},
            "multi":  {_MLT_ID2LBL[i]: float(p_mlt[i]) for i in range(4)},
            "raw":    text,
        }
        # include the radar file so the next tool can validate cleanly
        if file_path:
            out["file_path"] = file_path
        return out

    async def _arun(self, *a, **k):
        raise NotImplementedError
