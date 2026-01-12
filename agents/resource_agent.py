# agents/resource_agent.py
from __future__ import annotations

import json, re
from typing import Any

from langchain.prompts         import ChatPromptTemplate
from langchain_core.messages   import AIMessage
from langchain_core.runnables  import RunnableLambda

from ._llm import llm


# ───────────────────────────────────────── prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            #  ↓↓↓ escape { } so they’re NOT interpreted as template vars
            "You are the Crisis-Resource Coordinator.\n"
            "Given radar output, tweet classifications and optional FEMA "
            "passages, decide concrete actions in **JSON**.\n be as detailed as you possibly can in your output"
            "Required schema →  {{alert: bool, actions: list[str]}}",
        ),
        ("human", "{context}"),
    ]
)


# ───────────────────────────────────────── helpers
def format_ctx(
    radar: dict[str, Any] | None = None,
    tweet: dict[str, Any] | None = None,
    fema: str | list[str] | None = None,
    **extras,
) -> str:
    radar_line = (
        f"Radar tornado_prob: {radar['tornado_prob']:.3f}"
        if radar and "tornado_prob" in radar
        else "Radar tornado_prob: n/a"
    )

    tweet_lines = (
        f"Tweet binary : {tweet['binary']}\n"
        f"Tweet multi  : {tweet['multi']}"
        if tweet
        else "Tweet: n/a"
    )

    fema_brief = (fema[:2] if isinstance(fema, list) else fema) or "none"

    return f"{radar_line}\n{tweet_lines}\nFEMA docs    : {fema_brief}"


def _as_json(msg: AIMessage) -> dict:
    txt = msg.content.strip()

    # 1) strict
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) first {...}
    m = re.search(r"\{.*?\}", txt, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) safest fallback
    return {
        "alert": False,
        "actions": [txt[:200] + ("…" if len(txt) > 200 else "")]
    }


# ───────────────────────────────────────── chain
ResourceChain = (
    RunnableLambda(lambda x: {"context": format_ctx(**x)})  # wrap for prompt
    | prompt
    | llm
    | RunnableLambda(_as_json)
)
