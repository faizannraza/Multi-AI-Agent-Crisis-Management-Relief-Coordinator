# agents/coordinator.py
from __future__ import annotations

from langgraph.graph  import StateGraph
from langgraph.pregel import Pregel

from .tweet_agent      import TweetTool
from .radar_agent      import RadarTool
from .rag_agent        import FemaTool
from .resource_agent   import ResourceChain     # Runnable
from .summarizer_agent import summarize


# ─────────────────────────────────────────  raw tools
_tweet_tool = TweetTool()
_radar_tool = RadarTool()
_fema_tool  = FemaTool()

# ─────────────────────────────────────────  wrapper nodes
def tweet_node(state: dict) -> dict:
    """Run Tweet classifier → add result under `tweet`."""
    out = _tweet_tool.run({"text": state["text"]})
    return {**state, "tweet": out}       #  ← keep text & file_path


def radar_node(state: dict) -> dict:
    """Run TinyTorCNN → add result under `radar`."""
    out = _radar_tool.run({"file_path": state["file_path"]})
    return {**state, "radar": out}       #  ← keep everything for next steps


def fema_node(state: dict) -> dict:
    """Optional RAG over FEMA docs; add under `fema`."""
    out = _fema_tool.run({"text": state["text"]})
    return {**state, "fema": out}


def resource_node(state: dict) -> dict:
    """LLM decides alert/actions from radar+tweet(+fema)."""
    out = ResourceChain.invoke(
        {
            "radar": state["radar"],
            "tweet": state["tweet"],
            "fema":  state.get("fema"),
        }
    )
    return {**state, "resource": out}


def summary_node(state: dict) -> dict:
    """Generate final human-readable summary; add under `summary`."""
    state["summary"] = summarize(state)
    return state


# ─────────────────────────────────────────  FEMA routing helper
def need_fema(state: dict) -> bool:
    multi = state["tweet"]["multi"]
    return multi.get("damage", 0) > 0.4 or multi.get("request", 0) > 0.3


# ─────────────────────────────────────────  graph definition
g = StateGraph(dict)

g.add_node("tweet",    tweet_node)
g.add_node("radar",    radar_node)
g.add_node("fema",     fema_node)
g.add_node("resource", resource_node)
g.add_node("summary",  summary_node)


# entry
g.set_entry_point("tweet")

# mandatory radar step after tweets are classified
g.add_edge("tweet", "radar")

# decide if we need RAG
g.add_conditional_edges(
    "tweet",
    {
        "fema":     need_fema,
        "resource": lambda s: not need_fema(s),
    },
)

# normal forward flow
g.add_edge("radar",    "resource")
g.add_edge("fema",     "resource")
g.add_edge("resource", "summary")

# ─────────────────────────────────────────  merge / finish
def _merge(state: dict) -> dict:
    """Collect final report (every key is guaranteed to exist)."""
    return {
        "radar": state.get("radar", {
            "file": None,
            "tornado_prob": 0.0,
        }),
        "tweet": state.get("tweet", {
            "binary": {"on": 0.0, "off": 1.0},
            "multi":  {"damage": 0.0, "request": 0.0,
                       "warning": 0.0, "other": 1.0},
            "raw":    "",
        }),
        "resource": state.get("resource", {
            "alert":   False,
            "actions": [],
        }),
        "summary": state.get("summary",
            "No summary available."),
    }

g.add_node("merge", _merge)
g.add_edge("summary", "merge")
g.set_finish_point("merge")

pipeline: Pregel = g.compile()


# ─────────────────────────────────────────  external helper
def process_event(text: str, nc_path: str) -> dict:
    """
    Run the pipeline and return the complete, detailed report.
    """
    return pipeline.invoke({
        "text":       text,
        "file_path":  nc_path,
    })
