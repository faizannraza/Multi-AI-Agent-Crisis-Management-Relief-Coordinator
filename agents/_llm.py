# agents/_llm.py
import os, logging
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage

def _make_stub(msg: str) -> AIMessage:
    logging.warning("LLM offline – returning stubbed response.")
    return AIMessage(content=msg)

try:
    llm = ChatOllama(
        model="llama3:8b",
        temperature=0.2,
        max_tokens=512,
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )
    # quick connectivity check
    llm.invoke("ping")
except Exception:
    # server missing → replace with a dummy that echoes a default answer
    llm = lambda _: _make_stub(
        "⚠️  LLM unavailable. Generated fallback coordination plan.")
