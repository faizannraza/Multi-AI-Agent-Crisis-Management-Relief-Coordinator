"""
agents/rag_agent.py
-------------------
Light-weight RAG helper that surfaces FEMA-guideline passages.

* Embeds queries with all-MiniLM-L6-v2 (fast, 30 MB in RAM)
* Performs similarity search over your FAISS index stored in
  `project_root/models/fema_kb/`
* Exposes the retrieval step as a LangChain-style `BaseTool`
  called **FemaTool** so the rest of the crew can invoke it.
"""

from pathlib import Path
from typing import List

from langchain.tools import BaseTool                # crew-friendly Tool base
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS


# ---------------------------------------------------------------------------
# 1) Embedding model  (≈ 30 MB, multilingual, GPL-free)
# ---------------------------------------------------------------------------
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="models/.hf"        # optional local cache
)

# ---------------------------------------------------------------------------
# 2) Load the FAISS index that lives under  models/fema_kb/
#    `allow_dangerous_deserialization=True` is OK here because you created
#    the pickle yourself; it is not coming from an un-trusted party.
# ---------------------------------------------------------------------------
KB_DIR      = Path("models/fema_kb")          # directory
INDEX_NAME  = "fema_index"                    # stem (no suffix)

try:
    _db: FAISS = FAISS.load_local(
        KB_DIR,
        _embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
except Exception as e:                        # legacy pickle?  -> rebuild
    import json, warnings, faiss
    warnings.warn(f"Re-creating FAISS KB because: {e!s}")
    with open(KB_DIR / "fema_chunks.json") as fh:
        chunks = json.load(fh)
    texts   = [c["text"] if isinstance(c, dict) else c for c in chunks]
    _db     = FAISS.from_texts(texts, _embeddings)
    _db.save_local(KB_DIR, INDEX_NAME)

# ---------------------------------------------------------------------------
# 3) Expose as a Tool
# ---------------------------------------------------------------------------
class FemaTool(BaseTool):
    """
    Answer disaster-relief questions with authoritative FEMA guidance.

    **Input** : plain-English question  
    **Output**: list[str] – the top-3 most relevant guideline passages
    """
    name: str = "fema_rag"
    description: str = (
        "Use this tool whenever you need official FEMA guidance. "
        "Provide a question about disaster response or relief and you will "
        "receive up to three relevant text snippets."
    )

    # sync
    def _run(self, query: str) -> List[str]:
        hits = _db.similarity_search(query, k=3)
        return [doc.page_content for doc in hits]

    # async (CrewAI / LangGraph can call either)
    async def _arun(self, query: str) -> List[str]:
        return self._run(query)
