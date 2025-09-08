"""
chatbot.py
RAG-lite + direct LLM Q&A. If docs exist in ./docs, embed them and attach top results as context.
"""

import os
import glob
import json
from typing import Optional, List
from llm_utils import chat_llm
from embedding_utils import embed_texts, top_k_similar
import numpy as np

_DOC_TEXTS: List[str] = []
_DOC_EMB = None

def _load_docs():
    global _DOC_TEXTS, _DOC_EMB
    if _DOC_TEXTS:
        return
    paths = sorted(list(glob.glob("docs/*.md")) + list(glob.glob("docs/*.txt")))
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                _DOC_TEXTS.append(f.read())
        except Exception:
            pass
    if _DOC_TEXTS:
        _DOC_EMB = embed_texts(_DOC_TEXTS)

_SYSTEM = "You answer questions about Snowflake schema evolution. Prefer any provided context."

def answer_question(question: str, extra_context: Optional[str] = None) -> str:
    _load_docs()
    context = extra_context or ""
    try:
        if _DOC_TEXTS and _DOC_EMB is not None:
            sims = top_k_similar(question, _DOC_TEXTS, _DOC_EMB, k=2)
            ctxs = [f"[Doc {i}]\\n{_DOC_TEXTS[i][:1200]}" for (i, _s) in sims]
            context = (context or "") + "\n\n" + "\n\n".join(ctxs)
    except Exception:
        # if embeddings fail, continue without doc context
        pass

    prompt = f"""
Context (may be empty):
{context}

Question: {question}

Answer clearly and concisely. When relevant, include Snowflake SQL.
"""
    return chat_llm(prompt, system_instructions=_SYSTEM)
