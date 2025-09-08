"""
embedding_utils.py
Uses sentence-transformers to produce normalized embeddings and simple retrieval helpers.
"""

import os
from typing import List, Tuple
import numpy as np

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from sentence_transformers import SentenceTransformer

def _DEF(k, d=None):
    if _HAS_ST:
        return st.secrets.get(k, os.getenv(k, d))
    return os.getenv(k, d)

_EMB_MODEL_NAME = _DEF("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMB_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = _load_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(emb)

def cosine_similarity_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # query_vec: (d,), doc_vecs: (N, d)
    return np.dot(doc_vecs, query_vec)

def top_k_similar(query: str, corpus_texts: List[str], corpus_vecs: np.ndarray, k: int = 3):
    model = _load_model()
    qv = model.encode([query], normalize_embeddings=True)[0]
    sims = cosine_similarity_matrix(qv, corpus_vecs)
    idx = np.argsort(sims)[-k:][::-1]
    return [(int(i), float(sims[i])) for i in idx]
