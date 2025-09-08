"""
llm_utils.py
Handles local Ollama chat or remote HTTP model endpoint.
"""

import os
import json
import requests

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Helper to read secrets (Streamlit secrets prioritized)
def _DEF(k, d=None):
    if _HAS_ST:
        return st.secrets.get(k, os.getenv(k, d))
    return os.getenv(k, d)

LLM_PROVIDER = _DEF("LLM_PROVIDER", "ollama")  # "ollama" or "remote"
OLLAMA_HOST = _DEF("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _DEF("OLLAMA_MODEL", "llama3.1:8b")
LLM_ENDPOINT = _DEF("LLM_ENDPOINT", "")
LLM_API_KEY = _DEF("LLM_API_KEY", "")

SYSTEM_INSTRUCTIONS = (
    "You are a Snowflake schema evolution assistant. Be concise, precise, and include runnable SQL when asked."
)


def _chat_ollama(messages):
    """
    Chat via local Ollama HTTP API (fallback). Expects messages list like OpenAI chat.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Handle Ollama response variants
    if isinstance(data, dict) and "message" in data:
        return data["message"].get("content", "")
    # support list style
    if isinstance(data, list):
        return "".join([chunk.get("message", {}).get("content", "") for chunk in data])
    return ""


def _chat_remote(messages):
    """
    Chat via a remote HTTP endpoint (simple JSON contract).
    POST { messages: [...], stream: false } -> { text: "..." }
    """
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {"messages": messages, "stream": False}
    r = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "text" in data:
            return data["text"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "")
    return ""


def chat_llm(user_prompt: str, system_instructions: str = SYSTEM_INSTRUCTIONS) -> str:
    """
    Simple wrapper: system + user messages, returns plain text reply.
    """
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_prompt},
    ]
    if LLM_PROVIDER == "remote" and LLM_ENDPOINT:
        return _chat_remote(messages)
    # default local Ollama
    return _chat_ollama(messages)
