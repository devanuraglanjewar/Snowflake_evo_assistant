# llm_utils.py
import os
import requests
import streamlit as st

def _DEF(k, d=None):
    try:
        return st.secrets.get(k, os.getenv(k, d))
    except Exception:
        return os.getenv(k, d)

LLM_PROVIDER = _DEF("LLM_PROVIDER", "ollama")  # "ollama" or "remote"
LLM_ENDPOINT = _DEF("LLM_ENDPOINT", "")
LLM_API_KEY = _DEF("LLM_API_KEY", "")

def chat_llm(prompt: str) -> str:
    """
    Send prompt to LLM (local Ollama or remote Hugging Face API).
    """
    if LLM_PROVIDER == "ollama":
        import ollama
        resp = ollama.chat(model=_DEF("OLLAMA_MODEL", "llama3.1:8b-instruct"),
                           messages=[{"role": "user", "content": prompt}])
        return resp["message"]["content"]

    elif LLM_PROVIDER == "remote":
        if not LLM_ENDPOINT or not LLM_API_KEY:
            return "❌ Hugging Face endpoint or API key not set."

        headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
        payload = {"inputs": prompt}

        try:
            resp = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                return f"❌ Hugging Face API error {resp.status_code}: {resp.text}"

            data = resp.json()
            # Some models return list of dicts
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            return str(data)

        except Exception as e:
            return f"❌ Error calling Hugging Face API: {e}"

    else:
        return "❌ Invalid LLM_PROVIDER. Use 'ollama' or 'remote'."
