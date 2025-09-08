"""
Minimal FastAPI app to run a small open-source model on Hugging Face Spaces or any server.
Deploy this to a HF Space (SDK: Python) if you want a remote LLM endpoint for Streamlit Cloud.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Choose a very small model suitable for HF Spaces CPU
MODEL_NAME = "TheBloke/tiny-stablelm-3b-1"  # example; pick a small CPU-friendly model

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False

@app.post("/run")
def run(req: ChatRequest):
    sys = "\n".join([m.content for m in req.messages if m.role == "system"]) or ""
    usr = "\n".join([m.content for m in req.messages if m.role == "user"]) or ""
    prompt = (sys + "\n\n" + usr).strip()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = text[len(prompt):].strip()
    return {"text": completion}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
