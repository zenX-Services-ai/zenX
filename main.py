from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="ZenX API", version="1.0.0")
@app.get("/")
def home():
    return {"message": "ZenX API is live 🚀"}
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AskRequest(BaseModel):
    question: str

@app.get("/api/health")
def health():
    return {"status":"ok","service":"ZenX API"}

@app.post("/api/ask")
def ask(body: AskRequest):
    q = body.question.strip()
    if not q:
        return {"error":"Question is required."}
    # Safe local demo response. Replace this block with your chosen AI provider SDK/API.
    return {"answer": f"ZenX demo received your question: “{q}”\n\nFor the production version, connect this endpoint to an AI model provider using a server-side API key. Never put secret API keys in frontend JavaScript."}
