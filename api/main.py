from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os

app = FastAPI(
    title="ZenX AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "ZenX AI",
        "message": "ZenX AI API is running"
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "model": MODEL_NAME
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )

    try:
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=request.message.strip()
        )

        return {
            "success": True,
            "response": interaction.output_text,
            "model": MODEL_NAME
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini error: {str(e)}"
        )
