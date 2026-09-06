from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured"
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash"
            ),
            contents=request.message
        )

        return {
            "success": True,
            "response": response.text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
