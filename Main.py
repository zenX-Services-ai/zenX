import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai


# =========================================================
# ZENX AI - FASTAPI BACKEND
# =========================================================

app = FastAPI(
    title="ZenX AI API",
    description="AI backend for ZenX AI",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "ZenX AI",
        "message": "ZenX AI API is running"
    }


@app.get("/api")
async def api_status():
    return {
        "status": "ok",
        "service": "ZenX AI API",
        "gemini_configured": bool(GEMINI_API_KEY)
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY)
    }


# =========================================================
# CHAT ENDPOINT
# =========================================================

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
            detail="GEMINI_API_KEY is not configured on the server."
        )

    try:

        prompt = request.message.strip()

        if request.system_prompt:
            prompt = (
                f"{request.system_prompt.strip()}\n\n"
                f"User:\n{prompt}"
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        text = getattr(response, "text", None)

        if not text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return {
            "success": True,
            "response": text,
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI request failed: {str(e)}"
        )


# =========================================================
# SIMPLE GENERATE ENDPOINT
# =========================================================

@app.post("/api/generate")
async def generate(request: GenerateRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty."
        )

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=request.prompt.strip()
        )

        text = getattr(response, "text", None)

        if not text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return {
            "success": True,
            "response": text,
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


# =========================================================
# STUDENT AI ENDPOINT
# =========================================================

@app.post("/api/student")
async def student_ai(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )

    try:

        student_prompt = f"""
You are ZenX AI, an AI assistant designed for students.

Help the student with:
- Physics
- Chemistry
- Mathematics
- JEE preparation
- School studies
- Concepts
- Doubt solving
- Study planning
- Programming
- General academic questions

Explain answers clearly and step-by-step.
Do not unnecessarily make answers complicated.

Student question:
{request.message.strip()}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=student_prompt
        )

        text = getattr(response, "text", None)

        if not text:
            raise HTTPException(
                status_code=502,
                detail="No response generated."
            )

        return {
            "success": True,
            "response": text,
            "mode": "student",
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Student AI failed: {str(e)}"
        )


# =========================================================
# ERROR HANDLING
# =========================================================

@app.get("/api/test")
async def test():
    return {
        "success": True,
        "message": "ZenX API test successful"
    }
