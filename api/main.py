from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import base64
import tempfile
from pathlib import Path

app = FastAPI(
    title="ZenX AI",
    version="1.1.0"
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


# ==============================
# NORMAL CHAT
# ==============================

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


# ==============================
# IMAGE ANALYSIS
# ==============================

@app.post("/api/analyze-image")
async def analyze_image(
    message: str = Form("Analyze this image and explain it clearly."),
    image: UploadFile = File(...)
):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        image_bytes = await image.read()

        if len(image_bytes) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image is too large. Please use an image under 15 MB."
            )

        image_data = base64.b64encode(image_bytes).decode("utf-8")

        prompt = message.strip() or (
            "Analyze this image carefully. "
            "If it contains a JEE question, solve it step by step. "
            "Read all visible text, equations, diagrams and options."
        )

        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image",
                    "data": image_data,
                    "mime_type": image.content_type
                }
            ]
        )

        return {
            "success": True,
            "response": interaction.output_text,
            "filename": image.filename,
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )


# ==============================
# DOCUMENT / FILE ANALYSIS
# ==============================

@app.post("/api/analyze-file")
async def analyze_file(
    message: str = Form("Analyze this document and explain it clearly."),
    file: UploadFile = File(...)
):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )

    allowed_types = {
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }

    filename = file.filename or "document"

    if file.content_type not in allowed_types:
        extension = Path(filename).suffix.lower()

        if extension not in [".pdf", ".txt", ".doc", ".docx"]:
            raise HTTPException(
                status_code=400,
                detail="Supported files: PDF, TXT, DOC and DOCX."
            )

    temp_path = None

    try:
        file_bytes = await file.read()

        if len(file_bytes) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File is too large. Please use a file under 50 MB."
            )

        suffix = Path(filename).suffix or ".bin"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(file_bytes)
            temp_path = temp.name

        uploaded_file = client.files.upload(
            file=temp_path
        )

        prompt = message.strip() or (
            "Analyze this document carefully. "
            "Explain the important content clearly. "
            "If it contains a JEE question, solve it step by step."
        )

        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "document",
                    "uri": uploaded_file.uri,
                    "mime_type": uploaded_file.mime_type
                }
            ]
        )

        return {
            "success": True,
            "response": interaction.output_text,
            "filename": filename,
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File analysis failed: {str(e)}"
        )

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass
