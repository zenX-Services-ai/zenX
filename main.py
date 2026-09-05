from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import base64

app = FastAPI(title="ZenX AI", version="2027.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


class AskRequest(BaseModel):
    question: str
    mode: str = "student"
    history: list = []
    user_id: str | None = None


class ImageRequest(BaseModel):
    prompt: str
    user_id: str | None = None


def system_prompt(mode):
    prompts = {
        "student": "You are ZenX AI, a friendly student AI tutor. Explain concepts simply and accurately.",
        "jee": "You are ZenX JEE AI. Teach Physics, Chemistry and Mathematics for JEE Main and Advanced with step-by-step solutions.",
        "school": "You are ZenX School AI. Help students understand school concepts, revision and exam preparation.",
        "coding": "You are ZenX Coding AI. Explain programming clearly, debug code and provide clean examples.",
        "general": "You are ZenX AI. Answer questions accurately and clearly."
    }

    return prompts.get(mode, prompts["student"])


@app.get("/")
def root():
    return {
        "status": "online",
        "name": "ZenX AI",
        "version": "2027"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": client is not None
    }


@app.post("/api/ask")
def ask(request: AskRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured."
        )

    prompt = system_prompt(request.mode)

    recent_history = request.history[-20:]

    if recent_history:
        prompt += "\n\nConversation history:\n"

        for item in recent_history:
            role = item.get("role")
            content = item.get("content")

            if content:
                prompt += f"{role}: {content}\n"

    prompt += f"\n\nCurrent question:\n{request.question}"

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response."
            )

        return {
            "answer": answer,
            "mode": request.mode
        }

    except HTTPException:
        raise

    except Exception as error:
        print("GEMINI ERROR:", error)

        raise HTTPException(
            status_code=500,
            detail="Gemini AI request failed."
        )


@app.post("/api/image")
def generate_image(request: ImageRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Image prompt cannot be empty."
        )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured."
        )

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=request.prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        for part in response.candidates[0].content.parts:

            if getattr(part, "inline_data", None):

                image_bytes = part.inline_data.data

                mime_type = (
                    part.inline_data.mime_type
                    or "image/png"
                )

                encoded = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                return {
                    "image_url":
                    f"data:{mime_type};base64,{encoded}"
                }

        raise HTTPException(
            status_code=500,
            detail="No image was returned by Gemini."
        )

    except HTTPException:
        raise

    except Exception as error:

        print("IMAGE ERROR:", error)

        raise HTTPException(
            status_code=500,
            detail="Gemini image generation failed."
        )


if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
