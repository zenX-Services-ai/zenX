from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import base64

app = FastAPI(title="ZenX API", version="1.1.0")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class ImageRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"message": "ZenX API is live"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# TEXT AI
@app.post("/api/ask")
def ask(body: AskRequest):
    q = body.question.strip()

    if not q:
        return {"error": "Question is required."}

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=q
        )

        return {"answer": response.text}

    except Exception as e:
        return {"error": str(e)}
@app.post("/api/generate-image")
def generate_image(body: ImageRequest):
    prompt = body.prompt.strip()

    if not prompt:
        return {"error": "Image prompt is required."}

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-image",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )

        for part in response.parts:
            if part.inline_data is not None:
                import base64

                image_data = base64.b64encode(
                    part.inline_data.data
                ).decode("utf-8")

                mime_type = part.inline_data.mime_type or "image/png"

                return {
                    "image": f"data:{mime_type};base64,{image_data}"
                }

        return {"error": "No image was generated."}

    except Exception as e:
        return {"error": str(e)}

