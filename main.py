from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os

app = FastAPI(title="ZenX API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class AskRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "ZenX AI is live 🚀"}


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ZenX AI"}


@app.post("/api/ask")
def ask(body: AskRequest):
    q = body.question.strip()

    if not q:
        return {"error": "Question is required."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=q,
        )

        return {"answer": response.text}

    except Exception:
        return {"error": "AI request failed."}
