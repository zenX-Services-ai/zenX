from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ZenX API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "ZenX API is live"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/ask")
def ask(body: AskRequest):
    q = body.question.strip()
    if not q:
        return {"error": "Question is required."}
    return {"answer": f"ZenX received: {q}"}
    if not q:
        return {"error": "Question is required."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=q,
        )
        return {"answer": response.text}

    except Exception as e:
        return {"error": str(e)}
    if not q:
        return {"error": "Question is required."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=q,
        )
        return {"answer": response.text}

    except Exception as e:
        return {"error": str(e)}   
        q = body.question.strip()

    if not q:
        return {"error": "Question is required."}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=q,
        )

        return {"answer": response.text}

    except Exception as e:
    return {"error": str(e)}
