from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import base64

app = FastAPI(
    title="ZenX AI",
    version="2.0.0"
)

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# OPENAI
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

CHAT_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-1"
)

# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str
    mode: str = "student"
    history: list = []
    user_id: str | None = None


class ImageRequest(BaseModel):
    prompt: str
    user_id: str | None = None


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

def get_system_prompt(mode: str) -> str:

    base = """
You are ZenX AI, a helpful educational AI assistant.

Your goal is to help students understand concepts clearly,
solve problems step by step, create revision material,
practice questions, and learn efficiently.

Be accurate, concise when possible, and explain difficult
ideas in simple language.

For mathematics and physics, show useful equations and
logical steps.

For chemistry, explain concepts, reactions, formulas and
reasoning clearly.

Never pretend to have performed an action that you did not
actually perform.
"""

    modes = {

        "student": """
Focus on school and student learning.
Adapt explanations to the student's level.
""",

        "jee": """
You are in JEE preparation mode.
Focus on JEE Main and JEE Advanced level Physics,
Chemistry and Mathematics.
Give conceptual explanations, shortcuts only when valid,
and step-by-step solutions.
""",

        "school": """
Focus on school-level learning, NCERT-style concepts,
exam preparation, revision and homework understanding.
""",

        "coding": """
You are in coding mode.
Explain programming concepts, debug code,
write clean examples and explain why the code works.
""",

        "general": """
Answer general questions accurately and clearly.
"""
    }

    return base + modes.get(
        mode,
        modes["student"]
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "online",
        "name": "ZenX AI",
        "version": "2.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "ai_configured": client is not None
    }


# --------------------------------------------------
# AI CHAT
# --------------------------------------------------

@app.post("/api/ask")
def ask_ai(request: AskRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured. Add OPENAI_API_KEY in Render Environment Variables."
        )

    messages = [
        {
            "role": "system",
            "content": get_system_prompt(request.mode)
        }
    ]

    # Keep only recent conversation
    history = request.history[-20:]

    for item in history:

        role = item.get("role")
        content = item.get("content")

        if role not in ["user", "assistant"]:
            continue

        if not content:
            continue

        messages.append({
            "role": role,
            "content": str(content)
        })

    # Avoid duplicating the current question
    if not history or history[-1].get("content") != request.question:

        messages.append({
            "role": "user",
            "content": request.question
        })

    try:

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="AI returned an empty response."
            )

        return {
            "answer": answer,
            "mode": request.mode
        }

    except HTTPException:
        raise

    except Exception as error:

        print("AI ERROR:", error)

        raise HTTPException(
            status_code=500,
            detail="AI request failed."
        )


# --------------------------------------------------
# IMAGE GENERATION
# --------------------------------------------------

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
            detail="Image AI is not configured. Add OPENAI_API_KEY in Render Environment Variables."
        )

    try:

        result = client.images.generate(
            model=IMAGE_MODEL,
            prompt=request.prompt,
            size="1024x1024"
        )

        image_data = result.data[0]

        # Some API responses provide a URL
        if getattr(image_data, "url", None):

            return {
                "image_url": image_data.url
            }

        # Some image models provide base64
        if getattr(image_data, "b64_json", None):

            image_bytes = base64.b64decode(
                image_data.b64_json
            )

            image_url = (
                "data:image/png;base64,"
                + base64.b64encode(image_bytes).decode()
            )

            return {
                "image_url": image_url
            }

        raise HTTPException(
            status_code=500,
            detail="Image was generated but no image data was returned."
        )

    except HTTPException:
        raise

    except Exception as error:

        print("IMAGE ERROR:", error)

        raise HTTPException(
            status_code=500,
            detail="Image generation failed."
        )


# --------------------------------------------------
# RUN LOCALLY
# --------------------------------------------------

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
