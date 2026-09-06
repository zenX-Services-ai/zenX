
import { GoogleGenAI } from "@google/genai";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed"
    });
  }

  try {
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        error: "GEMINI_API_KEY is not configured in Vercel."
      });
    }

    const {
      messages = [],
      prompt = "",
      mode = "Student"
    } = req.body || {};

    if (!prompt.trim()) {
      return res.status(400).json({
        error: "Message is required."
      });
    }

    const ai = new GoogleGenAI({
      apiKey: apiKey
    });

    const history = messages
      .filter(
        message =>
          message.role === "user" ||
          message.role === "model"
      )
      .slice(-20)
      .map(message => ({
        role: message.role,
        parts: [
          {
            text: String(
              message.content || ""
            ).slice(0, 10000)
          }
        ]
      }));

    if (
      history.length === 0 ||
      history[history.length - 1]?.parts?.[0]?.text !== prompt
    ) {
      history.push({
        role: "user",
        parts: [
          {
            text: prompt.slice(0, 10000)
          }
        ]
      });
    }

    const systemInstruction = `
You are ZenX.AI.

Founder: Abhay Srivastava.

You are a fast, helpful, accurate and friendly AI assistant.

Current mode: ${mode}

STUDENT MODE:
Help students understand concepts clearly.
Support school studies, JEE, NEET, homework, revision,
doubt solving and exam preparation.

JEE MODE:
Focus on Physics, Chemistry and Mathematics.
Give step-by-step solutions.
Use formulas and shortcuts when useful.
Keep explanations exam-oriented.

NEET MODE:
Focus on Biology, Physics and Chemistry.
Explain concepts clearly and accurately.

CODING MODE:
Help with HTML, CSS, JavaScript, Python and other programming.
Provide clean code and explain important errors.
Never claim that code was executed if it was not.

STUDY PLANNER:
Create realistic study schedules with revision,
practice and breaks.

GENERAL:
Answer directly.
Do not unnecessarily repeat the question.
Use simple language when the user asks for an explanation.
Be accurate.
If you are uncertain, say so rather than inventing facts.
`;

    const response = await ai.models.generateContent({
      model: "gemini-3.7-flash",

      contents: history,

      config: {
        systemInstruction: systemInstruction
      }
    });

    const answer =
      response.text ||
      "Sorry, I couldn't generate a response.";

    res.setHeader(
      "Cache-Control",
      "no-store"
    );

    return res.status(200).json({
      text: answer
    });

  } catch (error) {

    console.error(
      "ZenX.AI error:",
      error
    );

    return res.status(500).json({
      error:
        error?.message ||
        "ZenX.AI request failed."
    });
  }
}
