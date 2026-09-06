import os
import json
import uuid
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GROQ_API_KEY environment variable before running this server.")

MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"
SYSTEM_PROMPT = (
    "You are a helpful, professional AI assistant. Write in clear, natural "
    "prose, the way a knowledgeable person would explain something in "
    "conversation. Avoid excessive markdown formatting — no heavy use of "
    "headers, bullet lists, bold text, or emojis unless they genuinely aid "
    "clarity (e.g. a short list of distinct steps). Keep responses concise "
    "and directly useful rather than padded with sections. Ask clarifying "
    "questions when a request is ambiguous, and admit when you don't know "
    "something."
)
HISTORY_FILE = "conversations.json"

client = Groq(api_key=API_KEY)


def load_conversations() -> Dict[str, List[dict]]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_conversations():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


conversations: Dict[str, List[dict]] = load_conversations()

app = FastAPI(title="Free ChatGPT-style Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    image: str | None = None


@app.post("/session")
def new_session():
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    save_conversations()
    return {"session_id": session_id}


@app.get("/sessions")
def list_sessions():
    return [
        {
            "session_id": sid,
            "preview": (msgs[0]["content"][:50] if msgs else "New chat"),
        }
        for sid, msgs in conversations.items()
    ]


@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    return conversations[session_id]


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    del conversations[session_id]
    save_conversations()
    return {"deleted": session_id}


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id
    if session_id is None or session_id not in conversations:
        raise HTTPException(status_code=400, detail="Unknown or missing session_id")

    history = conversations[session_id]

    # ---- Vision request (an image was attached) ----
    if req.image:
        history.append({"role": "user", "content": req.message or "[sent an image]"})
        save_conversations()

        def vision_stream():
            assistant_text = ""
            vision_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (req.message or "Describe this image.") + " Answer in clear, natural, professional prose without heavy markdown formatting, headers, or emojis."},
                    {"type": "image_url", "image_url": {"url": req.image}},
                ],
            }]
            stream = client.chat.completions.create(
                model=VISION_MODEL,
                messages=vision_messages,
                stream=True,
                reasoning_effort="none",  # skip internal <think> reasoning output
                max_tokens=800,  # stay under Groq's free-tier output-tokens-per-minute limit
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    assistant_text += delta
                    yield f"data: {delta}\n\n"
            history.append({"role": "assistant", "content": assistant_text})
            save_conversations()
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(vision_stream(), media_type="text/event-stream")

    # ---- Normal text chat ----
    history.append({"role": "user", "content": req.message})
    save_conversations()

    def event_stream():
        assistant_text = ""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                assistant_text += delta
                yield f"data: {delta}\n\n"
        history.append({"role": "assistant", "content": assistant_text})
        save_conversations()
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory="static", html=True), name="static")