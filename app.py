import os
import json
import uuid
import sqlite3
import datetime
from typing import Dict, List

import jwt
import bcrypt
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GROQ_API_KEY environment variable before running this server.")

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-secret-change-me")

MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"
SYSTEM_PROMPT = (
    "You are a helpful, professional AI assistant. Write in clear, natural "
    "prose, the way a knowledgeable person would explain something in "
    "conversation. Avoid excessive markdown formatting. Ask clarifying "
    "questions when a request is ambiguous, and admit when you don't know "
    "something."
)
HISTORY_FILE = "conversations.json"
DB_FILE = "users.db"

client = Groq(api_key=API_KEY)


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


def get_user_by_username(username: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


def create_user(username: str, password: str) -> str:
    user_id = str(uuid.uuid4())
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
        (user_id, username, password_hash),
    )
    conn.commit()
    conn.close()
    return user_id


def make_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def get_current_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["user_id"]


def load_conversations() -> Dict[str, Dict[str, List[dict]]]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_conversations():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


conversations: Dict[str, Dict[str, List[dict]]] = load_conversations()

app = FastAPI(title="Free ChatGPT-style Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AuthRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    image: str | None = None


@app.post("/signup")
def signup(req: AuthRequest):
    if len(req.username) < 3 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Username must be 3+ chars, password 6+ chars")
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    user_id = create_user(req.username, req.password)
    conversations[user_id] = {}
    save_conversations()
    token = make_token(user_id)
    return {"token": token, "username": req.username}


@app.post("/login")
def login(req: AuthRequest):
    row = get_user_by_username(req.username)
    if not row or not bcrypt.checkpw(req.password.encode("utf-8"), row[2].encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user_id = row[0]
    if user_id not in conversations:
        conversations[user_id] = {}
        save_conversations()
    token = make_token(user_id)
    return {"token": token, "username": req.username}


@app.post("/session")
def new_session(user_id: str = Depends(get_current_user_id)):
    session_id = str(uuid.uuid4())
    conversations.setdefault(user_id, {})[session_id] = []
    save_conversations()
    return {"session_id": session_id}


@app.get("/sessions")
def list_sessions(user_id: str = Depends(get_current_user_id)):
    user_convos = conversations.get(user_id, {})
    return [
        {"session_id": sid, "preview": (msgs[0]["content"][:50] if msgs else "New chat")}
        for sid, msgs in user_convos.items()
    ]


@app.get("/history/{session_id}")
def get_history(session_id: str, user_id: str = Depends(get_current_user_id)):
    user_convos = conversations.get(user_id, {})
    if session_id not in user_convos:
        raise HTTPException(status_code=404, detail="Session not found")
    return user_convos[session_id]


@app.delete("/session/{session_id}")
def delete_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    user_convos = conversations.get(user_id, {})
    if session_id not in user_convos:
        raise HTTPException(status_code=404, detail="Session not found")
    del user_convos[session_id]
    save_conversations()
    return {"deleted": session_id}


@app.post("/chat")
def chat(req: ChatRequest, user_id: str = Depends(get_current_user_id)):
    session_id = req.session_id
    user_convos = conversations.setdefault(user_id, {})
    if session_id is None or session_id not in user_convos:
        raise HTTPException(status_code=400, detail="Unknown or missing session_id")

    history = user_convos[session_id]

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
                reasoning_effort="none",
                max_tokens=800,
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