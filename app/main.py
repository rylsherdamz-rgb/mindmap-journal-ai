"""MindMap Journal — FastAPI application entrypoint.

Routes:
  GET  /health          -> liveness probe + model metadata
  POST /predict         -> classify emotion with the self-trained ML model
  POST /reflect         -> Gemini reflection (fallback ladder) + local fallback
  POST /entries         -> save an entry to the user's private Firestore collection
  GET  /entries         -> list the authenticated user's entries (mood dashboard data)
  GET  /                -> the single-page frontend app

Security posture:
  * Body/JSON parsing is configured before routes (FastAPI does this by default).
  * All inputs are validated with Pydantic and defensively guarded.
  * /entries requires a verified Firebase ID token (JWT) and is user-isolated.
  * The Gemini key comes from Secret Manager; nothing secret is hardcoded.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import init_firebase, verify_token
from app.config import get_settings
from app.emotion_model import EmotionModel
from app.gemini_service import generate_reflection

settings = get_settings()
app = FastAPI(title="MindMap Journal", version="1.0.0")

# Load the trained ML model once at startup (fail fast if the artifact is missing).
emotion_model = EmotionModel(settings.model_path)

# Best-effort Firebase init (won't crash if creds are absent locally).
init_firebase()

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --- Request/response schemas ------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ReflectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    emotion: str = Field(default="", max_length=32)


class EntryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    emotion: str | None = Field(default=None, max_length=32)
    confidence: float | None = None
    scores: dict[str, float] | None = None
    reflection: str | None = Field(default=None, max_length=4000)
    reflection_model: str | None = Field(default=None, max_length=64)


# --- Health ------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "tfidf+logreg",
        "emotions": list(emotion_model.label_names.values()),
    }


# --- ML prediction (public; no user data involved) ---------------------------

@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    return emotion_model.predict(text)


# --- Gemini reflection --------------------------------------------------------

@app.post("/reflect")
def reflect(req: ReflectRequest) -> dict:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    emotion = (req.emotion or "").strip()
    if not emotion:
        # If the client didn't classify first, do it here so the reflection is informed.
        emotion = emotion_model.predict(text)["emotion"]
    return generate_reflection(text, emotion)


# --- User-isolated journal entries (auth required) ----------------------------

@app.post("/entries")
def create_entry(req: EntryRequest, uid: str = Depends(verify_token)) -> dict:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    from app import firestore_service

    try:
        saved = firestore_service.save_entry(uid, req.model_dump())
        return {"ok": True, "entry": saved}
    except Exception as exc:  # noqa: BLE001
        # Explicit error escalation — never fail silently on a write.
        raise HTTPException(
            status_code=502, detail=f"Failed to save entry: {exc}"
        ) from exc


@app.get("/entries")
def get_entries(uid: str = Depends(verify_token)) -> dict:
    from app import firestore_service

    try:
        entries = firestore_service.list_entries(uid)
        return {"ok": True, "entries": entries}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Failed to load entries: {exc}"
        ) from exc


# --- Frontend (served last so API routes take precedence) --------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
