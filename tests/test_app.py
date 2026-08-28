"""Automated test suite for MindMap Journal.

Covers:
  * The self-trained ML model loads and classifies sensibly.
  * Public API endpoints: /health, /predict, /reflect (local fallback path).
  * Input validation (empty text rejected).
  * Auth enforcement: /entries requires a valid Bearer token.

These tests run without any cloud credentials — the Gemini and Firestore paths
degrade gracefully, which is exactly what we assert.

Run:  pytest -q
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Ensure no Gemini key is present so /reflect deterministically uses the local fallback.
os.environ.pop("GEMINI_API_KEY", None)

from app.emotion_model import EmotionModel  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


# --- ML model (unit) ---------------------------------------------------------

def test_model_loads_and_has_six_emotions():
    model = EmotionModel("model/emotion_model.joblib")
    assert set(model.label_names.values()) == {
        "sadness", "joy", "love", "anger", "fear", "surprise",
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I am so scared something terrible will happen tomorrow", "fear"),
        ("I feel so hopeless and empty and full of grief", "sadness"),
        ("I am absolutely thrilled and delighted with the great news", "joy"),
    ],
)
def test_model_predicts_expected_emotion(text, expected):
    model = EmotionModel("model/emotion_model.joblib")
    result = model.predict(text)
    assert result["emotion"] == expected
    assert 0.0 <= result["confidence"] <= 1.0
    # Probability distribution should sum to ~1 across the six classes
    # (scores are rounded to 4 decimals, so allow a small tolerance).
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-3


# --- /health -----------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["emotions"]) == 6


# --- /predict ----------------------------------------------------------------

def test_predict_returns_emotion_and_scores():
    r = client.post("/predict", json={"text": "I'm terrified of the dark"})
    assert r.status_code == 200
    body = r.json()
    assert body["emotion"] in {"sadness", "joy", "love", "anger", "fear", "surprise"}
    assert "scores" in body and len(body["scores"]) == 6


def test_predict_rejects_empty_text():
    r = client.post("/predict", json={"text": ""})
    # Pydantic min_length validation -> 422.
    assert r.status_code == 422


def test_predict_rejects_missing_field():
    r = client.post("/predict", json={})
    assert r.status_code == 422


# --- /reflect (local fallback, no Gemini key) --------------------------------

def test_reflect_uses_local_fallback_without_key():
    r = client.post("/reflect", json={"text": "I lost my keys and I'm furious", "emotion": "anger"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "local-fallback"
    assert body["degraded"] is True
    assert isinstance(body["reflection"], str) and body["reflection"]


def test_reflect_infers_emotion_when_omitted():
    r = client.post("/reflect", json={"text": "I am overjoyed and celebrating"})
    assert r.status_code == 200
    assert "reflection" in r.json()


# --- /entries (auth enforced) ------------------------------------------------

def test_entries_get_requires_auth():
    r = client.get("/entries")
    assert r.status_code == 401


def test_entries_post_requires_auth():
    r = client.post("/entries", json={"text": "hello"})
    assert r.status_code == 401


def test_entries_rejects_invalid_token():
    r = client.get("/entries", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


# --- Frontend ----------------------------------------------------------------

def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "MindMap Journal" in r.text
