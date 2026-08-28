"""Application configuration and secret retrieval.

Secrets (the Gemini API key) are never hardcoded. Resolution order:
1. `GEMINI_API_KEY` environment variable (useful for local dev).
2. Google Cloud Secret Manager, using `GEMINI_API_KEY_SECRET` as the secret id
   and the ambient project id (from `GOOGLE_CLOUD_PROJECT` / metadata server).

This mirrors the challenge's "Zero-Hardcoding Hygiene" directive.
"""

from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Runtime configuration sourced from the environment."""

    def __init__(self) -> None:
        # Google Cloud project id (auto-populated on Cloud Run).
        self.project_id: str = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
            or ""
        )
        # Secret Manager secret id that stores the Gemini API key.
        self.gemini_secret_id: str = os.environ.get(
            "GEMINI_API_KEY_SECRET", "GEMINI_API_KEY"
        )
        # Optional direct key for local development (never commit a real value).
        self.gemini_api_key_env: str = os.environ.get("GEMINI_API_KEY", "")
        # Path to the trained model artifact.
        self.model_path: str = os.environ.get(
            "MODEL_PATH", "model/emotion_model.joblib"
        )
        # Server port (Cloud Run injects PORT).
        self.port: int = int(os.environ.get("PORT", "8080"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_gemini_api_key() -> str | None:
    """Resolve the Gemini API key from env first, then Secret Manager.

    Returns None if no key is configured — callers must degrade gracefully
    (the /reflect endpoint falls back to a local, non-AI reflection).
    """
    settings = get_settings()

    # 1. Environment variable (local dev / explicit override).
    if settings.gemini_api_key_env:
        return settings.gemini_api_key_env

    # 2. Secret Manager (production on Cloud Run).
    if not settings.project_id:
        return None
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        name = (
            f"projects/{settings.project_id}"
            f"/secrets/{settings.gemini_secret_id}/versions/latest"
        )
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash startup
        print(f"[config] Secret Manager lookup failed: {exc}")
        return None
