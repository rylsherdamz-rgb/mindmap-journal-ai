"""Firebase authentication — backend JWT verification.

The frontend signs users in with Google via Firebase Auth and sends the resulting
ID token as a Bearer token. Here we verify that token server-side using the
Firebase Admin SDK, so every protected endpoint has a trusted `uid`. This enforces
the challenge's "Auth State Integrity" directive.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)
_initialized = False


def init_firebase() -> None:
    """Initialize the Admin SDK once, using Application Default Credentials.

    On Cloud Run the runtime service account provides credentials automatically;
    locally you can point GOOGLE_APPLICATION_CREDENTIALS at a service account file.
    """
    global _initialized
    if _initialized:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials

        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.ApplicationDefault())
        _initialized = True
    except Exception as exc:  # noqa: BLE001
        # Defer failure to request time so the ML endpoints still work locally
        # without Firebase configured.
        print(f"[auth] Firebase Admin init deferred/failed: {exc}")


def verify_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency: verify the Bearer ID token and return the user's uid."""
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token.",
        )
    init_firebase()
    try:
        from firebase_admin import auth as fb_auth

        decoded = fb_auth.verify_id_token(creds.credentials)
        uid = decoded.get("uid")
        if not uid:
            raise ValueError("token missing uid")
        return uid
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        ) from exc
