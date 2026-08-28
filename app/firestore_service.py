"""Cloud Firestore persistence — user-isolated journal entries.

Entries live at `users/{uid}/entries/{entryId}`. All reads and writes are scoped
to the authenticated uid, and the Firestore security rules (firestore.rules)
enforce the same isolation at the database layer as defence in depth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_client = None


def _get_client():
    """Lazily create a Firestore client (Admin SDK)."""
    global _client
    if _client is None:
        from firebase_admin import firestore

        _client = firestore.client()
    return _client


def _clean(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip None values so we never send `undefined`-like fields to the driver."""
    return {k: v for k, v in payload.items() if v is not None}


def save_entry(uid: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Persist a journal entry under the user's private collection.

    Returns the stored document (including its generated id and server timestamp).
    """
    client = _get_client()
    doc_ref = client.collection("users").document(uid).collection("entries").document()

    record = _clean(
        {
            "text": entry.get("text", ""),
            "emotion": entry.get("emotion"),
            "confidence": entry.get("confidence"),
            "scores": entry.get("scores"),
            "reflection": entry.get("reflection"),
            "reflection_model": entry.get("reflection_model"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    doc_ref.set(record)
    return {"id": doc_ref.id, **record}


def list_entries(uid: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return the user's entries, newest first."""
    from firebase_admin import firestore

    client = _get_client()
    query = (
        client.collection("users")
        .document(uid)
        .collection("entries")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    results = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        results.append({"id": doc.id, **data})
    return results
