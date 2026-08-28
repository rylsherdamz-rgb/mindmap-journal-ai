"""Emotion classification service.

Loads the self-trained scikit-learn pipeline (produced by `ml/train.py`) once at
startup and serves low-latency predictions with per-class confidence scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class EmotionModel:
    """Wraps the trained TF-IDF + LogisticRegression pipeline."""

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                f"Run `python ml/train.py` first to produce it."
            )
        artifact: dict[str, Any] = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        # label_names maps int -> emotion string.
        self.label_names: dict[int, str] = {
            int(k): v for k, v in artifact["label_names"].items()
        }

    def predict(self, text: str) -> dict[str, Any]:
        """Classify a single piece of text.

        Returns the dominant emotion, its confidence, and the full probability
        distribution across all emotions (useful for the mood dashboard).
        """
        probabilities = self.pipeline.predict_proba([text])[0]
        scores = {
            self.label_names[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
        top_index = int(probabilities.argmax())
        return {
            "emotion": self.label_names[top_index],
            "confidence": round(float(probabilities[top_index]), 4),
            "scores": scores,
        }
