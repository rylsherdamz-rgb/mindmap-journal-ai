"""
MindMap Journal — Emotion Classifier Training Pipeline
======================================================

Trains a text-emotion classifier on the public `dair-ai/emotion` dataset
(https://huggingface.co/datasets/dair-ai/emotion): ~20,000 English messages,
each labeled with one of six emotions.

Pipeline: TF-IDF vectorizer -> Logistic Regression (scikit-learn).
The whole fitted Pipeline is serialized with joblib to `model/emotion_model.joblib`
and loaded once at API startup for low-latency inference.

Run:
    python ml/train.py

This downloads the data, trains, prints a full classification report, and saves
the model artifact + a metrics.json alongside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# --- Configuration -----------------------------------------------------------

# Integer label -> human-readable emotion (defined by the dair-ai/emotion dataset).
LABEL_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# Public dataset location (Parquet files hosted on the Hugging Face Hub).
_HF_BASE = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split/"
DATA_URLS = {
    "train": _HF_BASE + "train-00000-of-00001.parquet",
    "validation": _HF_BASE + "validation-00000-of-00001.parquet",
    "test": _HF_BASE + "test-00000-of-00001.parquet",
}

# Output locations.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "emotion_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DATA_CACHE = PROJECT_ROOT / "data" / "raw"


# --- Data loading ------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    """Load a dataset split, caching the parquet locally so re-runs are offline-friendly."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_CACHE / f"{split}.parquet"

    if cache_file.exists():
        print(f"  [{split}] using cached {cache_file.relative_to(PROJECT_ROOT)}")
        return pd.read_parquet(cache_file)

    print(f"  [{split}] downloading {DATA_URLS[split]}")
    df = pd.read_parquet(DATA_URLS[split])
    df.to_parquet(cache_file)
    return df


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading dair-ai/emotion dataset...")
    train = load_split("train")
    val = load_split("validation")
    test = load_split("test")
    print(f"  train={len(train)}  validation={len(val)}  test={len(test)}")
    return train, val, test


# --- Model -------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """TF-IDF (1-2 grams) + multinomial Logistic Regression.

    Balanced class weights compensate for the dataset's skew (joy/sadness are far
    more common than love/surprise).
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# --- Training entrypoint -----------------------------------------------------

def main() -> None:
    train, val, test = load_dataset()

    # Train on train + validation for the final shipped model; evaluate on the
    # held-out test split (which the model never sees during fitting).
    X_train = pd.concat([train["text"], val["text"]], ignore_index=True)
    y_train = pd.concat([train["label"], val["label"]], ignore_index=True)
    X_test, y_test = test["text"], test["label"]

    print("\nTraining TF-IDF + LogisticRegression pipeline...")
    t0 = time.time()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    train_secs = round(time.time() - t0, 2)
    print(f"  done in {train_secs}s")

    print("\nEvaluating on held-out test split...")
    y_pred = pipeline.predict(X_test)
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    report_text = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n" + report_text)
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    # Persist artifact + label map so the API can decode predictions.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"pipeline": pipeline, "label_names": LABEL_NAMES}
    joblib.dump(artifact, MODEL_PATH, compress=3)
    print(f"\nSaved model artifact -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    metrics = {
        "dataset": "dair-ai/emotion",
        "model": "TfidfVectorizer(1-2gram) + LogisticRegression(balanced)",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_seconds": train_secs,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "labels": LABEL_NAMES,
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics        -> {METRICS_PATH.relative_to(PROJECT_ROOT)}")

    # Quick smoke test to demonstrate inference on fresh text.
    print("\nSample predictions:")
    samples = [
        "I finally got the job I've been dreaming about for years!",
        "I miss my grandmother so much, the house feels empty without her.",
        "How dare they cancel the project after all our hard work.",
        "I'm terrified about the results coming back tomorrow.",
    ]
    probs = pipeline.predict_proba(samples)
    for text, dist in zip(samples, probs):
        top = dist.argmax()
        print(f"  [{LABEL_NAMES[top]:8s} {dist[top]:.2f}]  {text}")


if __name__ == "__main__":
    main()
