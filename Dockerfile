# MindMap Journal — single-container image for Google Cloud Run.
#
# Python 3.12 (stable, full wheel coverage for scikit-learn / firebase-admin /
# google-genai). The trained model artifact is copied into the image so the
# container serves predictions immediately with no training step at runtime.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code, trained model artifact, and frontend.
COPY app/ ./app/
COPY model/ ./model/

# Cloud Run sends traffic to $PORT (default 8080).
EXPOSE 8080

# Start the ASGI server. Single worker keeps the in-memory model footprint small;
# Cloud Run scales horizontally by adding instances.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
