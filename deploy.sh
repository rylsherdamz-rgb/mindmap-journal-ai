#!/usr/bin/env bash
#
# MindMap Journal — one-command Cloud Run deploy.
#
# Prerequisites (all one-time):
#   1. gcloud CLI installed and authenticated:   gcloud auth login
#   2. A GCP project with billing enabled.
#   3. (Optional but recommended) a Gemini API key from https://aistudio.google.com/app/apikey
#
# Usage:
#   PROJECT_ID=my-project ./deploy.sh
#   PROJECT_ID=my-project REGION=us-central1 GEMINI_API_KEY=xxxx ./deploy.sh
#
# Environment variables:
#   PROJECT_ID       (required) your Google Cloud project id
#   REGION           (optional) default: us-central1
#   SERVICE_NAME     (optional) default: mindmap-journal
#   GEMINI_API_KEY   (optional) if set, it is stored in Secret Manager and wired to the service
#
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-mindmap-journal}"

# --- Preflight ---------------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
  echo "❌ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

if [[ -z "${PROJECT_ID:-}" ]]; then
  echo "❌ PROJECT_ID is required.  Example:  PROJECT_ID=my-project ./deploy.sh"
  exit 1
fi

# Verify auth is live (fails fast with a clear message if tokens expired).
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  echo "❌ gcloud is not authenticated (or your token expired)."
  echo "   Run:  gcloud auth login"
  exit 1
fi

echo "▶ Project : $PROJECT_ID"
echo "▶ Region  : $REGION"
echo "▶ Service : $SERVICE_NAME"
gcloud config set project "$PROJECT_ID" >/dev/null

# --- 1. Enable required APIs -------------------------------------------------
echo "▶ Enabling required APIs (idempotent)…"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  artifactregistry.googleapis.com

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# --- 2. (Optional) store the Gemini key in Secret Manager --------------------
DEPLOY_SECRET_FLAG=()
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  echo "▶ Storing Gemini API key in Secret Manager…"
  if ! gcloud secrets describe GEMINI_API_KEY >/dev/null 2>&1; then
    gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"
  fi
  echo -n "$GEMINI_API_KEY" | gcloud secrets versions add GEMINI_API_KEY --data-file=-
  gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor" >/dev/null
  DEPLOY_SECRET_FLAG=(--set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest")
else
  echo "⚠ No GEMINI_API_KEY provided — the app will use its local fallback reflections."
fi

# --- 3. Ensure Firestore access for the runtime service account --------------
echo "▶ Granting Firestore (datastore.user) to the runtime service account…"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/datastore.user" >/dev/null || true

# --- 4. Deploy from source (Cloud Build uses the Dockerfile) -----------------
echo "▶ Deploying to Cloud Run…"
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
  "${DEPLOY_SECRET_FLAG[@]}"

# --- 5. Apply the mandatory challenge verification label ---------------------
echo "▶ Applying challenge label (dev-tutorial=cloud-run-ai-challenge)…"
gcloud run services update "$SERVICE_NAME" \
  --update-labels=dev-tutorial=cloud-run-ai-challenge \
  --region="$REGION" >/dev/null

# --- 6. Print the live URL ---------------------------------------------------
URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"
echo ""
echo "✅ Deployed:  $URL"
echo ""
echo "Next steps:"
echo "  • Provision Firestore (Native mode) once per project:"
echo "      gcloud firestore databases create --location=$REGION"
echo "  • Deploy Firestore rules:  firebase deploy --only firestore:rules --project $PROJECT_ID"
echo "  • Enable Google Sign-In in the Firebase console and paste your web config"
echo "    into app/static/index.html (firebaseConfig), then add $URL to Authorized domains."
