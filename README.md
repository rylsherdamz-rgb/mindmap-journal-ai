# MindMap Journal 🧠

**An AI journaling companion that understands how you feel — powered by a machine-learning
emotion model I trained myself, not just an off-the-shelf chatbot.**

Built for the [Cloud Run "Accelerate AI" Challenge](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-ai-challenge)
· `#AccelerateAIwithCloudRun`

---

## 📖 What is this?

Most journaling apps in this challenge are a thin wrapper around a large language model: you
type, the LLM replies, the text is saved. **MindMap Journal goes further.**

Every entry runs through a **custom emotion classifier I trained from scratch** on a real,
public dataset. That model tags the dominant emotion — *joy, sadness, anger, fear, love,* or
*surprise* — with a confidence score. **Gemini** then reads the same entry and writes a warm,
reflective response. Over time, your tagged entries build a **mood-over-time dashboard** so
you can *see* your emotional patterns emerge.

Three jobs at once:

1. **Classifies** your emotion with a purpose-built ML model *(the part I trained)*.
2. **Reflects** on your entry with Gemini's natural-language understanding.
3. **Remembers** everything privately, so patterns surface across days and weeks.

## 🌟 How it scores on the judging axes

| Axis | How MindMap Journal addresses it |
| :--- | :--- |
| **Authenticity** | A **self-trained scikit-learn emotion model** — real ML, reproducible from `ml/train.py`, with documented metrics. Not a Gemini wrapper. |
| **Usability** | One-click **Google Sign-In**, a single clean dashboard, instant emotion tags, and a live mood chart. |
| **Stability** | A **Gemini model-fallback ladder** (retries across models on 429/503/500/404) plus a deterministic local fallback, defensive input validation, and an error banner with **Retry** so nothing fails silently. |
| **Security** | **User-isolated Firestore rules** (`request.auth.uid == userId`, zero insecure defaults), the Gemini key in **Secret Manager** (no hardcoded secrets), and backend **JWT verification** via the Firebase Admin SDK. |

## ✨ Features

- 🔐 **Google Sign-In** via Firebase Authentication — no passwords stored.
- 🤖 **Self-trained ML emotion classifier** auto-tags every entry (`POST /predict`).
- 💬 **Gemini reflection** with a resilient multi-model fallback ladder (`POST /reflect`).
- 📊 **Mood-over-time dashboard** built live from your accumulated entries.
- 🔒 **Strict per-user data isolation** — you can only read/write your own entries.
- 🗝️ **Secret Manager** integration for the Gemini API key.
- ☁️ **Single-container deploy** to Google Cloud Run.

## 🧠 The machine learning model

- **Task:** multi-class text emotion classification (6 classes).
- **Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — ~20,000
  labeled English messages (sadness, joy, love, anger, fear, surprise).
- **Approach:** **TF-IDF (1–2 grams) + Logistic Regression** with balanced class weights
  (scikit-learn `Pipeline`). Fast to train (~19s), tiny to ship (<1 MB), cheap to serve on
  Cloud Run's free tier. The pipeline is structured so the classifier can later be swapped
  for a fine-tuned transformer without touching the API layer.
- **Artifact:** serialized with `joblib` to `model/emotion_model.joblib`, loaded once at
  startup for low-latency inference.

### Evaluation (held-out test split, 2,000 samples)

| Metric | Score |
| :--- | :--- |
| Accuracy | **0.900** |
| Weighted F1 | **0.902** |
| Macro F1 | **0.867** |

| Emotion | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| sadness | 0.957 | 0.923 | 0.940 |
| joy | 0.947 | 0.892 | 0.919 |
| love | 0.709 | 0.887 | 0.788 |
| anger | 0.880 | 0.909 | 0.894 |
| fear | 0.898 | 0.862 | 0.879 |
| surprise | 0.690 | 0.909 | 0.784 |

### Reproduce the model

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ml/train.py          # downloads data → trains → evaluates → saves model/ + metrics.json
```

## 🏗️ Architecture

```
                 Browser (single-page app)
        Firebase Auth: Google Sign-In → ID token (JWT)
                          │
                          ▼
         Cloud Run service — FastAPI, one container
   ┌──────────────────────────────────────────────────┐
   │  POST /predict  → self-trained emotion model       │  scikit-learn (joblib)
   │  POST /reflect  → Gemini API, fallback ladder       │  key from Secret Manager
   │  GET/POST /entries → Cloud Firestore, user-isolated │  JWT verified (Admin SDK)
   └──────────────────────────────────────────────────┘
                          │
                          ▼
     Cloud Firestore  (rules: request.auth.uid == userId)
```

## 📁 Repository layout

```
.
├── ml/train.py            # ML training pipeline (download → train → evaluate → save)
├── app/
│   ├── main.py            # FastAPI app + routes
│   ├── config.py          # Settings + Secret Manager key retrieval
│   ├── emotion_model.py   # Loads & serves the trained model
│   ├── gemini_service.py  # Gemini client + fallback ladder + local fallback
│   ├── auth.py            # Firebase Admin JWT verification
│   ├── firestore_service.py # User-isolated Firestore reads/writes
│   └── static/index.html  # Single-page frontend (Sign-In, journal, dashboard)
├── model/                 # Trained artifact + metrics.json (produced by ml/train.py)
├── firestore.rules        # User-isolated Firestore security rules
├── Dockerfile             # Single-container build for Cloud Run
├── requirements.txt
└── README.md
```

## 🔌 API reference

| Method | Path | Auth | Description |
| :--- | :--- | :--- | :--- |
| GET | `/health` | — | Liveness + model metadata |
| POST | `/predict` | — | `{ "text": "..." }` → emotion + confidence + score distribution |
| POST | `/reflect` | — | `{ "text": "...", "emotion": "..." }` → Gemini reflection (or local fallback) |
| POST | `/entries` | ✅ Bearer | Save an entry to the user's private collection |
| GET | `/entries` | ✅ Bearer | List the authenticated user's entries |
| GET | `/` | — | Frontend SPA |

---

# 🚀 Deploy to Google Cloud Run

> **One-command deploy:** after `gcloud auth login`, run
> `PROJECT_ID=your-project GEMINI_API_KEY=your-key ./deploy.sh`. The script enables APIs,
> stores the secret, grants IAM, deploys, and applies the challenge label for you. The manual
> steps below explain exactly what it does.

> **Cost:** New Google Cloud accounts get **$300 in free credits**. Cloud Run, Firestore, and
> the Gemini API all have generous free tiers — a demo like this typically costs **$0**.

### 0. Prerequisites

- A Google Cloud project with **billing enabled**.
- [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
  (`gcloud auth login`), or use Cloud Shell.
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="mindmap-journal"
gcloud config set project "$PROJECT_ID"
```

### 1. Enable the required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  artifactregistry.googleapis.com
```

### 2. Provision Cloud Firestore + deploy security rules

Create a Firestore database in **Native mode** (once per project):

```bash
gcloud firestore databases create --location="$REGION"
```

Deploy the user-isolation rules in [`firestore.rules`](./firestore.rules) using the Firebase
CLI (`npm i -g firebase-tools`, then `firebase login`):

```bash
firebase deploy --only firestore:rules --project "$PROJECT_ID"
```

The rules enforce that a user can only read/write `users/{uid}/entries/*` where
`request.auth.uid == uid`, with an explicit deny-all fallback.

### 3. Store the Gemini API key in Secret Manager

```bash
# Create the secret and add your key as a version
gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets versions add GEMINI_API_KEY --data-file=-

# Grant the Cloud Run runtime service account read access
export PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

The runtime service account also needs Firestore access (usually granted by default; if not):

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user"
```

### 4. Configure Firebase Authentication (Google Sign-In)

1. In the [Firebase console](https://console.firebase.google.com), add your GCP project.
2. **Build → Authentication → Sign-in method →** enable **Google**.
3. **Project settings → Your apps → Web app** → copy the web config.
4. Paste the values into the `firebaseConfig` object in
   [`app/static/index.html`](./app/static/index.html) (these are **client-side, non-secret**
   identifiers):

   ```js
   const firebaseConfig = {
     apiKey: "…",
     authDomain: "your-project.firebaseapp.com",
     projectId: "your-project-id",
     appId: "…"
   };
   ```
5. After you know your Cloud Run URL (step 5), add its domain under
   **Authentication → Settings → Authorized domains**.

### 5. Deploy to Cloud Run

Deploy straight from source (Cloud Build containerizes using the included `Dockerfile`):

```bash
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
```

`--allow-unauthenticated` lets users reach the sign-in page; **application** auth is enforced
per request via Firebase JWT verification on `/entries`.

### 6. Apply the mandatory challenge label

```bash
gcloud run services update "$SERVICE_NAME" \
  --update-labels=dev-tutorial=cloud-run-ai-challenge \
  --region="$REGION"
```

### 7. Done

```bash
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)'
```

Open the URL, sign in with Google, write an entry, and watch the emotion tag, Gemini
reflection, and mood chart update live.

---

## 🧪 Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ml/train.py                       # produces model/emotion_model.joblib (or use the committed one)
uvicorn app.main:app --reload --port 8080
```

- `/predict` and `/reflect` work without any cloud setup. Without a `GEMINI_API_KEY` env var,
  `/reflect` returns a graceful **local fallback** reflection.
- `/entries` requires Firebase credentials (set `GOOGLE_APPLICATION_CREDENTIALS`).

Or with Docker:

```bash
docker build -t mindmap-journal .
docker run -p 8080:8080 -e GEMINI_API_KEY="your-key" mindmap-journal
```

## ✅ Testing

An automated test suite (`tests/test_app.py`) covers the ML model, all public endpoints,
input validation, and auth enforcement — no cloud credentials required (the Gemini and
Firestore paths degrade gracefully, which the tests assert):

```bash
pip install -r requirements-dev.txt
pytest -q          # 14 tests
```

## 🛡️ Security notes

- **No hardcoded secrets.** The Gemini key is read from Secret Manager (prod) or an env var
  (local). The Firebase web config in the frontend is public by design and not a secret.
- **User isolation** is enforced in two places: Firestore security rules *and* backend JWT
  verification of the `uid` on every `/entries` call.
- **Untrusted input** (journal text) is length-validated, HTML-escaped before rendering, and
  passed to Gemini strictly as data, never as instructions.

---

*Challenge verification label:* `dev-tutorial=cloud-run-ai-challenge`
*Social tag:* `#AccelerateAIwithCloudRun`
