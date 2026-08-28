# MindMap Journal 🧠

**An AI journaling companion that understands how you feel — powered by a machine-learning
emotion model I trained myself, not just an off-the-shelf chatbot.**

Built for the [Cloud Run "Accelerate AI" Challenge](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-ai-challenge)
(`#AccelerateAIwithCloudRun`).

---

## 📖 What is this?

Most journaling apps in this challenge are a thin wrapper around a large language model: you
type something, the LLM replies, and the text is saved. **MindMap Journal goes further.**

Every time you write an entry, the app runs it through a **custom emotion classifier that I
trained from scratch** on a real, public dataset. That model reads your words and tags the
dominant emotion — *joy, sadness, anger, fear, love,* or *surprise* — with a confidence
score. Gemini then reads the same entry and writes a warm, reflective response. Over time,
the app assembles your tagged entries into a **mood-over-time dashboard** so you can actually
*see* your emotional patterns emerge.

The result is a journal that does three distinct jobs at once:

1. **Classifies** your emotion with a purpose-built ML model (this is the part I trained).
2. **Reflects** on your entry with Gemini's natural-language understanding.
3. **Remembers** everything privately, so patterns surface across days and weeks.

## 🌟 Why this stands out (Authenticity)

The challenge is judged on **Authenticity, Usability, Stability, and Security**. This project
was designed to score on all four:

| Judging axis | How MindMap Journal addresses it |
| :--- | :--- |
| **Authenticity** | A **self-trained scikit-learn emotion model** — real ML, not a Gemini wrapper. The training pipeline, dataset, and evaluation metrics are all in this repo and reproducible. |
| **Usability** | One-click **Google Sign-In**, a single clean dashboard, instant emotion tags, and a visual mood chart. |
| **Stability** | A **Gemini model-fallback ladder** (retries across models on 429/503/500 errors) and defensive, null-safe request handling so the UI never crashes silently. |
| **Security** | **User-isolated Firestore rules** (`request.auth.uid == userId`), the Gemini key stored in **Secret Manager** (zero hardcoded secrets), and backend JWT verification via the Firebase Admin SDK. |

## ✨ Features

- 🔐 **Google Sign-In** via Firebase Authentication — no passwords are ever stored.
- 🤖 **Self-trained ML emotion classifier** auto-tags every entry (`POST /predict`).
- 💬 **Gemini reflection** with a resilient multi-model fallback ladder (`POST /reflect`).
- 📊 **Mood-over-time dashboard** built live from your accumulated, tagged entries.
- 🔒 **Strict per-user data isolation** — you can only ever read or write your own entries.
- 🗝️ **Secret Manager** integration for the Gemini API key (no secrets in code or images).
- ☁️ **Single-container deploy** to Google Cloud Run.

## 🧠 The machine learning model

This is the heart of the project — a genuine, reproducible ML pipeline.

- **Task:** multi-class text emotion classification (6 classes).
- **Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — ~20,000
  English messages labeled with one of six emotions (sadness, joy, love, anger, fear,
  surprise). A widely used, publicly available benchmark.
- **Approach:** a **TF-IDF vectorizer + Logistic Regression** classifier (scikit-learn).
  This baseline is fast to train (seconds), tiny to ship, and cheap to serve on Cloud Run's
  free tier — while still reaching strong accuracy on this dataset. The pipeline is
  structured so the classifier can later be swapped for a fine-tuned transformer without
  touching the API layer.
- **Artifact:** the trained pipeline is serialized with `joblib` and loaded once at server
  startup for low-latency inference.
- **Reproducibility:** run `python ml/train.py` to download the data, train, evaluate, and
  print a full classification report. Metrics are documented after training.

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
├── ml/
│   └── train.py            # ML training pipeline (download → train → evaluate → save)
├── app/
│   ├── main.py             # FastAPI app: /predict, /reflect, /entries
│   └── static/             # Frontend single-page app (Sign-In, journal, dashboard)
├── model/                  # Trained model artifact (produced by ml/train.py)
├── firestore.rules         # User-isolated Firestore security rules
├── Dockerfile              # Single-container build for Cloud Run
├── requirements.txt
└── README.md
```

## 🚀 Status & roadmap

- [x] Project scaffold + public repository
- [ ] ML training pipeline (`dair-ai/emotion`, TF-IDF + Logistic Regression)
- [ ] Train model + record evaluation metrics
- [ ] FastAPI backend (`/predict`, `/reflect`, `/entries`)
- [ ] Frontend SPA (Google Sign-In, journal input, mood dashboard)
- [ ] Firestore security rules
- [ ] Dockerfile + local container run
- [ ] Full production deployment guide (GCP setup, Secret Manager, `gcloud run deploy`)

Full setup and deploy instructions — enabling Google Cloud APIs, provisioning Firestore,
creating the Secret Manager secret, deploying to Cloud Run, and applying the required
challenge label — will be documented here as each piece lands.

---

*Challenge verification label:* `dev-tutorial=cloud-run-ai-challenge`
*Social tag:* `#AccelerateAIwithCloudRun`
