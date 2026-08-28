# MindMap Journal 🧠

An AI journaling app with a **real, self-trained ML emotion classifier** — built for the
[Cloud Run "Accelerate AI" Challenge](https://codelabs.developers.google.com/codelabs/cloud-run/cloud-run-ai-challenge).

Unlike a plain Gemini chatbot wrapper, MindMap Journal trains its **own** text-emotion
model (TF-IDF + Logistic Regression on the public `dair-ai/emotion` dataset), serves it on
Cloud Run, and uses it to auto-tag every journal entry. Gemini then provides a reflective
response, and a mood-over-time dashboard visualizes your emotional trends.

## ✨ Features

- 🔐 **Google Sign-In** via Firebase Authentication (no passwords stored)
- 🤖 **Self-trained ML model** classifies entry emotion (`/predict`)
- 💬 **Gemini reflection** with a resilient model-fallback ladder (`/reflect`)
- 📊 **Mood-over-time dashboard** built from your accumulated entries
- 🔒 **User-isolated Firestore** — you can only ever read/write your own data
- 🗝️ **Secret Manager** for the Gemini API key (zero hardcoded secrets)
- ☁️ **One-container deploy** to Google Cloud Run

## 🏗️ Architecture

```
Browser (Firebase Auth: Google Sign-In)
        │
        ▼
Cloud Run service (FastAPI, single container)
  ├── /predict  → self-trained emotion model (scikit-learn)
  ├── /reflect  → Gemini API (fallback ladder, key from Secret Manager)
  └── /entries  → Cloud Firestore (user-isolated security rules)
```

## 🚧 Status

Work in progress. Deployment guide, ML training instructions, and challenge label steps
are documented further down as the project is built out.

---
*Challenge label:* `dev-tutorial=cloud-run-ai-challenge`
