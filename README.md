# 🧠 Parkinson's AI Screening System

An AI-powered screening tool that uses vocal biomarkers to detect Parkinson's Disease. The system analyses acoustic features extracted from sustained vowel recordings and classifies them using a machine learning pipeline trained on clinically validated datasets.

> **This README is for the entire team.** Whether you're working on the model, backend, or frontend — start here.

---

## 📋 Table of Contents

- [What We're Building](#what-were-building)
- [Team Structure & Ownership](#team-structure--ownership)
- [System Architecture](#system-architecture)
- [API Contract — Read This First](#api-contract--read-this-first)
- [Repository Structure](#repository-structure)
- [How the Team Connects](#how-the-team-connects)
- [Git Workflow](#git-workflow)
- [Getting Started by Role](#getting-started-by-role)
- [ML Model — Context & Progress](#ml-model--context--progress)
- [Environment Setup](#environment-setup)
- [FAQ](#faq)

---

## What We're Building

A web application where a clinician or researcher can:

1. **Upload or record** a sustained vowel sound (e.g., saying "ahhh")
2. **Submit** it to the system
3. **Receive** an AI-generated screening result — whether the voice shows acoustic markers consistent with Parkinson's Disease — along with a confidence score and explanation

> **Important:** This is a screening aid, not a diagnostic tool. The model flags risk; a clinician makes the final call.

### Why Voice?

Parkinson's affects 90% of patients' voices before motor symptoms are obvious. Vocal biomarkers like jitter, shimmer, and harmonic-to-noise ratio (HNR) are measurable, non-invasive indicators used in clinical research.

---

## Team Structure & Ownership

| Role | Owns | Does NOT own |
|------|------|-------------|
| **ML Engineer** (Mohan) | `ml/` — model training, inference server | Frontend UI, backend routing |
| **Backend Developer** | `backend/` — API layer, routing, proxy to model | ML model internals, frontend UI |
| **Frontend Developer** | `frontend/` — UI, forms, results display | ML model, backend logic |

### The golden rule
> **Never reach across your boundary without agreeing first.** If the frontend needs a new field in the response, the ML engineer and backend dev agree on the contract change — then everyone updates their side.

---

## System Architecture

```
┌─────────────────────┐         ┌──────────────────────┐         ┌───────────────────────────┐
│                     │         │                       │         │                           │
│   Frontend          │─────────▶   Backend             │─────────▶   ML Model Server         │
│   (React / Vue)     │  HTTP   │   (FastAPI / Flask)   │  HTTP   │   (Flask, runs locally    │
│                     │  POST   │                       │  POST   │    on ML Engineer's PC)   │
│   - Upload audio    │         │   - Validates request │         │                           │
│   - Show results    │◀────────│   - Calls model server│◀────────│   - Loads pipeline        │
│   - Confidence viz  │  JSON   │   - Returns response  │  JSON   │   - Returns prediction    │
│                     │         │                       │         │                           │
└─────────────────────┘         └──────────────────────┘         └───────────────────────────┘

                                                                         ↑
                                                                 Exposed via ngrok
                                                                 during development
                                                                 (public HTTPS URL)
```

### Why this split?

- **Frontend** never talks directly to the ML model — it talks to the backend. This means if the model URL changes (e.g., different ngrok URL), only the backend config changes.
- **Backend** acts as a stable middle layer. It can add auth, logging, caching, or rate limiting without touching the model or frontend.
- **ML model** runs locally during development and can be deployed independently later.

---

## API Contract — Read This First

> [!IMPORTANT]
> **All three team members must agree on this contract before writing code.** If you need to change it, open a PR and discuss — don't change unilaterally.

### Endpoint: `POST /predict`

This is the only endpoint the frontend needs to know about. It lives on the **backend**.

#### Request

```json
{
  "MDVP:Fo(Hz)":       119.992,
  "MDVP:Fhi(Hz)":      157.302,
  "MDVP:Flo(Hz)":       74.997,
  "MDVP:Jitter(%)":     0.00784,
  "MDVP:Jitter(Abs)":   0.00007,
  "MDVP:RAP":           0.00370,
  "MDVP:PPQ":           0.00554,
  "Jitter:DDP":         0.01109,
  "MDVP:Shimmer":       0.04374,
  "MDVP:Shimmer(dB)":   0.42600,
  "Shimmer:APQ3":       0.02182,
  "Shimmer:APQ5":       0.03130,
  "MDVP:APQ":           0.02971,
  "Shimmer:DDA":        0.06545,
  "NHR":                0.02211,
  "HNR":               21.03300,
  "RPDE":               0.41478,
  "DFA":                0.81529,
  "spread1":           -4.81303,
  "spread2":            0.26648,
  "D2":                 2.30144,
  "PPE":                0.28465
}
```

> These 22 features are acoustic measurements extracted from a voice recording. The backend receives the audio file, the **backend or a future audio-processing step** extracts these features, then sends them to the ML server.

#### Response (Success `200 OK`)

```json
{
  "prediction":           1,
  "label":                "PARKINSONS",
  "confidence":           0.9746,
  "pd_probability":       0.9746,
  "healthy_probability":  0.0254,
  "model_version":        "0.1.0-baseline"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `int` | `1` = Parkinson's, `0` = Healthy |
| `label` | `string` | Human-readable label |
| `confidence` | `float` | Probability of the predicted class (0–1) |
| `pd_probability` | `float` | Raw probability of Parkinson's |
| `healthy_probability` | `float` | Raw probability of Healthy |
| `model_version` | `string` | Model artifact version for traceability |

#### Response (Error `422 Unprocessable Entity`)

```json
{
  "error": "Missing required features",
  "missing": ["HNR", "RPDE"]
}
```

#### Response (Error `503 Service Unavailable`)

```json
{
  "error": "Model server unreachable",
  "detail": "Could not connect to http://localhost:8001"
}
```

---

### Internal Endpoint: `POST /model/predict` (Backend → ML Server)

The backend calls the ML server at this internal endpoint. **Frontend devs don't need to touch this.**

- **During development:** `http://<ngrok-url>/predict` (ML engineer shares this URL)
- **During production:** `http://ml-server:8001/predict` (once deployed)

The request and response shapes are identical to the public `/predict` endpoint above.

---

## Repository Structure

```
parkinsons-ai-screening/
│
├── README.md                        ← You are here
├── .gitignore
│
├── docs/
│   └── api_contract.md              ← Detailed API spec (source of truth)
│
├── frontend/                        ← Frontend Developer owns this
│   └── (React / Vue / your choice)
│
├── backend/                         ← Backend Developer owns this
│   └── (FastAPI / Flask / your choice)
│
└── ml/                              ← ML Engineer (Mohan) owns this
    ├── pathway_a/                   ← Stage 1: UCI tabular features (DONE ✅)
    │   ├── data/
    │   │   └── raw/                 ← parkinsons.data (gitignored)
    │   ├── models/
    │   │   ├── baseline_pipeline.joblib   ← trained model (gitignored)
    │   │   └── baseline_metrics.json      ← metrics report (tracked)
    │   ├── train_baseline.py        ← trains the model
    │   ├── predict.py               ← inference contract (predict_single, predict_batch)
    │   └── server.py                ← Flask server exposing /predict
    │
    └── pathway_b/                   ← Stage 2: Raw audio → acoustic features (IN PROGRESS 🔄)
        ├── data/
        │   ├── raw/italian_pd/      ← .wav files (gitignored)
        │   └── processed/           ← features.csv (gitignored)
        ├── models/
        ├── plots/
        ├── extract_features.py      ← .wav → acoustic features
        ├── train_compare.py         ← 5-fold CV: LR vs RF vs XGBoost vs SVM
        └── explain_shap.py          ← SHAP explainability plots
```

---

## How the Team Connects

### Phase 1 — Right Now: Use the Mock Server (No Mohan needed)

> [!IMPORTANT]
> **Backend and frontend devs: start here.** You do NOT need Mohan's machine or the real model to begin development. Run the mock server yourself locally.

**Step 1 — Clone the repo and install dependencies:**
```bash
git clone <repo-url>
cd parkinsons-ai-screening
pip install flask flask-cors
```

**Step 2 — Run the mock server:**
```bash
python ml/pathway_a/mock_server.py
```
```
🟡 MOCK MODEL SERVER — for development only
  POST http://localhost:8001/predict
  GET  http://localhost:8001/health
  POST http://localhost:8001/predict/deterministic
```

**Step 3 — Set your backend env var to point at it:**
```bash
# backend/.env
MODEL_SERVER_URL=http://localhost:8001
```

That's it. The mock server:
- ✅ Validates all 22 required features (returns `422` if any are missing — same as real model)
- ✅ Returns a realistic random prediction (`_mock: true` flag in the response)
- ✅ Has a `/predict/deterministic` endpoint that always returns the same response — use this for writing tests
- ✅ Has a `/health` endpoint to confirm it's running

**Mock response includes `"_mock": true`** — your frontend can use this to show a "⚠️ Mock Mode" banner during development if you want.

---

### Phase 2 — Later Today: Switch to Mohan's Real Model

When Mohan is ready, he will:
1. Start `python ml/pathway_a/server.py` on his machine
2. Run `ngrok http 8001` and share the public URL in team chat

You just update **one line** in your `.env`:
```bash
# backend/.env  — only this line changes, nothing else
MODEL_SERVER_URL=https://abc123.ngrok-free.app
```

Restart your backend server — done. The API contract is identical.

> [!NOTE]
> The ngrok URL changes each session. When Mohan shares a new URL, update `.env` and restart the backend. No code changes ever needed.

---

### Phase 3 — Production (Later)

Once deployed, `MODEL_SERVER_URL` becomes the cloud URL. Same swap, same process.

---

### During Development (Mohan's machine)

**What Mohan does:**
```bash
# 1. Start the model server
python ml/pathway_a/server.py
# → Running on http://0.0.0.0:8001

# 2. In a second terminal, expose it publicly
ngrok http 8001
# → Forwarding: https://abc123.ngrok-free.app → localhost:8001

# 3. Share the ngrok URL in the team chat
```

**What the backend dev does:**

Set the model URL as an environment variable — never hardcode it:
```bash
# .env file (never commit this)
MODEL_SERVER_URL=https://abc123.ngrok-free.app
```

```python
# backend/config.py
import os
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")
```

```python
# backend/routes/predict.py — example proxy call
import httpx, os

async def call_model(features: dict) -> dict:
    url = f"{os.getenv('MODEL_SERVER_URL')}/predict"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, json=features)
        response.raise_for_status()
        return response.json()
```

> [!NOTE]
> The ngrok URL changes every time Mohan restarts the tunnel (unless using a paid static domain). **Post the new URL in team chat each session.**

---

### During Production (Later)

Once we're ready to deploy:
- ML model → hosted on a cloud VM or container
- Backend → deployed to cloud (Railway / Render / AWS / etc.)
- Frontend → deployed to Vercel / Netlify / etc.
- The `MODEL_SERVER_URL` env var in the backend just changes to the deployed URL — **nothing else changes**

---

## Git Workflow

### Branches

```
main          ← stable, always working, demo-ready
feat/*        ← your feature branches
fix/*         ← bug fixes
```

### Rules

1. **Never commit directly to `main`**
2. Work on a feature branch, open a Pull Request, get at least one review
3. Keep PRs small — one feature at a time
4. Write a clear commit message (see below)

### Starting your work

```bash
# Clone the repo (first time)
git clone <repo-url>
cd parkinsons-ai-screening

# Start a new feature
git checkout main
git pull origin main
git checkout -b feat/your-feature-name

# Work... commit often...
git add .
git commit -m "feat(frontend): add prediction result card component"

# Push and open a PR
git push origin feat/your-feature-name
```

### Commit message format

```
<type>(<scope>): <short description>

Types:  feat | fix | docs | refactor | test | chore
Scopes: frontend | backend | ml | docs | ci
```

Examples:
```
feat(frontend): add audio upload form with drag-and-drop
feat(backend): add /predict proxy endpoint with error handling
fix(ml): handle NaN in feature extraction for silent audio
docs: update API contract with new error codes
```

---

## Getting Started by Role

### 🎨 Frontend Developer

**Your job:** Build the UI where users upload/record audio and see results.

**You need to know:**
- The backend will be running at some local URL (e.g., `http://localhost:8000`)
- Call `POST /predict` with the 22 feature values (the backend handles extraction)
- Display `label`, `confidence`, `pd_probability`, `healthy_probability` from the response

**To test without the backend running,** mock the response:
```json
{
  "prediction": 1,
  "label": "PARKINSONS",
  "confidence": 0.9746,
  "pd_probability": 0.9746,
  "healthy_probability": 0.0254,
  "model_version": "0.1.0-baseline"
}
```

**Setup:**
```bash
cd frontend/
# follow frontend-specific README (to be added by frontend dev)
```

---

### ⚙️ Backend Developer

**Your job:** Build the API layer between the frontend and the ML model.

**You need to know:**
- Receive requests from frontend at `POST /predict`
- For now, the input is the 22 feature values (later: accept audio file and extract features)
- Forward to ML server URL (from `MODEL_SERVER_URL` env var)
- Return the response unchanged, or add extra fields if needed

**Key env variable:**
```bash
MODEL_SERVER_URL=https://<ngrok-url>  # get this from Mohan each session
```

**To test without the ML server running,** mock the internal call and return the sample response above.

**Setup:**
```bash
cd backend/
# follow backend-specific README (to be added by backend dev)
```

---

### 🤖 ML Engineer (Mohan)

**Your job:** Train and serve the ML model.

**Setup:**
```bash
cd parkinsons-ai-screening/

# Install ML dependencies
pip install scikit-learn joblib pandas flask flask-cors pyngrok \
            praat-parselmouth librosa xgboost shap soundfile antropy

# Verify Stage 1 is working
python ml/pathway_a/predict.py

# Start the model server
python ml/pathway_a/server.py
# → http://0.0.0.0:8001/predict

# In another terminal, expose it
ngrok http 8001
```

**Current model metrics (Stage 1 baseline):**

| Metric | Value |
|--------|-------|
| Recall (Sensitivity) | **0.7742** |
| Precision | **0.8571** |
| F1 Score | **0.8136** |
| ROC-AUC | **0.7151** |
| Specificity | 0.3333 |

> Recall is the primary metric — missing a PD case is high-cost.

---

## ML Model — Context & Progress

### Dataset
- **UCI Parkinson's Dataset** (`parkinsons.data`) — 195 recordings, 32 participants (24 PD, 8 Healthy)
- 22 acoustic features per recording (MDVP jitter, shimmer, HNR, nonlinear dynamics)
- Raw data is **gitignored** — contact Mohan or download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/174/parkinsons)

### Stage 1 — ✅ Complete
- Logistic Regression with `class_weight=balanced`
- `StratifiedGroupKFold` split (participant-level, prevents speaker leakage)
- Model artifact: `ml/pathway_a/models/baseline_pipeline.joblib`
- Inference contract: `ml/pathway_a/predict.py` → `predict_single(features)` and `predict_batch(df)`

### Stage 2 — 🔄 In Progress
- Raw `.wav` audio → acoustic feature extraction (Praat + librosa)
- 5-fold cross-validated comparison: LR vs Random Forest vs XGBoost vs SVM
- SHAP explainability (which features drove the prediction)
- Dataset: Italian Parkinson's Voice & Speech dataset (Dimauro et al.)
- Scripts: `ml/pathway_b/extract_features.py`, `train_compare.py`, `explain_shap.py`

### Future Stages (Planned)
- Stage 3: Accept raw audio from frontend, extract features server-side
- Stage 4: Clinical explainability report (SHAP waterfall per patient)
- Stage 5: Model versioning and A/B testing infrastructure

---

## Environment Setup

### Prerequisites (all roles)
- Git
- Python 3.10+ (ML and Backend)
- Node.js 18+ (Frontend)

### Python environment (ML + Backend)
```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies (ML)
pip install scikit-learn joblib pandas flask flask-cors \
            praat-parselmouth librosa xgboost shap soundfile antropy pyngrok
```

### Environment variables

Create a `.env` file in the backend directory. **Never commit `.env` to git.**

```bash
# backend/.env
MODEL_SERVER_URL=https://<ngrok-url-from-mohan>
PORT=8000
DEBUG=true
```

---

## FAQ

**Q: The ngrok URL changed and the backend can't reach the model. What do I do?**
> Ask Mohan to share the new ngrok URL. Update `MODEL_SERVER_URL` in your `.env`. No code changes needed.

**Q: I want to test my part without waiting for another team member's code. Can I?**
> Yes. Use the mock response JSON above to simulate responses. Backend devs: mock the `call_model()` function. Frontend devs: mock the fetch call.

**Q: Where are the actual `.wav` files and trained model?**
> Raw data and trained model binaries are gitignored (too large, sensitive). The metrics JSON and training scripts are tracked. Contact Mohan for the model file directly.

**Q: What's the difference between `pathway_a` and `pathway_b`?**
> `pathway_a` uses pre-extracted tabular features (the UCI dataset). `pathway_b` starts from raw `.wav` audio and extracts features itself — this is the path to accepting real audio from the frontend.

**Q: How do I run a quick test of the model API?**
```bash
curl -X POST https://<ngrok-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, "MDVP:Flo(Hz)": 74.997, "MDVP:Jitter(%)": 0.00784, "MDVP:Jitter(Abs)": 0.00007, "MDVP:RAP": 0.00370, "MDVP:PPQ": 0.00554, "Jitter:DDP": 0.01109, "MDVP:Shimmer": 0.04374, "MDVP:Shimmer(dB)": 0.426, "Shimmer:APQ3": 0.02182, "Shimmer:APQ5": 0.03130, "MDVP:APQ": 0.02971, "Shimmer:DDA": 0.06545, "NHR": 0.02211, "HNR": 21.033, "RPDE": 0.414783, "DFA": 0.815285, "spread1": -4.813031, "spread2": 0.266482, "D2": 2.301442, "PPE": 0.284654}'
```

**Q: Who do I contact if I'm blocked?**
> Post in the team chat with exactly what you tried, what you expected, and what happened. Don't stay blocked for more than 30 minutes without asking.
