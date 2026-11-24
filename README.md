# Parkinson’s AI Screening System

An early-stage research software prototype exploring acoustic voice features. This repository is **not a diagnostic system**, is not clinically validated, and must not be used to make health or treatment decisions.

> **Demo status:** the connected Flask service is a mock. Its predictions are simulated and do not depend on the submitted feature values. The React app supports recording and previewing audio, but the current backend does not extract features from audio. Audio submission is therefore disabled.

## Architecture

```text
React / Vite (browser) ──HTTP──▶ FastAPI ──HTTP──▶ Flask ML server
        port 5173                 port 8000           port 8001
```

The browser knows only the FastAPI URL. The backend validates a JSON object containing 22 precomputed acoustic features, proxies it to Flask, and normalizes the response. Flask URLs and model choice are configured on the backend with `MODEL_SERVER_URL`.

For details, see [architecture and limitations](docs/ARCHITECTURE.md), [API contract](docs/API.md), and the service setup guides for [backend](backend/README.md), [frontend](frontend/README.md), and [ML server](ml/README.md).

## Quick start

Use three terminals from the repository root. Python commands below assume a virtual environment at `.venv`.

```bash
# Terminal 1: development mock server
python -m pip install Flask flask-cors
python ml/pathway_a/mock_server.py
```

```bash
# Terminal 2: FastAPI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Terminal 3: React
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Open <http://localhost:5173>. FastAPI Swagger docs are at <http://localhost:8000/docs> and ReDoc at <http://localhost:8000/redoc>.

## Configure

Copy `backend/.env.example` to `backend/.env`; copy `frontend/.env.example` to `frontend/.env` only if the API URL differs from its local default. See [environment variables](docs/ARCHITECTURE.md#environment-variables). Do not put the ML URL or secrets in `VITE_*` variables: Vite exposes them in browser bundles.

## API overview

- `GET /health` — backend liveness plus best-effort ML-server connectivity. An unreachable ML service is reported in the JSON body; this endpoint still returns HTTP 200.
- `POST /predict` — accepts exactly the 22 numeric features listed in [API documentation](docs/API.md). It does **not** accept an audio file.
- `GET /docs` — interactive OpenAPI documentation.

Use the deterministic mock endpoint directly only for local integration checks: `POST http://localhost:8001/predict/deterministic`. It is not exposed by FastAPI as a public route.

## Tests

```bash
# Backend
cd backend
python -m pip install -r requirements.txt
pytest -q

# Frontend
cd ../frontend
npm ci
npm test
npm run build
```

Backend unit/API tests mock HTTPX upstream calls and do not require a running Flask service. The frontend tests use a simulated browser environment; they do not prove physical-device microphone behavior or clinical validity.

## Project layout

```text
backend/       FastAPI routes, schemas, settings, HTTPX model client, tests
frontend/      React pages/components, centralized API client, Vitest tests
ml/pathway_a/  tabular-feature baseline scripts, Flask mock and baseline servers
ml/pathway_b/  experimental audio feature extraction and model comparison scripts
docs/          architecture and API contract
```

## Status and limitations

- The connected `mock_server.py` generates randomized simulated results; `_mock: true` is surfaced publicly as `mock: true`.
- The feature-vector route forwards precomputed features. No audio-to-feature pipeline is wired into FastAPI, and no `/predict/audio` endpoint exists.
- `server.py` is a separate baseline-model Flask server and requires a locally available `baseline_pipeline.joblib` artifact; that artifact is not included in this repository checkout. Its existence does not establish clinical validity.
- Stage 2 audio extraction scripts are research work and are not integrated into the web request path.
- The UI and API are demonstration/research software. They are not intended to diagnose Parkinson’s disease, guide treatment, or replace a qualified clinician.

Before a real-model demonstration, verify the model server’s provenance, expected feature order/units, artifact availability, output contract, and deployment isolation. Before enabling audio submission, implement and validate server-side extraction and confirm compatibility with the selected model.
