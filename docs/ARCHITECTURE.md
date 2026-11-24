# Architecture, configuration, and deployment notes

## Current request flow

```text
Browser: React + Vite
  └─ GET /health, POST /predict → FastAPI (public application API)
       └─ GET /health, POST /predict → Flask (private model-service API)
```

The frontend is configured with `VITE_API_BASE_URL` and must call only FastAPI. `MODEL_SERVER_URL` exists only in the backend environment. The FastAPI model client uses HTTPX, validates upstream health/prediction JSON, maps Flask `_mock: true` to public `mock: true`, and enforces a configurable request timeout.

The current UI captures or uploads audio for preview. It checks backend health for explicit audio capability flags before enabling submission. The current FastAPI health schema does not expose those flags, and there is no `/predict/audio` route; consequently audio prediction remains unavailable. The separate numerical `/predict` route accepts 22 precomputed features and is not currently driven by a frontend form.

## Components and boundaries

| Component | Implemented responsibility | Important boundary |
|---|---|---|
| React/Vite | Pages, navigation, audio capture/upload preview, result rendering | No model-server URL and no direct Flask calls |
| FastAPI | Validation, CORS, health, prediction proxy, public response normalization | Contains no ML algorithm or audio feature extraction |
| Flask mock server | Feature presence/numeric checks; randomized fake result; deterministic testing endpoint | Not a trained model; binds broadly and has wildcard CORS in existing implementation |
| Flask baseline server | Loads saved tabular pipeline and proxies feature vector to `predict_single` | Separate from mock; requires artifact and is not clinically validated |
| Pathway B scripts | Exploratory audio feature/model research | Not connected to the web API |

## Environment variables

Backend settings are read by Pydantic Settings from `backend/.env` (or process environment). Environment values override file defaults.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SERVER_URL` | `http://localhost:8001` | Backend-only base URL for the Flask service |
| `MODEL_SERVER_TIMEOUT_SECONDS` | `10` | HTTPX upstream timeout, constrained to >0 and ≤120 |
| `FRONTEND_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated exact origins allowed by FastAPI CORS |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Browser-visible FastAPI base URL; Vite variables are public, never secrets |

Local examples are in `backend/.env.example` and `frontend/.env.example`. Do not commit actual `.env` files or put model service credentials/internal addresses in frontend configuration.

## Security and reliability observations

- FastAPI rejects missing, extra, non-numeric, and non-finite feature values through Pydantic and returns HTTP 422.
- Upstream connection/timeouts map to HTTP 503; invalid JSON/schema and upstream HTTP errors map to HTTP 502.
- FastAPI CORS is restricted to configured origins and GET/POST. The frontend does not need credentialed CORS; review `allow_credentials` before deployment if cookie auth is introduced.
- No API authentication, rate limiting, request-size limits, persistence, user/session management, or audit system is implemented. Do not expose publicly without an explicit security/privacy design.
- Existing Flask servers bind to `0.0.0.0`; the mock enables Flask debug mode and wildcard CORS. Do not expose these development servers to the public internet. Keep the ML service on a private network, behind network policy, and use a production WSGI server for any controlled deployment.
- The browser sends no recording while the current audio workflow is disabled. If audio processing is added, define consent, retention/deletion, encryption, upload limits, content validation, and privacy logging before collecting recordings.
- Backend application logs include request-level HTTP logs from Uvicorn/HTTPX. Prediction feature values are not intentionally logged by the app. Configure access-log retention and avoid logging raw audio, identifiers, or health-related payloads.

## Deployment recommendations (not deployment validation)

These are target patterns; this repository has no deployment manifests, hosting configuration, TLS setup, or production smoke-test evidence.

1. **Frontend:** build with `npm ci && npm run build`, host static `frontend/dist` on a CDN/static host with TLS. Set `VITE_API_BASE_URL` at build time to the public FastAPI URL. Because it is compiled into the bundle, it is not secret. Configure SPA fallback routing for React Router.
2. **FastAPI:** deploy as a container/service on a private or controlled network, terminate TLS at a trusted ingress, configure exact `FRONTEND_ORIGINS`, `MODEL_SERVER_URL` using private service discovery, and `MODEL_SERVER_TIMEOUT_SECONDS`. Use a managed secret store for any future credentials. Run multiple workers only after load/resource behavior is measured.
3. **Flask ML service:** deploy separately on a private subnet/service network, with an explicit artifact mount/version, least-privilege identity, liveness/readiness probes, resource limits, and a production WSGI server. Do not use Flask development server, debug mode, public wildcard CORS, or a public tunnel as production architecture.
4. **Operations:** add dependency/security scanning, pinned/reproducible lockfiles, CI, structured redacted logs, metrics, alerting, backups only for required data, and a rollback strategy. Define data governance and clinical/research review separately from software deployment.
5. **Model changes:** keep the public FastAPI response contract stable. A replacement model should implement the same private health and prediction contract, preserve explicit model/version/mock metadata, and pass contract tests before changing the backend environment URL.

Passing local unit tests is not deployment, security, or clinical validation.
