# FastAPI API contract

Base URL for local development: `http://localhost:8000`. Interactive schema: `/docs`; raw OpenAPI document: `/openapi.json`.

## `GET /health`

Performs a best-effort request to the configured Flask `GET /health` endpoint. The backend remains live even if Flask is unavailable.

Example when reachable:

```json
{
  "status": "ok",
  "service": "parkinsons-screening-backend",
  "model_server": "reachable",
  "model_server_detail": null
}
```

`model_server` is `reachable` or `unreachable` (schema also permits `unknown`). An unreachable model is communicated in the body; it does not change the HTTP 200 liveness response. This health API does not advertise audio processing capability.

## `POST /predict`

Accepts `application/json` with exactly these 22 feature keys as numerical values. Extra fields are forbidden; missing fields, invalid types, and non-finite numbers produce HTTP 422. This route accepts precomputed features only. It does not accept multipart audio.

```json
{
  "MDVP:Fo(Hz)": 119.992,
  "MDVP:Fhi(Hz)": 157.302,
  "MDVP:Flo(Hz)": 74.997,
  "MDVP:Jitter(%)": 0.00784,
  "MDVP:Jitter(Abs)": 0.00007,
  "MDVP:RAP": 0.0037,
  "MDVP:PPQ": 0.00554,
  "Jitter:DDP": 0.01109,
  "MDVP:Shimmer": 0.04374,
  "MDVP:Shimmer(dB)": 0.426,
  "Shimmer:APQ3": 0.02182,
  "Shimmer:APQ5": 0.0313,
  "MDVP:APQ": 0.02971,
  "Shimmer:DDA": 0.06545,
  "NHR": 0.02211,
  "HNR": 21.033,
  "RPDE": 0.41478,
  "DFA": 0.81529,
  "spread1": -4.81303,
  "spread2": 0.26648,
  "D2": 2.30144,
  "PPE": 0.28465
}
```

Example successful public response (values vary with the mock server):

```json
{
  "prediction": 1,
  "label": "PARKINSONS",
  "confidence": 0.8,
  "pd_probability": 0.8,
  "healthy_probability": 0.2,
  "model_version": "0.0.0-mock",
  "mock": true,
  "disclaimer": "Research prototype only; this output is not a diagnosis."
}
```

`confidence` and class probabilities are model-server fields. In mock mode these are simulated values, not meaningful evidence or medical certainty. Optional `feature_count` is emitted only when supplied by the model server. The private `_mock` property is normalized to public `mock`.

## Error behavior

| HTTP | Meaning |
|---|---|
| `422` | Request body missing/invalid/extra feature fields (FastAPI validation) |
| `503` | ML server is unreachable or request timed out |
| `502` | ML server returned an HTTP error, invalid JSON, or a response outside the public prediction schema |

Error bodies use FastAPI’s `{"detail": "..."}` shape, except request validation which includes structured `detail` entries. Internal upstream paths are not public API routes.

## Private Flask contract (backend to ML service)

- `GET /health` expects JSON containing `{"status":"ok"}`.
- `POST /predict` receives the same 22 feature-key JSON object and returns prediction fields (`prediction`, `label`, `confidence`, `pd_probability`, `healthy_probability`, `model_version`); the mock also sets `_mock: true`.
- Mock-only `POST /predict/deterministic` returns a stable test result but the FastAPI backend does not expose or call this endpoint.
- No `/predict/audio` endpoint is currently implemented.

## Example request

Save the request object above as `features.json`, then run:

```bash
curl -sS http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @features.json
```
