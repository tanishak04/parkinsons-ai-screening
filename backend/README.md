# FastAPI backend

## Setup and run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Start the Flask mock server in a separate terminal first (see [ML server setup](../ml/README.md)), then:

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI or `http://localhost:8000/redoc` for ReDoc.

## Configuration

See `.env.example` and [environment variable documentation](../docs/ARCHITECTURE.md#environment-variables). `MODEL_SERVER_URL` is server-side configuration; do not expose it through a Vite variable.

## API and behavior

- `GET /health`: FastAPI liveness and best-effort model-server reachability.
- `POST /predict`: validated JSON with 22 precomputed numerical features. Audio is not supported.
- Input validation is provided by Pydantic; proxy behavior is isolated in `app/services/model_client.py`.
- Upstream unreachable/timeout returns 503; invalid upstream prediction response returns 502.

Full field list and examples: [API contract](../docs/API.md).

## Tests

From `backend/` with dependencies installed:

```bash
pytest -q
```

Tests use an HTTPX mock transport for upstream calls, so the Flask server does not need to run.
