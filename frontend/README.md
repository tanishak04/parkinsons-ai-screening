# React frontend

## Setup and run

```bash
cd frontend
npm ci
cp .env.example .env # optional for local defaults
npm run dev -- --host 127.0.0.1 --port 5173
```

The dev server defaults to `http://localhost:5173`. `VITE_API_BASE_URL` in `.env` configures the FastAPI URL (default `http://localhost:8000`). Variables prefixed `VITE_` are bundled into browser code and must never contain secrets or the internal ML server URL.

## Current user flows

- Home, screening, methodology, about, results, and not-found routes are implemented.
- Screening supports WAV upload checks, preview, and browser recording/preview where supported.
- The current backend has no audio processing endpoint. The UI checks for explicit backend audio capability before enabling submission; with the current backend the audio submit control remains disabled.
- Results display the actual response passed through navigation state. There is no current frontend form for submitting the numerical 22-feature API request.

## Checks

```bash
npm test
npm run build
```

See [API behavior and limitations](../docs/API.md) and [architecture notes](../docs/ARCHITECTURE.md).
