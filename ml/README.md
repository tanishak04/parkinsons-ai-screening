# ML service and research workflows

`pathway_a/mock_server.py` is the Flask service used for the local application demonstration. It checks for the 22 required feature keys and numeric-convertible values, then returns a randomized simulated result with `_mock: true`. Its `/predict/deterministic` route returns a fixed test response. **Neither route performs inference.**

## Run the mock service

From the repository root, install its direct lightweight dependencies and start it:

```bash
python -m pip install Flask flask-cors
python ml/pathway_a/mock_server.py
```

It listens on port 8001 and exposes `GET /health`, `POST /predict`, and `POST /predict/deterministic`. This existing development server binds to `0.0.0.0`, enables Flask debug mode, and allows wildcard CORS; keep it local to a trusted development environment and do not expose it to the internet.

## Optional baseline model server

`pathway_a/server.py` is separate from the mock and attempts to load `ml/pathway_a/models/baseline_pipeline.joblib` on startup. That artifact is not included in this checkout. The model can only be started after a compatible artifact exists; the training script expects `ml/pathway_a/data/raw/parkinsons.data` and uses pandas, scikit-learn, and joblib. Install those dependencies in the project environment and obtain the dataset/artifact through the project’s approved research data process. The server’s existence and repository metrics do not establish clinical validity.

Do not treat either the baseline or pathway B scripts as audio support in the web application: FastAPI currently forwards precomputed features and exposes no `/predict/audio` endpoint. See [API contract](../docs/API.md) and [architecture/deployment notes](../docs/ARCHITECTURE.md).
