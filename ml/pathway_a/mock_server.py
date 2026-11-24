"""
ml/pathway_a/mock_server.py
━━━━━━━━━━━━━━━━━━━━━━━━━
A MOCK model server for frontend/backend development.
Returns realistic fake predictions — no trained model needed.

HOW TO RUN:
    pip install flask flask-cors
    python ml/pathway_a/mock_server.py

The server starts at: http://localhost:8001
Send a POST to:       http://localhost:8001/predict

THIS IS NOT THE REAL MODEL. It is a development stub.
When Mohan shares the real server URL, replace MODEL_SERVER_URL
in your backend .env — no other changes needed.
"""

import random
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from any origin during development

# The 22 features the real model expects
REQUIRED_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

MODEL_VERSION = "0.0.0-mock"


@app.route("/health", methods=["GET"])
def health():
    """Health check — use this to verify the server is running."""
    return jsonify({
        "status": "ok",
        "server": "mock",
        "model_version": MODEL_VERSION,
        "message": "This is a MOCK server. Real predictions are not being made.",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Mock /predict endpoint.

    Validates that all 22 required features are present,
    then returns a realistic-looking fake prediction.

    Behaviour:
      - If ANY required feature is missing → 422 error (same as real model)
      - If all features present → random prediction weighted 75% PD (matches dataset ratio)
      - Adds a small artificial delay (50ms) to simulate real inference
    """
    data = request.get_json(silent=True)

    # --- Validate content type ---
    if data is None:
        return jsonify({
            "error": "Invalid request",
            "detail": "Request body must be JSON with Content-Type: application/json",
        }), 400

    # --- Validate all required features are present ---
    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing": missing,
            "expected_count": len(REQUIRED_FEATURES),
            "received_count": len(data),
        }), 422

    # --- Validate all values are numeric ---
    invalid = []
    for f in REQUIRED_FEATURES:
        try:
            float(data[f])
        except (TypeError, ValueError):
            invalid.append(f)
    if invalid:
        return jsonify({
            "error": "Non-numeric feature values",
            "invalid_fields": invalid,
        }), 422

    # --- Simulate inference delay ---
    time.sleep(0.05)

    # --- Generate mock prediction ---
    # Weighted random: 75% PD, 25% Healthy (reflects dataset class imbalance)
    is_pd = random.random() < 0.75
    pd_prob   = round(random.uniform(0.70, 0.98) if is_pd else random.uniform(0.02, 0.35), 4)
    healthy_prob = round(1.0 - pd_prob, 4)
    prediction   = 1 if is_pd else 0
    conf         = pd_prob if is_pd else healthy_prob

    return jsonify({
        "prediction":           prediction,
        "label":                "PARKINSONS" if is_pd else "HEALTHY",
        "confidence":           round(conf, 4),
        "pd_probability":       pd_prob,
        "healthy_probability":  healthy_prob,
        "model_version":        MODEL_VERSION,
        # Extra field so your team knows this came from the mock
        "_mock":                True,
    })


@app.route("/predict/deterministic", methods=["POST"])
def predict_deterministic():
    """
    Deterministic version for unit tests — always returns the same response.
    Useful for writing frontend/backend tests that don't depend on randomness.

    Returns: PARKINSONS with confidence 0.9746 (matches the real smoke-test output).
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid request"}), 400

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing required features", "missing": missing}), 422

    return jsonify({
        "prediction":           1,
        "label":                "PARKINSONS",
        "confidence":           0.9746,
        "pd_probability":       0.9746,
        "healthy_probability":  0.0254,
        "model_version":        MODEL_VERSION,
        "_mock":                True,
        "_deterministic":       True,
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  🟡 MOCK MODEL SERVER — for development only")
    print("=" * 55)
    print(f"  POST http://localhost:8001/predict")
    print(f"  GET  http://localhost:8001/health")
    print(f"  POST http://localhost:8001/predict/deterministic")
    print("=" * 55)
    print("  Real predictions are NOT being made.")
    print("  Replace MODEL_SERVER_URL in .env when real")
    print("  model server is available from Mohan.")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8001, debug=True)
