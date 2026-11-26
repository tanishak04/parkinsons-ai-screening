"""
ml/pathway_a/server.py
━━━━━━━━━━━━━━━━━━━━━
The REAL model server. Loads the trained pipeline and serves predictions.

HOW TO RUN:
    python ml/pathway_a/train_baseline.py   # train model first (once)
    python ml/pathway_a/server.py           # then start this server

The server starts at: http://localhost:8001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ml.pathway_a.predict import (
    load_model,
    load_et13_model,
    predict_single,
    predict_single_et13,
)

app = Flask(__name__)
CORS(app)

# Load artifacts once at startup — not on every request
baseline_artifact = load_model()
et13_artifact = load_et13_model()
artifact = baseline_artifact  # backwards compatibility alias
print(f"✓ Baseline model loaded : {baseline_artifact['metadata']['model_version']}")
print(f"✓ ET-13 candidate loaded: {et13_artifact['metadata']['model_version']}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "server": "real",
        "model_version": baseline_artifact["metadata"]["model_version"],
        "algorithm": baseline_artifact["metadata"]["algorithm"],
        "baseline_model_version": baseline_artifact["metadata"]["model_version"],
        "baseline_algorithm": baseline_artifact["metadata"]["algorithm"],
        "et13_model_version": et13_artifact["metadata"]["model_version"],
        "et13_algorithm": et13_artifact["metadata"]["algorithm"],
        "supported_endpoints": ["/predict", "/predict/et13", "/health"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid request",
            "detail": "Request body must be JSON with Content-Type: application/json",
        }), 400

    try:
        result = predict_single(data, baseline_artifact)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": "Missing required features", "detail": str(e)}), 422
    except Exception as e:
        return jsonify({"error": "Inference failed", "detail": str(e)}), 500


@app.route("/predict/et13", methods=["POST"])
def predict_et13():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid request",
            "detail": "Request body must be JSON with Content-Type: application/json",
        }), 400

    try:
        result = predict_single_et13(data, et13_artifact)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": "Validation failed", "detail": str(e)}), 422
    except Exception as e:
        return jsonify({"error": "Inference failed", "detail": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  🟢 REAL MODEL SERVER")
    print("=" * 55)
    print(f"  POST http://localhost:8001/predict")
    print(f"  GET  http://localhost:8001/health")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8001, debug=False)
