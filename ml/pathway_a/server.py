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
from ml.pathway_a.predict import load_model, predict_single

app = Flask(__name__)
CORS(app)

# Load once at startup — not on every request
artifact = load_model()
print(f"✓ Model loaded: {artifact['metadata']['model_version']}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "server": "real",
        "model_version": artifact["metadata"]["model_version"],
        "algorithm": artifact["metadata"]["algorithm"],
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
        result = predict_single(data, artifact)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": "Missing required features", "detail": str(e)}), 422
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
