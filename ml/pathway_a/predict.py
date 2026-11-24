"""
predict.py — Inference contract for the Stage-1 baseline pipeline.

Usage (single record from CLI):
    python ml/pathway_a/predict.py

Usage (programmatic):
    from ml.pathway_a.predict import load_model, predict_single, predict_batch
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACT_PATH = BASE_DIR / "models" / "baseline_pipeline.joblib"
METRICS_PATH = BASE_DIR / "models" / "baseline_metrics.json"

# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------
# A single prediction result returned by predict_single / predict_batch.
PredictionResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model() -> dict[str, Any]:
    """Load and return the saved model artifact dict."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_ARTIFACT_PATH}.\n"
            "Run `python ml/pathway_a/train_baseline.py` first."
        )
    return joblib.load(MODEL_ARTIFACT_PATH)


def predict_single(features: dict[str, float], artifact: dict | None = None) -> PredictionResult:
    """
    Run inference on a single record.

    Parameters
    ----------
    features : dict mapping feature name → float value
    artifact : pre-loaded artifact dict (optional; loaded from disk if None)

    Returns
    -------
    PredictionResult with keys:
        prediction        int   (1 = Parkinson's, 0 = Healthy)
        label             str   ("PARKINSONS" | "HEALTHY")
        confidence        float probability of the predicted class
        pd_probability    float probability of Parkinson's (class=1)
        healthy_probability float probability of Healthy (class=0)
        model_version     str
        feature_count     int
    """
    if artifact is None:
        artifact = load_model()

    pipeline = artifact["pipeline"]
    expected_features: list[str] = artifact["feature_names"]

    # Validate & order features
    missing = set(expected_features) - set(features.keys())
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    X = pd.DataFrame([features])[expected_features]

    pred_class: int = int(pipeline.predict(X)[0])
    proba: np.ndarray = pipeline.predict_proba(X)[0]  # [P(healthy), P(PD)]
    pd_prob = float(proba[1])
    healthy_prob = float(proba[0])
    conf = pd_prob if pred_class == 1 else healthy_prob

    return {
        "prediction": pred_class,
        "label": "PARKINSONS" if pred_class == 1 else "HEALTHY",
        "confidence": round(conf, 4),
        "pd_probability": round(pd_prob, 4),
        "healthy_probability": round(healthy_prob, 4),
        "model_version": artifact["metadata"]["model_version"],
        "feature_count": len(expected_features),
    }


def predict_batch(df: pd.DataFrame, artifact: dict | None = None) -> list[PredictionResult]:
    """
    Run inference on a DataFrame of records.

    Parameters
    ----------
    df      : DataFrame with one row per recording; must contain all expected feature columns.
    artifact: pre-loaded artifact dict (optional)

    Returns
    -------
    List of PredictionResult dicts, one per row.
    """
    if artifact is None:
        artifact = load_model()

    pipeline = artifact["pipeline"]
    expected_features: list[str] = artifact["feature_names"]

    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[expected_features]
    pred_classes = pipeline.predict(X).astype(int)
    probas = pipeline.predict_proba(X)  # shape (n, 2)

    results: list[PredictionResult] = []
    for i in range(len(df)):
        pred_class = int(pred_classes[i])
        pd_prob = float(probas[i, 1])
        healthy_prob = float(probas[i, 0])
        conf = pd_prob if pred_class == 1 else healthy_prob
        results.append({
            "prediction": pred_class,
            "label": "PARKINSONS" if pred_class == 1 else "HEALTHY",
            "confidence": round(conf, 4),
            "pd_probability": round(pd_prob, 4),
            "healthy_probability": round(healthy_prob, 4),
            "model_version": artifact["metadata"]["model_version"],
            "feature_count": len(expected_features),
        })
    return results


# ---------------------------------------------------------------------------
# CLI smoke-test: runs a synthetic sample and a real sample from the dataset
# ---------------------------------------------------------------------------
def _cli_smoke_test() -> None:
    print("=" * 60)
    print("  Pathway-A Baseline — Inference Contract Smoke Test")
    print("=" * 60)

    artifact = load_model()
    expected_features: list[str] = artifact["feature_names"]
    print(f"\n✓ Model loaded  ({artifact['metadata']['model_version']})")
    print(f"✓ Expected {len(expected_features)} features\n")

    # --- Test 1: synthetic PD-like record (high jitter/shimmer) ----------
    synthetic_pd = {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.00370,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.426,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.03130,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.033,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654,
    }

    result1 = predict_single(synthetic_pd, artifact)
    print("[Test 1] Synthetic PD-like record:")
    print(json.dumps(result1, indent=2))

    # --- Test 2: first real row from the dataset -------------------------
    raw_data_path = BASE_DIR / "data" / "raw" / "parkinsons.data"
    if raw_data_path.exists():
        df_raw = pd.read_csv(raw_data_path)
        # Drop non-feature columns
        X_real = df_raw.drop(columns=["name", "status"])
        true_labels = df_raw["status"].tolist()

        results_batch = predict_batch(X_real.iloc[:5], artifact)
        print(f"\n[Test 2] First 5 real records (true labels: {true_labels[:5]}):")
        for i, r in enumerate(results_batch):
            match = "✓" if r["prediction"] == true_labels[i] else "✗"
            print(f"  Record {i}: pred={r['label']:11s} conf={r['confidence']:.4f}  true={true_labels[i]}  {match}")
    else:
        print("\n[Test 2] Raw dataset not found — skipping real-data test.")

    print("\n✓ Inference contract verified — predict.py is ready.\n")


if __name__ == "__main__":
    _cli_smoke_test()
