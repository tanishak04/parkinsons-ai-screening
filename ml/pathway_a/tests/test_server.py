"""
ml/pathway_a/tests/test_server.py
Unit and integration tests for the Flask model server:
  - Baseline 22-feature route (/predict)
  - Research candidate 13-feature route (/predict/et13)
  - Health endpoint (/health)
"""

import json
import pytest
from ml.pathway_a.server import app

SAMPLE_22_FEATURES = {
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

SAMPLE_13_FEATURES = {
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
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["server"] == "real"
    assert "baseline_model_version" in data
    assert "et13_model_version" in data
    assert "/predict/et13" in data["supported_endpoints"]


# ---------------------------------------------------------------------------
# Baseline Regression Tests (/predict)
# ---------------------------------------------------------------------------
def test_baseline_predict_success(client):
    res = client.post("/predict", json=SAMPLE_22_FEATURES)
    assert res.status_code == 200
    data = res.get_json()
    assert data["feature_count"] == 22
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("PARKINSONS", "HEALTHY")
    assert 0.0 <= data["pd_probability"] <= 1.0
    assert 0.0 <= data["healthy_probability"] <= 1.0
    assert round(data["pd_probability"] + data["healthy_probability"], 3) == 1.0


def test_baseline_predict_missing_feature(client):
    incomplete = dict(SAMPLE_22_FEATURES)
    del incomplete["PPE"]
    res = client.post("/predict", json=incomplete)
    assert res.status_code == 422
    data = res.get_json()
    assert "Missing required features" in data["error"]


def test_baseline_predict_malformed_json(client):
    res = client.post("/predict", data="not json", content_type="text/plain")
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# ET-13 Candidate Route Tests (/predict/et13)
# ---------------------------------------------------------------------------
def test_et13_predict_success(client):
    res = client.post("/predict/et13", json=SAMPLE_13_FEATURES)
    assert res.status_code == 200
    data = res.get_json()
    assert data["model_version"] == "1.0.0-research-candidate-et13"
    assert data["feature_count"] == 13
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("PARKINSONS", "HEALTHY")
    assert 0.0 <= data["pd_probability"] <= 1.0
    assert 0.0 <= data["healthy_probability"] <= 1.0
    assert round(data["pd_probability"] + data["healthy_probability"], 3) == 1.0
    assert data["reference_threshold"] == 0.50

    # Exploratory threshold verification
    exp = data["exploratory_classification"]
    assert exp["threshold"] == 0.45
    assert exp["prediction"] in (0, 1)
    assert exp["label"] in ("PARKINSONS", "HEALTHY")
    assert "Exploratory" in exp["note"]

    # Research disclaimer verification
    assert "Research prototype only" in data["disclaimer"]
    assert "Not a clinical diagnosis" in data["disclaimer"]


def test_et13_predict_missing_feature(client):
    incomplete = dict(SAMPLE_13_FEATURES)
    del incomplete["HNR"]
    res = client.post("/predict/et13", json=incomplete)
    assert res.status_code == 422
    data = res.get_json()
    assert "Validation failed" in data["error"]
    assert "HNR" in data["detail"]


def test_et13_predict_invalid_numerical_string(client):
    invalid = dict(SAMPLE_13_FEATURES)
    invalid["HNR"] = "not_a_float"
    res = client.post("/predict/et13", json=invalid)
    assert res.status_code == 422
    data = res.get_json()
    assert "Validation failed" in data["error"]
    assert "must be a finite numerical value" in data["detail"]


def test_et13_predict_nan_value(client):
    invalid = dict(SAMPLE_13_FEATURES)
    invalid["HNR"] = float("nan")
    res = client.post("/predict/et13", json=invalid)
    # JSON standard converts float('nan') to null or fails
    # Let's test with None or null
    invalid_null = dict(SAMPLE_13_FEATURES)
    invalid_null["HNR"] = None
    res_null = client.post("/predict/et13", json=invalid_null)
    assert res_null.status_code == 422
    assert "must be a finite numerical value" in res_null.get_json()["detail"]


def test_et13_predict_malformed_json(client):
    res = client.post("/predict/et13", data="plain text body", content_type="text/plain")
    assert res.status_code == 400
    assert "Invalid request" in res.get_json()["error"]


def test_route_isolation(client):
    # Sending 13 features to baseline /predict must fail (missing 9 features)
    res_baseline = client.post("/predict", json=SAMPLE_13_FEATURES)
    assert res_baseline.status_code == 422
    assert "Missing required features" in res_baseline.get_json()["error"]

    # Sending 22 features to /predict/et13 should succeed by selecting the 13 pitch-free features
    res_et13 = client.post("/predict/et13", json=SAMPLE_22_FEATURES)
    assert res_et13.status_code == 200
    assert res_et13.get_json()["feature_count"] == 13
