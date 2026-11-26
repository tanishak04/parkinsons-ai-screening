"""API contract tests that do not require the Flask model server."""

from contextlib import contextmanager
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.model_client import (
    InvalidModelResponse,
    ModelClient,
    ModelServerRejectedRequest,
    ModelServerUnavailable,
)


SAMPLE_FEATURES = {
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
    "PPE": 0.28465,
}

PREDICTION = {
    "prediction": 1,
    "label": "PARKINSONS",
    "confidence": 0.9746,
    "pd_probability": 0.9746,
    "healthy_probability": 0.0254,
    "model_version": "0.0.0-mock",
    "_mock": True,
}


@contextmanager
def make_client(handler):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app()
    try:
        with TestClient(app) as client:
            app.state.model_client._client = upstream
            yield client
    finally:
        asyncio.run(upstream.aclose())


def test_predict_forwards_exact_feature_aliases_and_marks_mock_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        import json

        assert set(json.loads(request.content)) == set(SAMPLE_FEATURES)
        return httpx.Response(200, json=PREDICTION)

    with make_client(handler) as client:
        response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 200
    assert response.json()["mock"] is True
    assert response.json()["prediction"] == 1
    assert "feature_count" not in response.json()
    assert "not a diagnosis" in response.json()["disclaimer"]


def test_predict_rejects_missing_and_non_numeric_features():
    with make_client(lambda request: httpx.Response(200, json=PREDICTION)) as client:
        missing = dict(SAMPLE_FEATURES)
        missing.pop("HNR")
        missing_response = client.post("/predict", json=missing)
        invalid = dict(SAMPLE_FEATURES, **{"HNR": "not-a-number"})
        invalid_response = client.post("/predict", json=invalid)

    assert missing_response.status_code == 422
    assert invalid_response.status_code == 422


def test_predict_reports_upstream_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with make_client(handler) as client:
        response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 503


def test_predict_reports_upstream_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow model", request=request)

    with make_client(handler) as client:
        response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model server request timed out"


def test_predict_maps_upstream_http_error():
    with make_client(lambda request: httpx.Response(500, json={"error": "model failure"})) as client:
        response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 502
    assert "rejected the request" in response.json()["detail"]
    assert "model failure" not in response.text


def test_predict_rejects_invalid_json_from_model():
    with make_client(lambda request: httpx.Response(200, content=b"not-json")) as client:
        response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 502
    assert response.json()["detail"] == "Model server returned invalid JSON"


def test_health_reports_upstream_connectivity():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "server": "mock"})

    with make_client(handler) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_server"] == "reachable"


def test_predict_rejects_invalid_model_response():
    with make_client(lambda request: httpx.Response(200, json={"unexpected": True})) as client:
        response = client.post("/predict", json=SAMPLE_FEATURES)

    assert response.status_code == 502


def test_predict_rejects_non_finite_values():
    payload = dict(SAMPLE_FEATURES, **{"HNR": "NaN"})
    with make_client(lambda request: httpx.Response(200, json=PREDICTION)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_model_client_predict_et13_success():
    from app.core.config import Settings

    et13_data = {"MDVP:RAP": 0.0037}
    expected_result = {"prediction": 1, "label": "PARKINSONS"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict/et13"
        return httpx.Response(200, json=expected_result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = ModelClient(Settings(), client=mock_http)
        result = await client.predict_et13(et13_data)
        assert result == expected_result


@pytest.mark.asyncio
async def test_model_client_predict_et13_rejected():
    from app.core.config import Settings

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "Invalid features"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = ModelClient(Settings(), client=mock_http)
        with pytest.raises(ModelServerRejectedRequest, match="Model server rejected"):
            await client.predict_et13({"MDVP:RAP": 0.0037})


@pytest.mark.asyncio
async def test_model_client_predict_et13_invalid_json():
    from app.core.config import Settings

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = ModelClient(Settings(), client=mock_http)
        with pytest.raises(InvalidModelResponse, match="invalid JSON"):
            await client.predict_et13({"MDVP:RAP": 0.0037})


@pytest.mark.asyncio
async def test_model_client_predict_et13_non_dict_json():
    from app.core.config import Settings

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        client = ModelClient(Settings(), client=mock_http)
        with pytest.raises(InvalidModelResponse, match="must be a JSON object"):
            await client.predict_et13({"MDVP:RAP": 0.0037})


