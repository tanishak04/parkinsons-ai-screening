"""Tests for audio feature extraction service and /predict/audio endpoint with ET-13 integration."""

import asyncio
from contextlib import contextmanager
import io
import json
import numpy as np
import pytest
import soundfile as sf
import httpx
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.audio_extractor import (
    AudioExtractionError,
    AudioFeatureExtractor,
    EXTRACTABLE_PRAAT_FEATURES,
    NONLINEAR_UNAVAILABLE_FEATURES,
    REQUIRED_FEATURES,
)

PITCH_FREE_13_FEATURES = [
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR"
]

ET13_MOCK_RESPONSE = {
    "model_version": "1.0.0-research-candidate-et13",
    "model_designation": "ET-13 Pitch-Free Research Candidate",
    "feature_count": 13,
    "pd_probability": 0.8587,
    "healthy_probability": 0.1413,
    "confidence": 0.8587,
    "prediction": 1,
    "label": "PARKINSONS",
    "reference_threshold": 0.50,
    "exploratory_classification": {
        "threshold": 0.45,
        "prediction": 1,
        "label": "PARKINSONS",
        "note": "Exploratory high-sensitivity screening operating point; non-diagnostic.",
    },
    "disclaimer": (
        "Research prototype only. Investigational machine-learning screening based on "
        "13 pitch-free vocal perturbation biomarkers. Not a clinical diagnosis."
    ),
}


def generate_synthetic_voice(
    duration: float = 1.5,
    sample_rate: int = 22050,
    f0: float = 160.0,
) -> bytes:
    """Generate a synthetic periodic vowel-like signal as WAV bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Fundamental frequency plus harmonics with slight perturbation
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.05 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    # Add small Gaussian noise
    noise = np.random.normal(0, 0.005, signal.shape)
    signal = (signal + noise) / np.max(np.abs(signal + noise))

    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@contextmanager
def make_mocked_client(handler):
    """Create a TestClient with a mocked upstream ModelClient transport."""
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app()
    try:
        with TestClient(app) as client:
            app.state.model_client._client = upstream
            yield client
    finally:
        asyncio.run(upstream.aclose())


# ---------------------------------------------------------------------------
# Unit tests for AudioFeatureExtractor
# ---------------------------------------------------------------------------


def test_extractor_extracts_16_features_from_synthetic_voice():
    wav_bytes = generate_synthetic_voice(duration=1.5, f0=150.0)
    result = AudioFeatureExtractor.extract_features(wav_bytes, filename="test.wav")

    assert result["status"] == "partial_extraction"
    assert result["is_fully_compatible"] is False
    assert result["extracted_feature_count"] == 16
    assert result["required_feature_count"] == 22
    assert result["duration_seconds"] >= 1.4

    extracted = result["extracted_features"]
    assert len(extracted) == 16
    for feat in EXTRACTABLE_PRAAT_FEATURES:
        assert feat in extracted, f"Missing feature: {feat}"
        assert np.isfinite(extracted[feat]), f"Feature {feat} is non-finite: {extracted[feat]}"

    # Fundamental frequency should be approximately 150 Hz
    assert 130 <= extracted["MDVP:Fo(Hz)"] <= 170

    # Missing features must be the 6 nonlinear dynamical biomarkers
    missing = result["missing_features"]
    assert missing == NONLINEAR_UNAVAILABLE_FEATURES
    assert "spread1" in missing
    assert "spread2" in missing
    assert "PPE" in missing
    assert "RPDE" in missing
    assert "DFA" in missing
    assert "D2" in missing

    # Total features must sum to 22
    assert len(extracted) + len(missing) == len(REQUIRED_FEATURES)
    assert result["blocker_code"] == "NONLINEAR_FEATURES_UNAVAILABLE"


def test_extractor_rejects_empty_bytes():
    with pytest.raises(AudioExtractionError, match="Empty audio payload"):
        AudioFeatureExtractor.extract_features(b"", filename="empty.wav")


def test_extractor_rejects_short_duration():
    short_wav = generate_synthetic_voice(duration=0.3)
    with pytest.raises(AudioExtractionError, match="too short"):
        AudioFeatureExtractor.extract_features(short_wav, filename="short.wav", min_duration_sec=0.8)


def test_extractor_rejects_pure_silence():
    sample_rate = 22050
    silence = np.zeros(sample_rate * 2)
    buffer = io.BytesIO()
    sf.write(buffer, silence, sample_rate, format="WAV")
    with pytest.raises(AudioExtractionError, match="No periodic pitch detected"):
        AudioFeatureExtractor.extract_features(buffer.getvalue(), filename="silence.wav")


# ---------------------------------------------------------------------------
# API tests for /health and /predict/audio with ET-13 inference
# ---------------------------------------------------------------------------


def test_health_reports_audio_capabilities(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert "capabilities" in data
    caps = data["capabilities"]
    assert caps["audio_feature_extraction"] is True
    assert caps["audio_input_compatible"] is False
    assert caps["supported_features_count"] == 16
    assert caps["required_features_count"] == 22
    assert "audio/wav" in caps["accepted_audio_formats"]
    assert caps["incompatibility_reason"] is not None
    assert "validation criteria" in caps["incompatibility_reason"].lower()


def test_predict_audio_executes_et13_inference_pipeline():
    """Audio endpoint extracts features, filters to 13 pitch-free features, and returns ET-13 inference."""
    received_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict/et13"
        payload = json.loads(request.content)
        received_requests.append(payload)

        # Verify exactly 13 features forwarded
        assert len(payload) == 13
        assert set(payload.keys()) == set(PITCH_FREE_13_FEATURES)

        # Strict isolation: verify pitch features are completely absent
        assert "MDVP:Fo(Hz)" not in payload
        assert "MDVP:Fhi(Hz)" not in payload
        assert "MDVP:Flo(Hz)" not in payload

        # All values must be finite floats
        for k, v in payload.items():
            assert isinstance(v, (int, float))
            assert np.isfinite(v)

        return httpx.Response(200, json=ET13_MOCK_RESPONSE)

    wav_bytes = generate_synthetic_voice(duration=1.2, f0=180.0)
    files = {"file": ("phonation.wav", wav_bytes, "audio/wav")}

    with make_mocked_client(handler) as client:
        response = client.post("/predict/audio", files=files)

    assert response.status_code == 200
    assert len(received_requests) == 1

    body = response.json()
    assert body["prediction"] == 1
    assert body["label"] == "PARKINSONS"
    assert body["pd_probability"] == 0.8587
    assert body["healthy_probability"] == 0.1413
    assert body["confidence"] == 0.8587
    assert body["model_version"] == "1.0.0-research-candidate-et13"
    assert body["model_designation"] == "ET-13 Pitch-Free Research Candidate"
    assert body["feature_count"] == 13
    assert body["reference_threshold"] == 0.50
    assert body["duration_seconds"] >= 1.1
    assert body["sampling_frequency"] == 22050
    assert len(body["extracted_features"]) == 13
    assert "MDVP:Fo(Hz)" not in body["extracted_features"]

    # Exploratory classification
    assert "exploratory_classification" in body
    exploratory = body["exploratory_classification"]
    assert exploratory["threshold"] == 0.45
    assert exploratory["prediction"] == 1
    assert exploratory["label"] == "PARKINSONS"
    assert "non-diagnostic" in exploratory["note"].lower()

    # Disclaimer
    assert "Research prototype only" in body["disclaimer"]
    assert "13 pitch-free" in body["disclaimer"]


def test_predict_audio_rejects_empty_file(client):
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/predict/audio", files=files)
    assert response.status_code == 400
    assert "Empty audio payload" in response.json()["detail"]


def test_predict_audio_rejects_unsupported_media_type(client):
    files = {"file": ("doc.pdf", b"some dummy bytes", "application/pdf")}
    response = client.post("/predict/audio", files=files)
    assert response.status_code == 400
    assert "Unsupported audio media type" in response.json()["detail"]


def test_predict_audio_rejects_silence(client):
    sample_rate = 22050
    silence = np.zeros(sample_rate * 2)
    buffer = io.BytesIO()
    sf.write(buffer, silence, sample_rate, format="WAV")
    files = {"file": ("silence.wav", buffer.getvalue(), "audio/wav")}
    response = client.post("/predict/audio", files=files)
    assert response.status_code == 400
    assert "No periodic pitch detected" in response.json()["detail"]


def test_predict_audio_rejects_too_short(client):
    short_wav = generate_synthetic_voice(duration=0.3)
    files = {"file": ("short.wav", short_wav, "audio/wav")}
    response = client.post("/predict/audio", files=files)
    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()


def test_predict_audio_reports_upstream_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Model server offline", request=request)

    wav_bytes = generate_synthetic_voice(duration=1.2, f0=180.0)
    files = {"file": ("phonation.wav", wav_bytes, "audio/wav")}

    with make_mocked_client(handler) as client:
        response = client.post("/predict/audio", files=files)

    assert response.status_code == 503
    assert "Model server is unavailable" in response.json()["detail"]


def test_predict_audio_reports_upstream_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out", request=request)

    wav_bytes = generate_synthetic_voice(duration=1.2, f0=180.0)
    files = {"file": ("phonation.wav", wav_bytes, "audio/wav")}

    with make_mocked_client(handler) as client:
        response = client.post("/predict/audio", files=files)

    assert response.status_code == 503
    assert "Model server request timed out" in response.json()["detail"]


def test_predict_audio_reports_upstream_rejection():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Inference computation error"})

    wav_bytes = generate_synthetic_voice(duration=1.2, f0=180.0)
    files = {"file": ("phonation.wav", wav_bytes, "audio/wav")}

    with make_mocked_client(handler) as client:
        response = client.post("/predict/audio", files=files)

    assert response.status_code == 502
    assert "Model server rejected the request" in response.json()["detail"]


def test_predict_audio_rejects_invalid_json_from_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"corrupt-non-json")

    wav_bytes = generate_synthetic_voice(duration=1.2, f0=180.0)
    files = {"file": ("phonation.wav", wav_bytes, "audio/wav")}

    with make_mocked_client(handler) as client:
        response = client.post("/predict/audio", files=files)

    assert response.status_code == 502
    assert "Model server returned invalid JSON" in response.json()["detail"]


def test_predict_audio_rejects_invalid_model_response_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected_field": True})

    wav_bytes = generate_synthetic_voice(duration=1.2, f0=180.0)
    files = {"file": ("phonation.wav", wav_bytes, "audio/wav")}

    with make_mocked_client(handler) as client:
        response = client.post("/predict/audio", files=files)

    assert response.status_code == 502
    assert "invalid ET-13 prediction response" in response.json()["detail"]

