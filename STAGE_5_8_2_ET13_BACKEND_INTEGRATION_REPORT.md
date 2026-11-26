# Stage 5.8.2 — ET-13 Backend Audio Inference Integration Report

**Project:** Parkinson's AI Voice Screening System — Pathway A (UCI Dataset)  
**Stage:** 5.8.2 — ET-13 Backend Audio Inference Integration  
**Date:** 2026-11-25  
**Author:** DeepMind Antigravity Agent  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

Stage 5.8.2 has successfully connected the locked ET-13 pitch-free research candidate (`ExtraTreesClassifier`, `max_depth=3`, 13 features) into the FastAPI backend audio inference pipeline (`POST /predict/audio`). 

The audio inference pipeline now takes uploaded voice recordings (`.wav`, `.ogg`, `.webm`, `.mp3`), extracts acoustic vocal perturbation features via Praat/Parselmouth through `AudioFeatureExtractor`, isolates the exact 13 pitch-free biomarkers, forwards them to the model server's `/predict/et13` endpoint, and returns a fully normalized research screening response featuring dual operating points (reference threshold 0.50 and exploratory high-sensitivity threshold 0.45) and explicit non-diagnostic disclaimers.

Crucially, **strict model isolation has been maintained**:
- The baseline 22-feature route (`POST /predict`) remains 100% operational and untouched.
- The model artifacts (`champion_pipeline.joblib` and `et13_pitch_free_candidate.joblib`) were untouched.
- Pitch biomarkers (`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`) and nonlinear features (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`) are strictly excluded from the ET-13 inference vector.
- All 28 backend unit/integration tests and 10 Flask server unit tests passed with 100% success.

---

## 2. Architecture & Integration Topology

```
                  ┌────────────────────────────────────────────────────────┐
                  │                   Client Request                       │
                  │             POST /predict/audio (multipart)            │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Audio Bytes (WAV/OGG/etc.)
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 FastAPI Audio Route                    │
                  │            (backend/app/routes/predict.py)             │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Validates MIME & size
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              AudioFeatureExtractor                     │
                  │           (backend/app/services/audio_extractor.py)    │
                  │   Praat / Parselmouth acoustic perturbation analysis   │
                  └───────────────────────────┬────────────────────────────┘
                                              │ 16 Extracted Features
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            Pitch-Free Feature Selection                │
                  │   Filters out 3 pitch features: Fo, Fhi, Flo           │
                  │   Strictly retains exactly 13 perturbation features    │
                  └───────────────────────────┬────────────────────────────┘
                                              │ 13 Numerical Features Payload
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                ModelClient.predict_et13()              │
                  │           (backend/app/services/model_client.py)       │
                  └───────────────────────────┬────────────────────────────┘
                                              │ HTTP POST /predict/et13
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 Flask ML Server                        │
                  │              (ml/pathway_a/server.py:8001)             │
                  │   Artifact: et13_pitch_free_candidate.joblib          │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Inference Result JSON
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │             AudioPredictionResponse Schema             │
                  │            (backend/app/schemas/prediction.py)         │
                  │   Dual thresholds, probabilities, audit metadata,      │
                  │   and non-diagnostic research disclaimer               │
                  └────────────────────────────────────────────────────────┘
```

---

## 3. Verified Feature Selection & Alignment

The 13 pitch-free features isolated from `AudioFeatureExtractor` and delivered to `POST /predict/et13`:

| Index | Feature Name | Extraction Method | Units | Model Inclusion |
| :---: | :--- | :--- | :--- | :---: |
| 1 | `MDVP:Jitter(%)` | Praat PointProcess Jitter (local) | % | Included |
| 2 | `MDVP:Jitter(Abs)` | Praat PointProcess Jitter (local, absolute) | Seconds | Included |
| 3 | `MDVP:RAP` | Praat PointProcess Jitter (rap) | Ratio | Included |
| 4 | `MDVP:PPQ` | Praat PointProcess Jitter (ppq5) | Ratio | Included |
| 5 | `Jitter:DDP` | Praat PointProcess Jitter (ddp) | Ratio | Included |
| 6 | `MDVP:Shimmer` | Praat PointProcess Shimmer (local) | Ratio | Included |
| 7 | `MDVP:Shimmer(dB)` | Praat PointProcess Shimmer (local_dB) | dB | Included |
| 8 | `Shimmer:APQ3` | Praat PointProcess Shimmer (apq3) | Ratio | Included |
| 9 | `Shimmer:APQ5` | Praat PointProcess Shimmer (apq5) | Ratio | Included |
| 10 | `MDVP:APQ` | Praat PointProcess Shimmer (apq11) | Ratio | Included |
| 11 | `Shimmer:DDA` | Praat PointProcess Shimmer (dda) | Ratio | Included |
| 12 | `NHR` | Harmonics-to-Noise Ratio inversion ($10^{-\text{HNR}/10}$) | Ratio | Included |
| 13 | `HNR` | Praat Harmonicity cross-correlation (0.01s step) | dB | Included |

### Excluded Features
- **Pitch Features (Excluded by Design):** `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`. Filtered out to eliminate fundamental frequency demographic confounding and pitch bias.
- **Unavailable Non-Linear Features (Excluded):** `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`. The ET-13 model does not require or accept these Little et al. algorithms.

---

## 4. Implementation Details by Component

### 4.1. Model Client (`backend/app/services/model_client.py`)
Added dedicated `predict_et13()` method:
- Retains existing connection pooling, timeout handling (`model_server_timeout_seconds`), and retry logic.
- Targets upstream endpoint: `{model_base_url}/predict/et13`.
- Validates HTTP response status: raises `ModelServerRejectedRequest(status_code=502)` on upstream 4xx/5xx.
- Validates JSON format: raises `InvalidModelResponse` on non-JSON or non-dict payloads.
- Retains existing baseline `predict()` and `health()` intact.

### 4.2. Prediction Schemas (`backend/app/schemas/prediction.py`)
- Implemented `ExploratoryClassification`:
  - `threshold: float = 0.45`
  - `prediction: Literal[0, 1]`
  - `label: Literal["PARKINSONS", "HEALTHY"]`
  - `note: str`
- Implemented `AudioPredictionResponse`:
  - `prediction: Literal[0, 1]`
  - `label: Literal["PARKINSONS", "HEALTHY"]`
  - `confidence: float` (bounded `[0, 1]`)
  - `pd_probability: float` (bounded `[0, 1]`)
  - `healthy_probability: float` (bounded `[0, 1]`)
  - `model_version: str` (`"1.0.0-research-candidate-et13"`)
  - `model_designation: str` (`"ET-13 Pitch-Free Research Candidate"`)
  - `feature_count: int` (`13`)
  - `reference_threshold: float` (`0.50`)
  - `exploratory_classification: ExploratoryClassification`
  - `duration_seconds: float | None`
  - `sampling_frequency: float | None`
  - `extracted_features: dict[str, float] | None`
  - `disclaimer: str`
- Preserved existing `PredictionRequest`, `PredictionResponse`, `AudioCapabilities`, `HealthResponse`, `AudioExtractionResponse`, and `AudioScreeningDisabledResponse` for full backward compatibility.

### 4.3. FastAPI Audio Route (`backend/app/routes/predict.py`)
Updated `POST /predict/audio`:
- Updated `response_model=AudioPredictionResponse`.
- Validates file MIME types against `ACCEPTED_AUDIO_MIME_TYPES`.
- Validates non-empty payload bytes (rejects with 400 if empty).
- Calls `AudioFeatureExtractor.extract_features()` (rejects with 400 on audio errors such as silence or short duration).
- Validates that all 13 ET-13 features are present (raises 422 if missing).
- Validates that all 13 features are finite numerical values (rejects NaN/Inf with 422).
- Filters out pitch features (`Fo`, `Fhi`, `Flo`).
- Calls `request.app.state.model_client.predict_et13(et13_payload)`.
- Handles upstream exceptions (`ModelServerUnavailable` $\to$ 503, `ModelServerRejectedRequest` $\to$ 502, `InvalidModelResponse` $\to$ 502).
- Returns structured `AudioPredictionResponse`.

---

## 5. Verification & Test Suite Execution

### 5.1. Unit & Integration Tests (`backend/tests/`)
All 28 tests passed in 3.63 seconds:
- `backend/tests/test_audio.py`:
  - `test_extractor_extracts_16_features_from_synthetic_voice`: PASSED
  - `test_extractor_rejects_empty_bytes`: PASSED
  - `test_extractor_rejects_short_duration`: PASSED
  - `test_extractor_rejects_pure_silence`: PASSED
  - `test_health_reports_audio_capabilities`: PASSED
  - `test_predict_audio_executes_et13_inference_pipeline`: PASSED (verified exact 13 features, no pitch features, valid response schema, dual thresholds)
  - `test_predict_audio_rejects_empty_file`: PASSED
  - `test_predict_audio_rejects_unsupported_media_type`: PASSED
  - `test_predict_audio_rejects_silence`: PASSED
  - `test_predict_audio_rejects_too_short`: PASSED
  - `test_predict_audio_reports_upstream_unavailable`: PASSED (503)
  - `test_predict_audio_reports_upstream_timeout`: PASSED (503)
  - `test_predict_audio_reports_upstream_rejection`: PASSED (502)
  - `test_predict_audio_rejects_invalid_json_from_model`: PASSED (502)
  - `test_predict_audio_rejects_invalid_model_response_schema`: PASSED (502)
- `backend/tests/test_api.py`:
  - `test_predict_forwards_exact_feature_aliases_and_marks_mock_response`: PASSED
  - `test_predict_rejects_missing_and_non_numeric_features`: PASSED
  - `test_predict_reports_upstream_unavailable`: PASSED
  - `test_predict_reports_upstream_timeout`: PASSED
  - `test_predict_maps_upstream_http_error`: PASSED
  - `test_predict_rejects_invalid_json_from_model`: PASSED
  - `test_health_reports_upstream_connectivity`: PASSED
  - `test_predict_rejects_invalid_model_response`: PASSED
  - `test_predict_rejects_non_finite_values`: PASSED
  - `test_model_client_predict_et13_success`: PASSED
  - `test_model_client_predict_et13_rejected`: PASSED
  - `test_model_client_predict_et13_invalid_json`: PASSED
  - `test_model_client_predict_et13_non_dict_json`: PASSED

### 5.2. Baseline ML Server Tests (`ml/pathway_a/tests/test_server.py`)
All 10 tests passed in 1.22 seconds:
- `test_health_endpoint`: PASSED
- `test_baseline_predict_success`: PASSED
- `test_baseline_predict_missing_feature`: PASSED
- `test_baseline_predict_malformed_json`: PASSED
- `test_et13_predict_success`: PASSED
- `test_et13_predict_missing_feature`: PASSED
- `test_et13_predict_invalid_numerical_string`: PASSED
- `test_et13_predict_nan_value`: PASSED
- `test_et13_predict_malformed_json`: PASSED
- `test_route_isolation`: PASSED

### 5.3. Live Daemon End-to-End Verification
Executed live integration against running Flask model server (`task-2105`, port 8001):
1. **Baseline route verification (`POST /predict`):**
   - HTTP Status: 200 OK
   - Model version: `0.1.0-baseline` (22 features verified)
2. **Audio pipeline end-to-end (`POST /predict/audio`):**
   - Input: Synthetic sustained phonation WAV (1.5s, 22050 Hz, 160 Hz F0)
   - Extracted features: 16 features extracted by Praat
   - Isolated payload: Exactly 13 pitch-free features
   - Forwarded to: `http://localhost:8001/predict/et13`
   - HTTP Status: 200 OK
   - Response Payload:
     ```json
     {
       "prediction": 0,
       "label": "HEALTHY",
       "confidence": 0.7296,
       "pd_probability": 0.2704,
       "healthy_probability": 0.7296,
       "model_version": "1.0.0-research-candidate-et13",
       "model_designation": "ET-13 Pitch-Free Research Candidate",
       "feature_count": 13,
       "reference_threshold": 0.5,
       "exploratory_classification": {
         "threshold": 0.45,
         "prediction": 0,
         "label": "HEALTHY",
         "note": "Exploratory high-sensitivity screening operating point; non-diagnostic."
       },
       "duration_seconds": 1.5,
       "sampling_frequency": 22050.0,
       "extracted_features": {
         "MDVP:Jitter(%)": 0.000228,
         "MDVP:Jitter(Abs)": 1e-06,
         "MDVP:RAP": 0.000136,
         "MDVP:PPQ": 0.000137,
         "Jitter:DDP": 0.000408,
         "MDVP:Shimmer": 0.001913,
         "MDVP:Shimmer(dB)": 0.0166,
         "Shimmer:APQ3": 0.001133,
         "Shimmer:APQ5": 0.001198,
         "MDVP:APQ": 0.001269,
         "Shimmer:DDA": 0.003398,
         "NHR": 0.000142,
         "HNR": 38.469
       },
       "disclaimer": "Research prototype only. Investigational machine-learning screening based on 13 pitch-free vocal perturbation biomarkers. Not a clinical diagnosis."
     }
     ```

---

## 6. Strict Boundary Adherence

1. **Model Immutability:** Neither `champion_pipeline.joblib` nor `et13_pitch_free_candidate.joblib` was retrained, converted, or modified.
2. **Audio Extractor Immutability:** `backend/app/services/audio_extractor.py` was left untouched.
3. **Route Isolation:** The baseline `/predict` endpoint continues to serve 22-feature tabular vectors without regression.
4. **Research Framing:** All ET-13 responses carry explicit research prototype and non-diagnostic disclaimers.

---

## 7. Conclusion

Stage 5.8.2 is **COMPLETE**. The FastAPI backend now fully supports end-to-end voice screening inference using the research ET-13 candidate.
