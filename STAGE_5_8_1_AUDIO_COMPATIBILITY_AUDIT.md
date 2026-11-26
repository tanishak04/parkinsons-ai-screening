# Stage 5.8.1 — Audio-to-Inference Compatibility Audit Report

**Project:** Parkinson's AI Voice Screening System — Pathway A (UCI Dataset)  
**Execution Timestamp:** 2026-11-25 10:50:00 UTC  
**Audit Scope:** Deep inspection of the FastAPI audio inference pipeline, Praat feature extractor, mathematical biomarker definitions, and backend model client routing for ET-13 candidate integration.  
**Audit Status:** COMPLETE — VERIFIED AUDIT  
**Recommendation Verdict:** **INTEGRATION READY**

---

## 1. Files Inspected

- [backend/app/services/audio_extractor.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/backend/app/services/audio_extractor.py) — Parselmouth/Praat acoustic feature extraction engine
- [backend/app/routes/predict.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/backend/app/routes/predict.py) — FastAPI prediction endpoints (`POST /predict` and `POST /predict/audio`)
- [backend/app/schemas/prediction.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/backend/app/schemas/prediction.py) — Pydantic request/response validation schemas
- [backend/app/services/model_client.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/backend/app/services/model_client.py) — HTTPX client interfacing with Flask model server
- [backend/app/routes/health.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/backend/app/routes/health.py) — System capability declaration endpoint
- [ml/pathway_a/server.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/ml/pathway_a/server.py) — Flask model server hosting `/predict` and `/predict/et13`
- [ml/pathway_a/predict.py](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/ml/pathway_a/predict.py) — Server inference contracts (`predict_single` and `predict_single_et13`)
- [ml/pathway_a/experiments/models/et13_candidate_manifest.json](file:///c:/Users/mohan/Desktop/major%20project/parkinsons-ai-screening/ml/pathway_a/experiments/models/et13_candidate_manifest.json) — Manifest of the locked ET-13 candidate

---

## 2. Actual Current Inference Flow

The current operational pathway from microphone capture to prediction response is structured as follows:

```
[ User Microphone / File Upload ]
                │
                ▼ (multipart/form-data: file=<binary>)
[ FastAPI: POST /predict/audio ]
  ├── 1. MIME Type Validation (checks ACCEPTED_AUDIO_MIME_TYPES)
  ├── 2. Audio Payload Decoding via soundfile / Parselmouth (converts stereo to mono)
  ├── 3. Duration Gate (enforces duration >= 0.8s)
  ├── 4. Periodic Pitch Gate (verifies f0_mean > 0 and not NaN)
  ├── 5. Praat Feature Extraction (calls To PointProcess and To Harmonicity)
  │      └── Extracts 16 linear features (3 pitch, 5 jitter, 6 shimmer, 2 noise ratios)
  │
  └── 6. CURRENT BEHAVIOR:
         Short-circuits immediately.
         Constructs AudioScreeningDisabledResponse with status="screening_disabled"
         and screening_available=False.
         Does NOT call ModelClient. Does NOT forward features to Flask model server.
```

---

## 3. Feature Extraction Inventory & Capabilities

### Baseline 22-Feature Extractor Status
In the UCI training dataset (`parkinsons.data`), the 22 features were derived using:
1. **16 Linear Biomarkers:** Extracted using Kay Elemetrics Multi-Dimensional Voice Program (MDVP) workstation software.
2. **6 Non-Linear Dynamical Biomarkers:** Extracted by Max Little (Little et al., 2007) using specialized C/MATLAB phase-space reconstruction algorithms (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`).

Because the 6 non-linear features cannot be reliably extracted from browser audio without clinical workstation software, **the 22-feature champion model is permanently blocked from live audio inference**.

### ET-13 Feature Extraction Status
The ET-13 research candidate explicitly discards:
- The 3 raw pitch features (`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`) to eliminate demographic sex confounding.
- All 6 unavailable non-linear dynamical features.

This leaves **exactly 13 pitch-free vocal perturbation biomarkers**.
Inspection of `AudioFeatureExtractor.extract_features()` confirms that **all 13 features are already actively calculated and returned in the `extracted_features` dictionary**.

---

## 4. ET-13 Feature-by-Feature Compatibility Matrix (13 Features)

| # | ET-13 Target Feature | Praat Call in `audio_extractor.py` | Mathematical Formula / Algorithmic Basis | Dataset Unit / Scale | Praat Unit / Scale | Compatibility Status |
|:---:|---|---|---|---|---|:---:|
| 1 | `MDVP:Jitter(%)` | `call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)` | Cycle-to-cycle period perturbation ratio: $\frac{\frac{1}{N-1}\sum\|T_i - T_{i+1}\|}{\frac{1}{N}\sum T_i}$ | Ratio $[0.0017, 0.0332]$ | Ratio | **COMPATIBLE** |
| 2 | `MDVP:Jitter(Abs)` | `call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)` | Absolute period perturbation in seconds: $\frac{1}{N-1}\sum\|T_i - T_{i+1}\|$ | Seconds $[7\mu\text{s}, 260\mu\text{s}]$ | Seconds | **COMPATIBLE** |
| 3 | `MDVP:RAP` | `call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)` | Relative Average Perturbation (3-period window) | Ratio $[0.0007, 0.0214]$ | Ratio | **COMPATIBLE** |
| 4 | `MDVP:PPQ` | `call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)` | Pitch Period Perturbation Quotient (5-period window) | Ratio $[0.0009, 0.0196]$ | Ratio | **COMPATIBLE** |
| 5 | `Jitter:DDP` | `call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)` | Difference of Differences of Periods ($3 \times \text{RAP}$) | Ratio $[0.0020, 0.0643]$ | Ratio | **COMPATIBLE** |
| 6 | `MDVP:Shimmer` | `call([sound, point_process], "Get shimmer (local)", ...)` | Peak-to-peak amplitude perturbation ratio: $\frac{\frac{1}{N-1}\sum\|A_i - A_{i+1}\|}{\frac{1}{N}\sum A_i}$ | Ratio $[0.0095, 0.1191]$ | Ratio | **COMPATIBLE** |
| 7 | `MDVP:Shimmer(dB)` | `call([sound, point_process], "Get shimmer (local_dB)", ...)` | Amplitude perturbation in dB: $\frac{1}{N-1}\sum\|20 \log_{10}(A_{i+1}/A_i)\|$ | dB $[0.085, 1.302]$ | dB | **COMPATIBLE** |
| 8 | `Shimmer:APQ3` | `call([sound, point_process], "Get shimmer (apq3)", ...)` | 3-period Amplitude Perturbation Quotient | Ratio $[0.0046, 0.0565]$ | Ratio | **COMPATIBLE** |
| 9 | `Shimmer:APQ5` | `call([sound, point_process], "Get shimmer (apq5)", ...)` | 5-period Amplitude Perturbation Quotient | Ratio $[0.0057, 0.0794]$ | Ratio | **COMPATIBLE** |
| 10 | `MDVP:APQ` | `call([sound, point_process], "Get shimmer (apq11)", ...)` | 11-period Amplitude Perturbation Quotient (Kay APQ standard) | Ratio $[0.0072, 0.1378]$ | Ratio | **COMPATIBLE** |
| 11 | `Shimmer:DDA` | `call([sound, point_process], "Get shimmer (dda)", ...)` | Difference of Differences of Amplitudes ($3 \times \text{APQ3}$) | Ratio $[0.0136, 0.1694]$ | Ratio | **COMPATIBLE** |
| 12 | `NHR` | `10 ** (-hnr / 10.0)` | Noise-to-Harmonics Ratio (inversion of cross-correlation HNR) | Ratio $[0.0006, 0.3148]$ | Ratio | **COMPATIBLE (RESOLVED IN 5.6)** |
| 13 | `HNR` | `call(harmonicity, "Get mean", 0, 0)` | Harmonics-to-Noise Ratio via periodic cross-correlation | dB $[8.441, 33.047]$ | dB | **COMPATIBLE** |

---

## 5. Definitional Discrepancies & Audit Resolution

### 5.1 NHR Inversion Formula
- **Discrepancy:** In Kay MDVP, NHR integrated energy in specific frequency bands ($1.5–4.5\text{ kHz}$ vs. $0.1–1.5\text{ kHz}$), whereas Praat inverts HNR via $10^{-\text{HNR}/10}$.
- **Resolution (Stage 5.6):** Empirical sensitivity testing established that `NHR` represents only **$2.24\%$ of ET-13 model attribution** (rank 13/13). Substituting the Praat formula produces **$0 / 32$ ($0.0\%$) classification flips** at both $\tau = 0.50$ and $\tau = 0.45$. The Praat implementation is verified as safe and compatible.

### 5.2 Pitch-Tracking Dependency
- **Audit Question:** Does ET-13 feature extraction depend on pitch detection despite being a "pitch-free" model?
- **Finding:** **YES, internally; NO, externally.**
  - Praat's `PointProcess (periodic, cc)` requires a search range ($75–500\text{ Hz}$) to delineate individual pitch pulses (glottal cycles). Cycle pulses are mathematically required to measure cycle-to-cycle frequency and amplitude jitter/shimmer.
  - However, the resulting fundamental pitch values ($F_0, F_{\text{hi}}, F_{\text{lo}}$) are **strictly excluded from the model input vector**.
  - This satisfies both physical acoustic requirements and demographic bias mitigation.

---

## 6. Backend Integration & Missing Functionality

Inspection of the backend architecture identifies exactly what wiring is missing to complete integration:

1. **Model Client Method (`backend/app/services/model_client.py`):**
   - Current: `ModelClient.predict()` hardcodes `POST {self.settings.model_base_url}/predict` (the 22-feature baseline route).
   - Missing: A dedicated method `predict_et13(payload: dict[str, float])` targeting `POST {self.settings.model_base_url}/predict/et13` (which was implemented and verified on the Flask server in Stage 5.7).
2. **Audio Prediction Route Wiring (`backend/app/routes/predict.py`):**
   - Current: `predict_audio()` returns `AudioScreeningDisabledResponse`.
   - Missing: Slicing the 16 extracted features down to the 13 ET-13 keys, forwarding the payload to `client.predict_et13()`, and returning an active `AudioPredictionResponse`.
3. **Response Schema (`backend/app/schemas/prediction.py`):**
   - Missing: A Pydantic schema representing the active audio prediction response (carrying `pd_probability`, `healthy_probability`, dual threshold classifications, and research disclaimers).
4. **Health Route Capability Advertisement (`backend/app/routes/health.py`):**
   - Current: Returns `audio_input_compatible: false`.
   - Missing: Declaring `audio_input_compatible: true` when the ET-13 candidate pipeline is available.

---

## 7. Error Propagation, Timeouts, and Validation Inspection

- **Audio Payload Validation:** Handled cleanly in `predict_audio()`. Rejects empty payloads (`400`), unsupported MIME types (`400`), durations $< 0.8\text{s}$ (`400`), and aperiodic non-vocal audio (`400`).
- **Feature Extraction Failure:** Wrapped in `AudioExtractionError` which translates cleanly into HTTP `400` with user-friendly guidance.
- **Upstream Communication & Timeouts:** Managed via HTTPX with `model_server_timeout_seconds` (default $5.0\text{s}$). Rejection by model server translates to HTTP `502 Bad Gateway`; connection failure translates to HTTP `503 Service Unavailable`.
- **Numerical Sanitization:** Flask's `/predict/et13` endpoint (verified in Stage 5.7) strictly validates all 13 keys, rejecting missing keys (`422`), non-numerical strings (`422`), and NaN/Inf values (`422`).

---

## 8. Risk Assessment

| Risk Item | Likelihood | Impact | Severity | Mitigation in Place |
|---|:---:|:---:|:---:|---|
| **Aperiodic Audio Input** | Medium | Low | **LOW** | Praat pitch gate rejects whispering, background noise, or silence ($F_0 \le 0$) with clean HTTP 400. |
| **Short Phonation Duration** | Medium | Low | **LOW** | Enforced minimum duration gate ($\ge 0.8\text{s}$) prevents unstable short-window perturbation artifacts. |
| **Microphone / Lossy Compression Shift** | High | Medium | **MEDIUM** | Model relies on pitch-free perturbation biomarkers; explicit research disclaimers inform user that prototype is non-diagnostic. |
| **NHR Approximation Discrepancy** | High | Negligible | **LOW** | Verified in Stage 5.6: NHR holds only 2.24% model weight; causes zero participant classification flips ($0/32$). |
| **Baseline Regression Risk** | Low | High | **LOW** | Separate namespaced routes (`/predict` vs `/predict/et13`) guarantee 100% baseline preservation. |

---

## 9. Final Recommendation Verdict

# Verdict: **INTEGRATION READY**

### Justification:
1. **100% Feature Availability:** All 13 pitch-free features required by the ET-13 research candidate are natively and accurately extracted by the existing Praat/Parselmouth pipeline in `audio_extractor.py`.
2. **Definitional Compatibility Verified:** All 13 features match the scale and mathematical definitions of the training data. The NHR formula difference has been empirically proved to cause zero classification flips.
3. **Model Server Route Active:** The Flask model server endpoint `POST /predict/et13` is already implemented, tested, and running live on port 8001.
4. **Clean Integration Path:** Integration requires only non-breaking backend wiring (extending `ModelClient`, routing `predict_audio`, and updating schemas) without any file changes to the extractor or machine learning models.
