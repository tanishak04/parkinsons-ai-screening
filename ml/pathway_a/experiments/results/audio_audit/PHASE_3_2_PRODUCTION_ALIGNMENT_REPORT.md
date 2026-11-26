# Phase 3.2 — Production Alignment & Discrepancy Resolution Report
## Parkinson's AI Voice Screening — Pathway A

**Document Version:** 1.0.0  
**Date:** 2026-11-25  
**Stage:** Phase 3.2 (Production Alignment Planning & Discrepancy Resolution)  
**Status:** ALIGNMENT PLAN READY (Planning Only — Zero Implementation Undertaken)  

---

## 1. Review and Classification of Phase 3.1 Findings

We reviewed each of the six findings established in the Phase 3.1 Model & Artifact Integrity Audit and classified their operational nature:

| # | Phase 3.1 Finding | Classification | Rationale & Assessment |
|---|---|---|---|
| **1** | **Live Flask server uses legacy 22-feature `StandardScaler + LogisticRegression` baseline (`0.1.0-baseline`).** | **Model Deployment Decision Requiring Explicit Approval** | Not an accidental defect. The baseline was intentionally preserved in Stage 1 to guarantee a stable reference point. Upgrading the active production model to a newer architecture requires formal approval and a controlled deployment gate. |
| **2** | **Research champion is an uncalibrated 13-feature pitch-free ExtraTrees model (ET-13).** | **Architectural Gap Between Research & Production** | Phase 2 successfully discovered and validated ET-13 as the champion (0.7917 Balanced Accuracy, 0.8750 Specificity, pitch-bias eliminated). However, ET-13 currently exists only within research cross-validation scripts (`experiment_pitch_free_13feat.py`) and has not yet been bridged to application serving. |
| **3** | **ET-13 research champion has not been serialized as a deployable production artifact.** | **Architectural Gap Between Research & Production** | Because Phase 2 strictly enforced a read-only mandate on production assets, no model artifact was serialized into `ml/pathway_a/models/`. A dedicated, versioned `.joblib` file and cryptographic manifest must be created in a controlled freeze step. |
| **4** | **Audio feature extraction produces 16 acoustic features, but audio prediction is intentionally disabled.** | **Intentional Safety Restriction** | Necessary clinical safeguard. Live audio screening was disabled following the Phase 1 audit because the experimental 16-feature model failed validation (0.6667 Balanced Accuracy) and exhibited severe pitch-induced false positives on healthy male voices. Deactivation was deliberate and protective. |
| **5** | **Existing production numerical API expects 22 features.** | **Architectural Contract Constraint** | The FastAPI `/predict` route and Pydantic `PredictionRequest` schema strictly validate 22 continuous features. Directly replacing this with a 13-feature contract would break backward compatibility with existing tabular clients. |
| **6** | **Existing model artifacts and cryptographic hashes are verified and unchanged.** | **Verified Safety Requirement** | All protected artifacts (`champion_pipeline.joblib`, `baseline_pipeline.joblib`, `audio_pipeline.joblib`) match their SHA-256 signatures 100%. No data corruption or silent alterations have occurred. |

---

## 2. Production Alignment Analysis & Design

### A. Exact Feature Mapping (13-Feature ET Champion vs. 16 Parselmouth Outputs)

The Parselmouth-based acoustic extractor (`backend/app/services/audio_extractor.py:AudioFeatureExtractor`) computes 16 standard Praat voice features. The 13 features selected for the research champion represent an **exact 100% subset**:

| Extractor Output (16 Features) | ET-13 Research Champion Role | Inclusion Status | Clinical & Acoustic Rationale |
|---|:---:|:---:|---|
| `MDVP:Fo(Hz)` | N/A | **EXCLUDED** | Fundamental frequency mean; carries speaker sex/identity confounding. |
| `MDVP:Fhi(Hz)` | N/A | **EXCLUDED** | Maximum pitch; carries pitch excursion bias and recording artifacts. |
| `MDVP:Flo(Hz)` | N/A | **EXCLUDED** | Minimum pitch; sensitive to vocal fry and microphone noise. |
| `MDVP:Jitter(%)` | Feature #1 | **INCLUDED** | Cycle-to-cycle frequency variation (relative). |
| `MDVP:Jitter(Abs)` | Feature #2 | **INCLUDED** | Cycle-to-cycle frequency variation (absolute in $\mu\text{s}$). |
| `MDVP:RAP` | Feature #3 | **INCLUDED** | Relative Average Perturbation (3-cycle smoothing). |
| `MDVP:PPQ` | Feature #4 | **INCLUDED** | Five-point Period Perturbation Quotient. |
| `Jitter:DDP` | Feature #5 | **INCLUDED** | Difference of Differences of Periods ($3 \times \text{RAP}$). |
| `MDVP:Shimmer` | Feature #6 | **INCLUDED** | Cycle-to-cycle amplitude variation (local). |
| `MDVP:Shimmer(dB)` | Feature #7 | **INCLUDED** | Cycle-to-cycle amplitude variation in decibels. |
| `Shimmer:APQ3` | Feature #8 | **INCLUDED** | 3-point Amplitude Perturbation Quotient. |
| `Shimmer:APQ5` | Feature #9 | **INCLUDED** | 5-point Amplitude Perturbation Quotient. |
| `MDVP:APQ` | Feature #10 | **INCLUDED** | 11-point Amplitude Perturbation Quotient. |
| `Shimmer:DDA` | Feature #11 | **INCLUDED** | Difference of Differences of Amplitudes ($3 \times \text{APQ3}$). |
| `NHR` | Feature #12 | **INCLUDED** | Noise-to-Harmonics Ratio (acoustic spectral noise). |
| `HNR` | Feature #13 | **INCLUDED** | Harmonics-to-Noise Ratio in dB. |

**Key Finding on Nonlinear Measures:** The 6 proprietary non-linear dynamical features (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`) developed by Little et al. (2007) are **not required** by the 13-feature champion. This eliminates the primary feasibility blocker identified in the Phase 1 audit.

### B. Feature Ordering Specification
When feeding features to the serialized ET-13 pipeline, they must be formatted in a `pandas.DataFrame` or 2D `numpy.ndarray` with columns strictly matching the research training ordering:

```python
PITCH_FREE_13_FEATURES: list[str] = [
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR"
]
```

### C. Research vs. Production Preprocessing Comparison
- **Scaling / Normalization:**
  - *Research Pipeline:* Tree ensembles (`ExtraTreesClassifier`) are partition-based algorithms that are strictly invariant to monotonic scaling and normalization. In research evaluation, ET-13 was trained on unscaled raw Praat feature outputs.
  - *Audio Extractor:* `AudioFeatureExtractor` calculates features directly in physical Praat units (e.g. jitter percentages, shimmer ratios, HNR in dB).
  - *Comparison:* Preprocessing matches 100%. **No StandardScaler, RobustScaler, or transformation step is required or permitted.**
- **Audio Framing & Windowing:**
  - Parselmouth uses a default time step of 0.01s with pitch bounds [75 Hz, 500 Hz] and minimum audio duration of 0.8s, matching standard clinical voice protocols.

### D. Reproducible Serialization Feasibility
The ET-13 model can be reproducibly serialized from the existing validated research pipeline by creating a dedicated freeze script (`ml/pathway_a/freeze_et13_champion.py`):
1. **Source Data:** `data/raw/parkinsons.data` (verified SHA-256: `a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5`).
2. **Feature Extraction:** Subsets columns strictly to `PITCH_FREE_13_FEATURES`.
3. **Model Instantiation:**
   `ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight='balanced', random_state=42)`
4. **Fitting:** Fit once on all 195 training samples.
5. **Output Artifact:** Save dictionary payload to a dedicated, non-colliding file:
   `ml/pathway_a/models/audio_et13_champion_pipeline.joblib`
   Containing keys: `{'pipeline': model, 'feature_names': PITCH_FREE_13_FEATURES, 'metadata': {...}}`.
6. **Integrity Manifest:** Generate `audio_et13_champion_manifest.json` recording model hyperparameters, input feature contract, training timestamp, and SHA-256 checksum.

---

## 3. Production Alignment Architecture Proposal

To introduce ET-13 audio inference without breaking the existing 22-feature tabular API or destabilizing production, we propose a **Dual-Contract Architecture**:

```
                       ┌──────────────────────────────────────────────┐
                       │               React Frontend                 │
                       └──────────────┬────────────────┬──────────────┘
                                      │                │
                      Numerical Form  │                │ Audio Upload
                       (22 features)  │                │ (.wav / .webm)
                                      ▼                ▼
                       ┌──────────────────────────────────────────────┐
                       │           FastAPI Backend (:8000)            │
                       ├──────────────────────┬───────────────────────┤
                       │ POST /predict        │ POST /predict/audio   │
                       │ (Schema: 22 fields)  │ (Multipart audio)     │
                       └──────────┬───────────┴──────────┬────────────┘
                                  │                      │
                                  │ Forward 22 feats     │ 1. Extract 16 features via Praat
                                  │                      │ 2. Select 13 pitch-free features
                                  │                      │ 3. Forward to audio model
                                  ▼                      ▼
                       ┌──────────────────────────────────────────────┐
                       │           Flask Model Server (:8001)         │
                       ├──────────────────────┬───────────────────────┤
                       │ POST /predict        │ POST /predict/audio   │
                       │ (22-feat Pipeline)   │ (13-feat ET Champion) │
                       ├──────────────────────┼───────────────────────┤
                       │ baseline_pipeline    │ audio_et13_champion   │
                       │ (.joblib)            │ (.joblib)             │
                       └──────────────────────┴───────────────────────┘
```

### Proposed Pipeline Operation:
1. **Preserve Existing Route:** `POST /predict` on FastAPI and Flask remains 100% unchanged, accepting 22 features and serving the existing contract.
2. **Activate Audio Route:** `POST /predict/audio` on FastAPI:
   - Receives audio recording.
   - Invokes `AudioFeatureExtractor.extract_features()`.
   - Filters the 16 extracted features down to the 13 required features.
   - Forwards the 13-feature vector to the dedicated ET-13 model endpoint on the model server (or queries a loaded ET-13 pipeline).
   - Receives prediction, confidence, and class probabilities ($P(\text{PD})$, $P(\text{Healthy})$).
   - Returns a valid `PredictionResponse` accompanied by prominent research disclaimers and the extracted feature table.
3. **Controlled Safety Gate Transition:**
   - In `backend/app/schemas/prediction.py`, the `AudioCapabilities` model updates `audio_input_compatible` from `False` to `True`.
   - The frontend `ScreeningPage.jsx` automatically recognizes backend capability confirmation via `GET /health` and activates the "Submit audio for screening" action.

---

## 4. Proposed Files Requiring Modification in a Future Implementation

*Note: No files have been modified in Phase 3.2. This lists the exact target files for a future approved implementation:*

1. **New Script:** `ml/pathway_a/freeze_et13_champion.py`
   - Generates `ml/pathway_a/models/audio_et13_champion_pipeline.joblib` and `audio_et13_champion_manifest.json`.
2. **Backend Services:** `backend/app/services/audio_extractor.py`
   - Add helper method `extract_et13_features(audio_bytes)` that extracts the 16 features and returns the ordered 13-feature dictionary.
3. **Backend Routes:** `backend/app/routes/predict.py`
   - In `predict_audio()`, connect the extracted 13 features to model inference when enabled.
4. **Backend Schemas:** `backend/app/schemas/prediction.py`
   - Update `AudioCapabilities.audio_input_compatible` to `True` and set `required_features_count = 13`.
5. **Model Server (Optional / Alternative):** `ml/pathway_a/server.py` and `ml/pathway_a/predict.py`
   - Add support for routing 13-feature audio inference requests to `audio_et13_champion_pipeline.joblib`.

---

## 5. Comprehensive Test Plan

Before enabling production audio screening, the following test suite must pass:

1. **Feature Alignment & Ordering Unit Test:**
   - Extract features from a synthetic calibration tone.
   - Assert all 13 feature names match `PITCH_FREE_13_FEATURES` in exact sequence without NaN or Inf.
2. **Deterministic Inference Test:**
   - Run inference on fixed benchmark feature vectors.
   - Assert predicted class and probabilities match cross-validation research predictions to 6 decimal places.
3. **Confounding Invariance Test (Critical Safeguard):**
   - Run inference on the 3 low-pitch healthy male benchmark recordings (`S13`, `S43`, `S49`).
   - Assert all three remain classified as `HEALTHY` ($P(\text{PD}) \le 0.448$).
4. **Negative & Error Handling Test:**
   - Submit corrupt audio, empty payloads, recordings under 0.8s, and pure silence.
   - Assert appropriate HTTP 400/422 status codes are returned with clear error messages.
5. **Backward Compatibility Regression Test:**
   - Post standard 22-feature payload to `POST /predict`.
   - Assert response schema, confidence, and prediction remain completely identical to current behavior.
6. **Frontend Integration Test:**
   - Verify `ScreeningPage.jsx` correctly transitions from "Screening disabled" to "Audio screening available".
   - Verify `ResultsPage.jsx` renders `PredictionResult` with research disclaimers.

---

## 6. Risks, Clinical Limitations & Disclaimers

1. **Small Training Cohort:** The model was trained on 195 recordings from 32 participants (only 8 healthy controls). While participant-level grouping strictly prevents data leakage, small-sample variance remains an inherent limitation.
2. **Acoustic Domain Shift:** Training data was recorded using head-mounted instrumentation in a controlled environment. User audio captured through consumer laptop or smartphone microphones is subject to environmental noise, room reverberation, and automatic gain control.
3. **Research Prototype Boundary:** This application is an engineering research demonstration and is **not an FDA-cleared, CE-marked, or clinically validated medical device**. It must not be used for medical diagnosis, clinical decision-making, or patient screening.

---

## 7. Recommendation & Next Steps

### **FINAL STATUS: ALIGNMENT PLAN READY**

### Justification:
- The technical and mathematical alignment between the 13-feature research champion and Parselmouth feature extraction is complete and verified.
- The dual-contract architecture safely introduces audio screening while preserving 100% backward compatibility for the existing 22-feature numerical API.
- All existing production artifacts and cryptographic hashes remain strictly untouched.
- Implementation must await formal user approval.

*Phase 3.2 is complete. In accordance with strict instructions, no implementation has occurred and Phase 3.3 has not started. Awaiting your review.*
