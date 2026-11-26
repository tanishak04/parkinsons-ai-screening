# Phase 3.1 — Model & Artifact Integrity Audit Report
## Pathway A: Parkinson's AI Voice Screening Application

**Audit Date:** 2026-11-25  
**Stage:** Phase 3.1 (Validation and Freeze)  
**Audit Scope:** Production Flask Server, FastAPI Backend, Frontend Interface, Saved Artifacts, and Research Champion Alignment  
**Audit Mode:** Strictly READ-ONLY (No code, models, configs, or artifacts altered)  

---

### 1. Actual Production Model Identification

An inspection of the live running service (`http://127.0.0.1:8001/health` and `predict.py`) reveals:
- **Active Model File:** `ml/pathway_a/models/baseline_pipeline.joblib`
- **Model Version:** `0.1.0-baseline`
- **Algorithm:** `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)` preceded by `StandardScaler()`
- **Feature Space:** 22 numerical features (expects all 22 tabular columns from `parkinsons.data`)
- **Decision Threshold:** 0.50 (logistic sigmoid)
- **Class Mapping:** `0 = HEALTHY`, `1 = PARKINSONS`

---

### 2. Research Champion vs. Production Model Comparison

| Characteristic | Phase 2 Research Champion | Live Production Model (`server.py`) | Frozen Disk Champion (`champion_pipeline.joblib`) |
|---|---|---|---|
| **Algorithm** | `ExtraTreesClassifier` | `LogisticRegression` (with `StandardScaler`) | `ExtraTreesClassifier` |
| **Hyperparameters** | `max_depth=3`, `min_samples_leaf=3`, `n_estimators=100`, `class_weight='balanced'` | `class_weight='balanced'`, `max_iter=1000` | `max_depth=4`, `min_samples_leaf=3`, `n_estimators=100`, `class_weight='balanced'` |
| **Feature Count** | **13 features** (Pitch-free) | **22 features** (Legacy tabular) | **22 features** (Full tabular) |
| **Pitch Features** | **EXCLUDED** (`Fo`, `Fhi`, `Flo` removed) | **INCLUDED** (Confounded) | **INCLUDED** (Confounded) |
| **Nonlinear Features** | **EXCLUDED** (`RPDE`, `DFA`, `spread1/2`, `D2`, `PPE` removed) | **INCLUDED** | **INCLUDED** |
| **Participant Bal Acc** | **0.7917** | 0.6667 | 0.7708 |
| **Participant Specificity**| **0.8750** (7/8 controls correct) | 0.5000 (4/8 controls correct) | 0.6250 (5/8 controls correct) |
| **Deployment Status** | **Research only** (Not serialized in `models/`) | **ACTIVE in production server** | **Serialized on disk, but inactive** |

---

### 3. Feature Extraction & Ordering Verification

1. **Extractable Features in Production Audio Extractor:**
   - Location: `backend/app/services/audio_extractor.py:AudioFeatureExtractor`
   - Uses: Parselmouth / Praat
   - Extracts: **16 features** (`MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`, and 13 jitter/shimmer/HNR measures).
2. **Alignment with 13-Feature Champion:**
   - The 13 features selected for the research champion are a **100% exact subset** of the 16 features extracted by `AudioFeatureExtractor`.
   - Feature names, formulas, and units match `parkinsons.data` column definitions.
   - The 3 excluded features are precisely the 3 pitch features (`Fo`, `Fhi`, `Flo`) that caused false-positive demographic confounding on healthy male voices.
   - The 6 unavailable non-linear dynamical features (`RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`) are **not required** by the 13-feature champion.
3. **Current Incompatibility:**
   - `predict.py` requires all 22 features. Passing the 13 features directly to `predict.py` triggers a `ValueError("Missing required features: [...]")`.

---

### 4. Artifact & Hash Verification

SHA-256 hashes of all artifacts were computed and verified against existing manifests:

| Artifact Path | SHA-256 Hash | Integrity Status |
|---|---|:---:|
| `ml/pathway_a/models/champion_pipeline.joblib` | `b2d052eed9c2515506a244b401dfb78e26acd8d24fa0e8ed68dfadc5aae7a414` | **VERIFIED (Unchanged)** |
| `ml/pathway_a/models/baseline_pipeline.joblib` | `81dc11f063a7bd52b404f36621d2a37bb33655456f209df1401eefd56b6b884f` | **VERIFIED (Unchanged)** |
| `ml/pathway_a/models/audio_pipeline.joblib` | `7024bc988db633bddea41970a0985d5cb4a779add54e3b3d08f54af52ff5e01f` | **VERIFIED (Unchanged)** |
| `ml/pathway_a/models/champion_manifest.json` | Manifest v1.0.0 (Stage 5 freeze) | **VERIFIED (Unchanged)** |
| `ml/pathway_a/models/baseline_metrics.json` | Baseline metrics record | **VERIFIED (Unchanged)** |

---

### 5. Complete Inference Pipeline Trace

#### Trace A: Audio Upload Route (`POST /predict/audio`)
1. **Frontend:** User records or uploads audio on `ScreeningPage.jsx`.
2. **Pre-Flight Health Check:** Frontend queries `GET /health` on FastAPI (`http://localhost:8000/health`).
   - Receives `capabilities.audio_input_compatible: false`.
   - Displays: *"Research extraction active · Screening disabled"*.
3. **Request:** Frontend sends audio via multipart/form-data to `POST /predict/audio`.
4. **Backend Processing:**
   - Validates MIME type and audio length ($\ge 0.8$s).
   - `AudioFeatureExtractor.extract_features()` decodes audio via `soundfile`/`parselmouth`.
   - Computes 16 acoustic vocal biomarkers.
5. **Screening Gate:**
   - Returns `AudioScreeningDisabledResponse` with `screening_available: false` and `status: "screening_disabled"`.
   - **No model inference is invoked.**
6. **Frontend Display:** `ResultsPage.jsx` renders *"AUTOMATED SCREENING DISABLED"* with clinical validation rationale and a grid of raw extracted features.

#### Trace B: Numerical Feature Route (`POST /predict`)
1. **Request:** Client submits 22-feature JSON to FastAPI `POST /predict`.
2. **Schema Validation:** `PredictionRequest` validates 22 non-null float fields.
3. **Proxy:** `ModelClient` forwards JSON payload to Flask `http://127.0.0.1:8001/predict`.
4. **Model Execution:** `server.py` invokes `predict_single()` which loads `baseline_pipeline.joblib`.
5. **StandardScaler & LogisticRegression:** Scales features $\to$ computes logit $\to$ outputs `prediction: 0|1`, `label: "HEALTHY"|"PARKINSONS"`, and probabilities.
6. **Response:** FastAPI normalizes to `PredictionResponse` and returns to client.

---

### 6. Identified Discrepancies and Risks

1. **Discrepancy 1 (Active Server Serves Legacy Model):**
   The Flask server is serving the Stage 1 baseline (`LogisticRegression`, 22 features), **not** the research champion (`ExtraTrees`, 13 features) and **not** even the Stage 5 frozen champion (`champion_pipeline.joblib`).
2. **Discrepancy 2 (Research Champion Is Not Serialized):**
   The 13-feature pitch-free ExtraTrees model validated in Phase 2 exists only within experiment training scripts; no serialized pipeline artifact or manifest currently exists in `ml/pathway_a/models/` for this champion.
3. **Discrepancy 3 (Audio Screening Is Completely Deactivated):**
   The audio pipeline cannot generate predictions because it was disabled following the Phase 1 forensic audit of the 16-feature model.
4. **Discrepancy 4 (API Schema Incompatibility):**
   The FastAPI `PredictionRequest` schema enforces 22 required fields, which prevents submitting the 13-feature vector without providing dummy or synthetic values for the 9 excluded features.

---

### 7. Audit Verdict

### **VERDICT: PASS WITH ISSUES**

- **Safety & Guardrails:** **PASS**
  - Unvalidated models are safely blocked from making false predictions.
  - Live audio screening is explicitly deactivated.
  - No fabricated labels or probabilities are returned.
  - All disk artifacts and SHA-256 hashes are 100% verified and uncorrupted.
- **Production Alignment:** **ISSUES IDENTIFIED**
  - The application is currently serving the legacy 22-feature baseline rather than the 13-feature research champion.
  - In Phase 3.2, a safe, backward-compatible migration plan will be required to serialize the ET-13 research champion and connect the Parselmouth feature extractor without breaking the existing 22-feature API contract.
