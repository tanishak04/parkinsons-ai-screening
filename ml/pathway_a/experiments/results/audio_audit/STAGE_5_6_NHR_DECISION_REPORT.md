# Stage 5.6 — Time-Boxed NHR Compatibility Decision Report

**Project:** Parkinson's AI Voice Screening System (Pathway A — UCI Dataset)  
**Execution Timestamp:** 2026-11-25 10:44:00 UTC  
**Audit Objective:** Time-boxed empirical resolution of the NHR compatibility blocker identified in Stage 5.5 to enable immediate application integration.  
**Decision Verdict:** **RESOLVED — RETAIN PRAAT-DERIVED NHR UNCHANGED (READY FOR INTEGRATION)**

---

## 1. Executive Summary & Core Decision

In Stage 5.5, a potential blocker was identified regarding the Noise-to-Harmonics Ratio (`NHR`):
- In the original UCI dataset, Kay Elemetrics MDVP computed NHR via high-frequency spectral band energy ratios ($1.5–4.5\text{ kHz}$ vs. $0.1–1.5\text{ kHz}$).
- In `audio_extractor.py`, Praat/Parselmouth calculates NHR using the acoustic formula:
  $$\text{NHR} = 10^{-\text{HNR} / 10}$$
- On the 195 recordings of the UCI dataset, this creates a systematic scale difference ($r = 0.893$, raw values on average $2.30\times$ higher).

### The Decision: **RETAIN UNCHANGED**
An empirical sensitivity analysis demonstrates that **NHR is the least influential feature in the ET-13 candidate model**, accounting for only **$2.24\%$** of total feature importance (rank 13 of 13).

Replacing raw UCI NHR with the Praat formula across the entire 195-recording dataset causes:
- **Zero participant classification flips at reference threshold $\tau = 0.50$ ($0 / 32$, $0.0\%$)**
- **Zero participant classification flips at exploratory threshold $\tau = 0.45$ ($0 / 32$, $0.0\%$)**
- Mean participant probability change of only **$0.0018$** (less than $0.2\%$).
- Maximum participant probability shift of **$0.0061$**.

Because the formula discrepancy has **zero impact on participant-level screening classifications**, no arbitrary correction factor, blind multiplier, or model retraining is required. Praat's current implementation in `audio_extractor.py` is deemed **sufficiently compatible for research prototype integration**.

---

## 2. Evidence Inspected

### 2.1 Feature Importance Hierarchy in ET-13 (`et13_pitch_free_candidate.joblib`)

The ExtraTrees classifier (`max_depth=3, min_samples_leaf=3, n_estimators=100`) heavily prioritizes amplitude perturbations (shimmers) and short-term frequency perturbation (jitter absolute):

| Rank | Feature Name | Clinical Biomarker Category | Feature Importance | Share of Model Attribution |
|:---:|---|---|---:|---:|
| 1 | `Shimmer:APQ5` | Amplitude Perturbation (5-cycle) | 0.1166 | 11.7% |
| 2 | `MDVP:APQ` | Amplitude Perturbation (11-cycle) | 0.1160 | 11.6% |
| 3 | `MDVP:Shimmer` | Amplitude Perturbation (Local) | 0.1125 | 11.3% |
| 4 | `MDVP:Jitter(Abs)` | Frequency Perturbation (Absolute) | 0.1053 | 10.5% |
| 5 | `Shimmer:DDA` | Amplitude Perturbation ($3 \times \text{APQ3}$) | 0.0813 | 8.1% |
| 6 | `MDVP:Shimmer(dB)` | Amplitude Perturbation (dB) | 0.0771 | 7.7% |
| 7 | `MDVP:Jitter(%)` | Frequency Perturbation (Local) | 0.0748 | 7.5% |
| 8 | `Shimmer:APQ3` | Amplitude Perturbation (3-cycle) | 0.0658 | 6.6% |
| 9 | `MDVP:RAP` | Frequency Perturbation (RAP) | 0.0592 | 5.9% |
| 10 | `Jitter:DDP` | Frequency Perturbation ($3 \times \text{RAP}$) | 0.0591 | 5.9% |
| 11 | `MDVP:PPQ` | Frequency Perturbation (5-cycle) | 0.0557 | 5.6% |
| 12 | `HNR` | Harmonics-to-Noise Ratio (dB) | 0.0542 | 5.4% |
| **13** | **`NHR`** | **Noise-to-Harmonics Ratio** | **0.0224** | **2.2%** |

*Finding:* The top 4 perturbation features account for over $45\%$ of model decisions. NHR accounts for only $2.2\%$.

---

### 2.2 Empirical Sensitivity Test (Impact of NHR Formula on Predictions)

To measure the real-world consequence of the formula divergence, the ET-13 candidate pipeline was evaluated on the raw UCI dataset with:
1. Original Kay MDVP NHR values ($X_{\text{orig}}$).
2. Praat-formula NHR values ($X_{\text{mod}}$, where $\text{NHR} = 10^{-\text{HNR}/10}$).

#### Recording-Level Results ($N = 195$):
- Mean Absolute Probability Change: **$0.0025$** ($0.25\%$)
- 95th Percentile Change: **$0.0092$** ($0.92\%$)
- Maximum Absolute Probability Change: **$0.0160$** ($1.60\%$)
- Prediction Flips at $\tau = 0.50$: **$0 / 195$ ($0.0\%$)**
- Prediction Flips at $\tau = 0.45$: **$1 / 195$ ($0.5\%$)**

#### Participant-Level Results ($N = 32$):
- Mean Subject Probability Change: **$0.0018$**
- Maximum Subject Probability Change: **$0.0061$**
- **Participant Classification Flips at $\tau = 0.50$:** **$0 / 32$ ($0.0\%$)**
- **Participant Classification Flips at $\tau = 0.45$:** **$0 / 32$ ($0.0\%$)**

*Conclusion:* Across all 32 subjects in the dataset, **not a single classification changes**. The decision margin of the ensemble is completely dominated by the 12 other features (especially shimmers and jitter absolute).

---

## 3. Evaluation of Alternatives Considered

| Option | Description | Technical Feasibility | Clinical / Scientific Validity | Recommendation Verdict |
|---|---|:---:|:---:|:---:|
| **Option 1: Retain Unchanged (Chosen)** | Keep Praat's current $\text{NHR} = 10^{-\text{HNR}/10}$ implementation in `audio_extractor.py`. | **Immediate** (zero code changes needed) | **High** (Standard acoustic formula; zero subject flips) | **APPROVED** |
| **Option 2: Apply a $2.3\times$ Multiplier** | Scale Praat NHR by the average empirical ratio observed in UCI. | Trivial | **Rejected** (Unprincipled heuristic; violates audit task constraint #5) | **REJECTED** |
| **Option 3: Alternative Recalculation** | Implement custom spectral band integration or Praat Voice Report parsing. | Complex (requires raw FFT band-pass filter re-implementation) | **Marginal** (No perceptible performance gain on a feature with $2.2\%$ weight) | **REJECTED** |
| **Option 4: Retrain Model Without NHR** | Exclude NHR and train a 12-feature model. | High | **Out of Scope** (Violates frozen candidate constraint; requires full re-audit) | **REJECTED** |

---

## 4. Remaining Risks & Pragmatic Mitigations

1. **Acoustic Background Noise in Consumer Microphones (Low-to-Medium Risk):**  
   - In noisy home environments, ambient noise can lower HNR, artificially elevating Praat's NHR.
   - *Mitigation:* Because NHR has only $2.2\%$ weight, background noise cannot single-handedly drive a false positive. Furthermore, `audio_extractor.py` enforces a minimum pitch periodicity check (`To PointProcess (periodic, cc)`), rejecting noisy recordings that lack sustained vocal periodicity.
2. **Research Framing (Governance Mitigation):**  
   - The user-facing application must maintain explicit framing: *"Investigational machine-learning screening prototype based on acoustic voice perturbation. Outputs do not constitute a clinical diagnosis."*

---

## 5. Integration Action Plan

With the NHR blocker officially resolved, integration of the ET-13 research candidate into the application can proceed along the following concrete path:

1. **Model Server Routing (`ml/pathway_a/server.py` & `predict.py`):**
   - Add support to load `et13_pitch_free_candidate.joblib` alongside `baseline_pipeline.joblib`.
   - Expose explicit model routing (e.g., `POST /predict?model=et13`).
2. **Backend Unblocking (`backend/app/routes/predict.py`):**
   - Update `POST /predict/audio` to map the 16 Praat features down to the 13 ET-13 features and forward them to the ET-13 server endpoint.
   - Update `app/routes/health.py` to declare `audio_input_compatible: true` for the ET-13 research candidate.
3. **Frontend Presentation (`ResultsPage.jsx`):**
   - Transition the UI from "Screening Disabled" to the active research result view with dual operating point displays ($\tau = 0.50$ and $\tau = 0.45$).

---

## 6. Audit Verdict & Sign-Off

# Status: **STAGE 5.6 COMPLETE — BLOCKER RESOLVED**

The NHR discrepancy has been investigated, quantified, and empirically verified as harmless to participant-level classification. Praat-derived NHR is approved to remain unchanged. **Stage 5.6 is officially concluded, and Stage 6 application integration is ready to commence immediately.**
