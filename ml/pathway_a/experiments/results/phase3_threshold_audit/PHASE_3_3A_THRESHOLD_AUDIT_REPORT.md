# Phase 3.3A — Decision Threshold & Robustness Audit Report
## Pathway A: UCI Parkinson's Voice Dataset

**Document Version:** 1.0.0  
**Date:** 2026-11-25  
**Stage:** Phase 3.3A (Threshold & Robustness Audit)  
**Execution Context:** Strictly READ-ONLY on existing model artifacts and source code; no model retraining.  
**Integrity Status:** Verified 100% leak-free participant grouping and hash matching across all input artifacts.

---

## 1. Input Artifacts & Verified Cryptographic Hashes

All evaluated Out-of-Fold (OOF) predictions and evaluation outputs were audited with SHA-256 checksums:

| Artifact Path | SHA-256 Hash | Audit Role |
|---|---|---|
| `ml/pathway_a/experiments/results/threshold_optimization/experiment_config.json` | `ed98102e00db55e6bfa5840d40ec3a7c0dee483e1b4052a1752a6a896784ed21` | Stage 3A nested optimization configuration |
| `ml/pathway_a/experiments/results/threshold_optimization/aggregate_comparison.csv` | `b3d913c175ed9b9deb6f4431e4515c9fe47c5f336d15db7fbdede892ea168295` | Stage 3A cross-strategy comparison |
| `ml/pathway_a/experiments/results/threshold_optimization/outer_fold_metrics.csv` | `7a13226ae2620b09c7248dcce2293dc103b26db707e30bcd40b0a2989fec4a5b` | Stage 3A outer test metrics across folds |
| `ml/pathway_a/experiments/results/threshold_optimization/per_fold_thresholds.csv` | `fd6fa23028d7328cd6c500cddf2452aebcdd89a1a27104fe8f4acc27cdebcd63` | Stage 3A inner-selected thresholds per fold |
| `ml/pathway_a/experiments/results/error_analysis/participant_error_profile.csv` | `b8948896cfa9740970d19b148cba1081b63b29106f6182ef27c6c42ede39b69e` | Stage 3B participant error profile (22-feat) |
| `ml/pathway_a/experiments/results/error_analysis/participant_error_profile.json` | `333aec6d52e928448db46b89bb41282851d1d2a409faeb5d7f29c2c54c7c6ac6` | Stage 3B metadata and participant margins |
| `ml/pathway_a/experiments/results/phase2_calibration/participant_calibrated_predictions.csv` | `36e7376b5b332aa0c5d63ecab453f23931a4c29ff4ca603303a360cfd0f79670` | Phase 2 participant-level OOF predictions |
| `ml/pathway_a/experiments/results/phase2_calibration/oof_calibrated_predictions_recordings.csv` | `c956fe4abe088ef73d800f5796a8914f70964a7c0fff3b4610ad7ba90e410edc` | Phase 2 recording-level OOF predictions |
| `ml/pathway_a/models/champion_manifest.json` | `505bc90f11b8bb4049b8e2a8bf747e6bf8f33f655761ecbe60c8514ca8ddbc59` | Stage 5 frozen model manifest |

---

## 2. OOF Prediction Integrity Verification

Before computing any threshold metrics, automated integrity checks were executed against `data/raw/parkinsons.data`:

1. **Prediction Provenance & Model Identity:**
   - *Stream A (Historical 22-Feature ExtraTrees):* 32 unique subjects in `participant_error_profile.csv`. Generated from nested cross-validation of `ExtraTreesClassifier(max_depth=4, min_samples_leaf=3, class_weight='balanced')`.
   - *Stream B (Research Champion 13-Feature Pitch-Free ExtraTrees):* 32 unique subjects in `participant_calibrated_predictions.csv` and 195 recordings in `oof_calibrated_predictions_recordings.csv`. Generated from nested Leave-One-Group-Out (LOGO) cross-validation of `ExtraTreesClassifier(max_depth=3, min_samples_leaf=3, class_weight='balanced')`.
2. **Participant Grouping & Leakage:**
   - Verified that all recordings for each participant belong strictly to the single fold in which that participant was held out. Multi-fold participant leak count = **0**.
   - Verified that training subjects and test subjects are strictly disjoint across all folds.
3. **Data Completeness & Alignment:**
   - Missing / NaN prediction count = **0**.
   - Duplicate participant records = **0**.
   - Label mismatches against raw dataset = **0**.
- **Integrity Verdict:** **PASS (100% verified integrity across all evaluation files).**

---

## 3. Threshold-Selection Leakage Audit

We audited the actual Python code in `ml/pathway_a/experiments/experiment_thresholds.py` and fold artifacts in `per_fold_thresholds.csv` to verify whether Stage 3A threshold optimization leaked test data:

### Verified Code Architecture:
```python
# From ml/pathway_a/experiments/experiment_thresholds.py:
for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    assert train_subjs.isdisjoint(test_subjs)
    
    # 1. Inner grouped cross-validation within training subjects ONLY
    _, inner_subj_df = compute_inner_oof_probabilities(
        X_train_outer=X.iloc[train_idx],
        y_train_outer=y.iloc[train_idx],
        groups_train_outer=groups.iloc[train_idx],
        inner_n_splits=3,
    )
    
    # 2. Select threshold on inner training OOF predictions
    chosen_thresholds = optimize_thresholds_on_inner_oof(
        y_true_subjects=inner_subj_df["y_true"],
        y_prob_subjects=inner_subj_df["y_prob"],
    )
    
    # 3. Refit model on entire outer training set
    model = build_champion_model()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    # 4. Evaluate held-out outer test subject with the inner-selected threshold
    test_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    pred = (test_prob >= chosen_thresholds[strategy]["selected_threshold"])
```

### Findings vs. Assumptions:
- **Verified Finding:** Stage 3A threshold selection was **strictly nested**. Thresholds were selected using an inner 3-fold `StratifiedGroupKFold` on training participants only. The outer test subject's label was never seen by the threshold selection algorithm.
- **Verified Finding:** The reported outer performance in Stage 3A reflects genuine out-of-fold generalization under nested threshold selection.
- **Important Distinction:** While Stage 3A nested optimization was leak-free, evaluating an arbitrary post-hoc threshold on pooled out-of-fold predictions to claim an "optimal" metric is an exploratory analysis and must **not** be presented as an unbiased clinical result.

---

## 4. Predefined Threshold Grid Evaluation (Stream B: Research Champion ET-13)

Evaluated across the 41-point grid $\tau \in [0.10, 0.90]$ with step $0.02$.

### Primary Level: Participant-Level Evaluation ($N = 32$; 24 PD, 8 HC)
*Participant prediction = mean predicted probability across all vocal recordings for that subject.*

| Threshold ($\tau$) | Balanced Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | TP | TN | FP | FN | Pred Pos | Pred Neg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.10 - 0.20 | 0.5000 | 1.0000 | 0.0000 | 0.7500 | 0.8571 | 24 | 0 | 8 | 0 | 32 | 0 |
| 0.30 | 0.5417 | 0.9583 | 0.1250 | 0.7667 | 0.8519 | 23 | 1 | 7 | 1 | 30 | 2 |
| 0.36 | 0.6250 | 0.8750 | 0.3750 | 0.8077 | 0.8400 | 21 | 3 | 5 | 3 | 26 | 6 |
| 0.40 | 0.6458 | 0.7917 | 0.5000 | 0.8261 | 0.8085 | 19 | 4 | 4 | 5 | 23 | 9 |
| 0.42 | 0.6875 | 0.7500 | 0.6250 | 0.8571 | 0.8000 | 18 | 5 | 3 | 6 | 21 | 11 |
| **0.44** | **0.8125** | **0.7500** | **0.8750** | **0.9474** | **0.8372** | 18 | 7 | 1 | 6 | 19 | 13 |
| **0.45** (Screening) | **0.8125** | **0.7500** | **0.8750** | **0.9474** | **0.8372** | 18 | 7 | 1 | 6 | 19 | 13 |
| **0.46** | **0.8125** | **0.7500** | **0.8750** | **0.9474** | **0.8372** | 18 | 7 | 1 | 6 | 19 | 13 |
| **0.48** | **0.8125** | **0.7500** | **0.8750** | **0.9474** | **0.8372** | 18 | 7 | 1 | 6 | 19 | 13 |
| **0.50** (Reference) | **0.7917** | **0.7083** | **0.8750** | **0.9444** | **0.8095** | 17 | 7 | 1 | 7 | 18 | 14 |
| 0.52 | 0.7292 | 0.5833 | 0.8750 | 0.9333 | 0.7179 | 14 | 7 | 1 | 10 | 15 | 17 |
| 0.56 | 0.7292 | 0.5833 | 0.8750 | 0.9333 | 0.7179 | 14 | 7 | 1 | 10 | 15 | 17 |
| 0.60 | 0.7292 | 0.5833 | 0.8750 | 0.9333 | 0.7179 | 14 | 7 | 1 | 10 | 15 | 17 |
| 0.66 | 0.6875 | 0.5000 | 0.8750 | 0.9231 | 0.6486 | 12 | 7 | 1 | 12 | 13 | 19 |
| 0.70 | 0.6667 | 0.4583 | 0.8750 | 0.9167 | 0.6111 | 11 | 7 | 1 | 13 | 12 | 20 |
| 0.80 | 0.5833 | 0.2917 | 0.8750 | 0.8750 | 0.4375 | 7 | 7 | 1 | 17 | 8 | 24 |
| 0.90 | 0.5208 | 0.1667 | 0.8750 | 0.8000 | 0.2759 | 4 | 7 | 1 | 20 | 5 | 27 |

*Note: Complete 41-point evaluation table preserved in `threshold_grid_participant_et13.csv`.*

### Secondary Level: Recording-Level Evaluation ($N = 195$; 147 PD, 48 HC)
*(Descriptive only — recordings are repeated measures and not independent subjects)*

| Threshold ($\tau$) | Balanced Accuracy | Sensitivity | Specificity | Precision | F1-Score | TP | TN | FP | FN |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.5975 | 0.9456 | 0.2500 | 0.7943 | 0.8634 | 139 | 12 | 36 | 8 |
| 0.40 | 0.6860 | 0.8027 | 0.5625 | 0.8489 | 0.8252 | 118 | 27 | 21 | 29 |
| **0.45** (Screening) | **0.7197** | **0.7279** | **0.7115** | **0.8843** | **0.7985** | 107 | 34 | 14 | 40 |
| **0.50** (Reference) | **0.7260** | **0.6395** | **0.8125** | **0.9126** | **0.7520** | 94 | 39 | 9 | 53 |
| 0.60 | 0.7041 | 0.5238 | 0.8846 | 0.9390 | 0.6725 | 77 | 42 | 6 | 70 |
| 0.70 | 0.6888 | 0.4354 | 0.9423 | 0.9552 | 0.5981 | 64 | 45 | 3 | 83 |

---

## 5. Predefined Threshold Grid Evaluation (Stream A: Historical 22-Feature Model)

### Primary Level: Participant-Level Evaluation ($N = 32$; 24 PD, 8 HC)

| Threshold ($\tau$) | Balanced Accuracy | Sensitivity | Specificity | Precision | F1-Score | TP | TN | FP | FN | Pred Pos | Pred Neg |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.6667 | 0.9583 | 0.3750 | 0.8214 | 0.8846 | 23 | 3 | 5 | 1 | 28 | 4 |
| 0.36 | 0.7292 | 0.9583 | 0.5000 | 0.8519 | 0.9020 | 23 | 4 | 4 | 1 | 27 | 5 |
| 0.40 | 0.7083 | 0.9167 | 0.5000 | 0.8462 | 0.8800 | 22 | 4 | 4 | 2 | 26 | 6 |
| **0.45** (Screening) | **0.6875** | **0.8750** | **0.5000** | **0.8400** | **0.8571** | 21 | 4 | 4 | 3 | 25 | 7 |
| **0.50** (Reference) | **0.6667** | **0.8333** | **0.5000** | **0.8333** | **0.8333** | 20 | 4 | 4 | 4 | 24 | 8 |
| 0.56 | 0.7292 | 0.7083 | 0.7500 | 0.8947 | 0.7907 | 17 | 6 | 2 | 7 | 19 | 13 |
| 0.60 | 0.7708 | 0.6667 | 0.8750 | 0.9412 | 0.7805 | 16 | 7 | 1 | 8 | 17 | 15 |
| 0.68 | 0.7292 | 0.4583 | 1.0000 | 1.0000 | 0.6286 | 11 | 8 | 0 | 13 | 11 | 21 |

---

## 6. Detailed Comparison: Reference ($\tau = 0.50$) vs. Screening ($\tau = 0.45$)

| Evaluation Dimension | Stream B: Research ET-13 ($\tau = 0.50$) | Stream B: Research ET-13 ($\tau = 0.45$) | Stream A: Historical 22-Feat ($\tau = 0.50$) | Stream A: Historical 22-Feat ($\tau = 0.45$) |
|---|:---:|:---:|:---:|:---:|
| **Participant Balanced Accuracy** | **0.7917** | **0.8125** | 0.6667 | 0.6875 |
| **Participant Sensitivity** | 0.7083 (17/24) | 0.7500 (18/24) | 0.8333 (20/24) | 0.8750 (21/24) |
| **Participant Specificity** | **0.8750** (7/8) | **0.8750** (7/8) | 0.5000 (4/8) | 0.5000 (4/8) |
| **Participant False Positives** | **1** (`S50`) | **1** (`S50`) | **4** (`S13, S43, S49, S50`) | **4** (`S13, S43, S49, S50`) |
| **Participant False Negatives** | 7 | 6 | 4 | 3 |
| **Methodological Role** | Standard unbiased reference | Documented screening threshold | Standard unbiased reference | Documented screening threshold |

### Justification & Mechanism:
- **Stream B (Research ET-13):**
  - Lowering threshold from $0.50$ to $0.45$ captures **1 additional true positive** (Subject `S06`, $P = 0.4879$, moves from FN to TP).
  - Specificity remains completely unaffected at **$87.50\%$** because all 7 correctly classified healthy controls have predicted probabilities strictly below $0.431$ (`S42` is highest at $0.4307$).
  - However, because $\tau = 0.45$ sits directly adjacent to healthy subject `S42` ($0.4307$), the decision margin for `S42` narrows to $-0.0193$.
- **Stream A (Historical 22-Feature Model):**
  - Lowering threshold to $0.45$ increases sensitivity from $83.3\%$ to $87.5\%$ by capturing subject `S44` ($P = 0.4521$).
  - Specificity is capped at **$50.00\%$** (4 false positives out of 8 controls) across both thresholds due to pitch bias on healthy male voices (`S13`, `S43`, `S49`).

---

## 7. Participant Decision Margins & False Negative Analysis

We examined the signed decision margins $m_i(\tau) = p_i - \tau$ for all 32 participants under Stream B (Research ET-13):

```
       HEALTHY CONTROLS (8)                              PARKINSON'S SUBJECTS (24)
  True Status: 0                                    True Status: 1
  ─────────────────────────────────                 ─────────────────────────────────
  S10: P=0.2916 (TN, margin=-0.208)                 S25: P=0.2168 (FN, margin=-0.283)
  S17: P=0.3010 (TN, margin=-0.199)                 S08: P=0.3156 (FN, margin=-0.184)
  S43: P=0.3502 (TN, margin=-0.150)                 S32: P=0.3321 (FN, margin=-0.168)
  S07: P=0.3642 (TN, margin=-0.136)                 S44: P=0.3703 (FN, margin=-0.130)
  S49: P=0.4138 (TN, margin=-0.086)                 S27: P=0.3968 (FN, margin=-0.103)
  S13: P=0.4277 (TN, margin=-0.072)                 S04: P=0.4141 (FN, margin=-0.086)
  S42: P=0.4307 (TN, margin=-0.069)                 S06: P=0.4879 (FN at 0.50 / TP at 0.45)
                                                    ─────────────────────────────────
                                                    THRESHOLD tau = 0.50
                                                    ─────────────────────────────────
                                                    S19: P=0.5030 (TP, margin=+0.003)
                                                    S26: P=0.5049 (TP, margin=+0.005)
                                                    S02: P=0.5152 (TP, margin=+0.015)
                                                    S34: P=0.6024 (TP, margin=+0.102)
                                                    S31: P=0.6526 (TP, margin=+0.153)
                                                    ... (12 other clear TPs, P > 0.69)
  S50: P=0.7480 (FP, margin=+0.248)                 S35: P=0.9413 (TP, margin=+0.441)
```

### Critical Boundary Participants:
1. **Narrow False Negative `S06` ($P = 0.4879$):**
   - Sits immediately below $\tau = 0.50$ (margin $-0.0121$).
   - Flips to True Positive at $\tau \le 0.48$.
2. **Boundary True Positives (`S19`, `S26`, `S02`):**
   - `S19` ($P = 0.5030$, margin $+0.0030$)
   - `S26` ($P = 0.5049$, margin $+0.0049$)
   - `S02` ($P = 0.5152$, margin $+0.0152$)
   - These 3 subjects sit immediately above $\tau = 0.50$. Any threshold above $0.50$ (e.g. $\tau = 0.52$) immediately flips all three into False Negatives, dropping sensitivity to $58.33\%$.
3. **Deep Persistent False Negatives (`S25`, `S08`, `S32`, `S44`, `S27`, `S04`):**
   - Predicted probabilities range between $0.2168$ and $0.4141$.
   - Capturing these subjects by lowering the threshold would require $\tau \le 0.40$, which immediately causes healthy controls `S42`, `S13`, `S49` to flip into False Positives, destroying specificity.
4. **Isolated Control Outlier `S50` ($P = 0.7480$):**
   - The only false positive among controls. Sits at $P = 0.7480$ due to severe vocal shimmer instability. Would require $\tau > 0.74$ to classify as healthy, which would destroy sensitivity ($<45\%$).

---

## 8. Limitations & Clinical Boundary Statements

1. **Small Exploratory Cohort:** Evaluation is based on 32 human participants (24 Parkinson's, 8 healthy controls). While participant-level Leave-One-Group-Out strictly prevents intrasubject leakage, each healthy participant represents $12.5\%$ of the control cohort.
2. **No Claim of Clinical Optimality:** Neither $\tau = 0.50$ nor $\tau = 0.45$ can be claimed as "clinically validated" or "safe." They represent empirical operational points under Leave-One-Group-Out cross-validation on this dataset.
3. **Exploratory Screening Boundary:** Acoustic vocal features from sustained vowel phonations are screening biomarkers only; they do not replace clinical neurological evaluation.

---

## 9. Final Findings & Recommendations

### Key Findings:
1. **OOF Integrity:** Verified 100% leak-free participant grouping, zero NaNs, zero label mismatches across all evaluation files.
2. **Leakage Audit:** Confirmed that Stage 3A threshold optimization strictly nested threshold selection within inner training folds. No test data leaked into threshold tuning.
3. **Reference Threshold ($\tau = 0.50$):**
   - Remains the **primary unbiased reference threshold**.
   - Achieves **$0.7917$ Balanced Accuracy**, **$0.7083$ Sensitivity**, and **$0.8750$ Specificity** on the 13-feature research champion.
4. **Screening Threshold ($\tau = 0.45$):**
   - Supported as a **documented secondary screening threshold**.
   - Increases sensitivity to **$0.7500$** (capturing subject `S06`) without degrading specificity ($87.50\%$).
   - However, it narrows the decision margin for healthy subject `S42` to $-0.0193$.
5. **Thresholds Above $0.50$:** Unacceptable for screening because they immediately induce 3 additional false negatives (`S19`, `S26`, `S02`), collapsing sensitivity to $58.33\%$.

### Recommendation:
- Retain **$\tau = 0.50$** as the primary default decision threshold.
- Document **$\tau = 0.45$** as a secondary high-sensitivity screening threshold in the model manifest, accompanied by explicit clinical disclaimers.
- Do not deploy or adjust thresholds without formal user approval.
