# Phase 2.2C — Probability Calibration Comparison Report
## Parkinson's AI Voice Screening — Pathway A (UCI Parkinson's Dataset)

**Document Version:** 1.0.0  
**Date:** 2026-11-25  
**Evaluation Scope:** Formal Stage 1 Baseline (22 features) & Phase 2 Research Baseline (13 pitch-free features)  
**Evaluation Scheme:** Strictly subject-grouped nested Leave-One-Group-Out (LOGO, 32 folds) and 5-repeat 5-fold StratifiedGroupKFold (25 folds)  
**Artifact Integrity Check:** 100% SHA-256 match on all protected production and baseline artifacts before and after benchmarking.

---

## 1. Executive Summary

Phase 2.2C evaluated probability calibration for the Parkinson's AI voice screening system. Using a **nested, subject-grouped calibration architecture** that strictly prevents data leakage, we compared:
1. **Uncalibrated (raw model probabilities)**
2. **Platt / Sigmoid calibration** (parametric logistic calibration)
3. **Isotonic calibration** (exploratory non-parametric step calibration)
4. **Prevalence / Intercept adjustment** (logit offset correction for balanced class weighting)

### Core Finding: The "Calibration-Discrimination Paradox"
- Standard calibration metrics (Brier Score and Expected Calibration Error, ECE) showed superficial numerical "improvements" under Platt and Isotonic calibration (e.g., ECE dropped from 0.1778 to 0.0741 for the 13-feature research model).
- However, this "improvement" is mathematically deceptive: because the UCI dataset contains 75% Parkinson's patients (24 PD vs. 8 Healthy), post-hoc calibrators fit to the high sample prevalence by **shifting all probabilities upward**.
- Consequently, at the clinical reference threshold ($\tau = 0.50$):
  - **Specificity collapses catastrophically from 87.5% down to 0.0%** (every single healthy control participant is misclassified as Parkinson's!).
  - **Balanced Accuracy collapses from 0.7917 to chance level (0.4792 - 0.5000)**.
- Furthermore, true clinical prevalence of Parkinson's disease in older adults is **~1% to 2%**, not 75%. Calibrating probabilities to a 75% cohort sample prevalence is clinically invalid and completely destroys screening discrimination.

---

## 2. Experimental Architecture & Leak-Free Design

To prevent data leakage, calibration was integrated into a strict **nested subject-grouped cross-validation framework**:

```
Outer Cross-Validation Fold k (Test: Subject S_k, Untouched)
 │
 ├── Outer Training Data: 31 Participants
 │    │
 │    ├── Inner 5-Fold StratifiedGroupKFold (Groups: Subject IDs):
 │    │    ├── Train candidate on 4/5 of training subjects
 │    │    └── Predict on 1/5 held-out training subjects
 │    │    └── Aggregate Inner Out-of-Fold (OOF) Scores
 │    │
 │    ├── Fit Calibrators on Inner OOF Validation Scores:
 │    │    ├── Sigmoid / Platt: P(y=1|s) = 1 / (1 + exp(A*s + B)) with Bayesian target smoothing
 │    │    ├── Isotonic: Monotonic regression clipped to [0, 1]
 │    │    └── Prevalence Adjustment: logit shift by ln(N_neg / N_pos)
 │    │
 │    ├── Refit Estimator on All 31 Training Participants
 │    └── Compose Refitted Estimator with Fitted Calibrators
 │
 └── Predict on Held-Out Test Subject S_k (Completely unseen)
```

- **Subject Grouping:** Confirmed with assertion `train_subjects.isdisjoint(test_subjects)` at both outer and inner levels.
- **Participant-Level Aggregation:** Mean predicted probability per participant across all of their recordings:
  $$\bar{p}_{\text{sub}} = \frac{1}{K_i} \sum_{j=1}^{K_i} p_{ij}$$

---

## 3. Score-Distribution Mismatch Audit

A fundamental question in nested calibration is whether inner-fold models (trained on ~80% of outer training data, i.e., ~24-25 subjects) produce score distributions that match the outer estimator refitted on 100% of outer training data (31 subjects).

We audited this distribution shift using the two-sample Kolmogorov-Smirnov (KS) test across all folds:

| Model Configuration | Inner Scores (Mean $\pm$ Std) | Outer Refitted Scores (Mean $\pm$ Std) | Mean KS $p$-value | Min KS $p$-value | Mismatch Assessment |
|---|:---:|:---:|:---:|:---:|---|
| **Baseline LR 22 (LOGO)** | $2.041 \pm 4.095$ | $1.753 \pm 3.688$ | 0.4633 | **0.0315** | Measurable variance shrinkage; significant in outlier folds ($p < 0.05$) |
| **Baseline LR 22 (SGKF)** | $2.283 \pm 4.323$ | $1.924 \pm 3.918$ | 0.5880 | 0.0934 | Moderate score compression in refitted models |
| **Research ET 13 (LOGO)** | $0.432 \pm 1.203$ | $0.533 \pm 1.439$ | 0.5442 | 0.1546 | Refitted outer models exhibit wider variance ($+19.6\%$) due to deeper trees |
| **Research ET 13 (SGKF)** | $0.492 \pm 1.221$ | $0.567 \pm 1.539$ | 0.4914 | **0.0283** | Statistically significant score variance shift in multiple folds |

**Conclusion on Mismatch:** Score-distribution mismatch is **not harmless**. Because the outer refitted model is trained on more subjects, its trees split more aggressively or its logistic coefficients achieve higher confidence, causing outer test scores to have wider dispersion than the inner scores on which the calibrator was trained. This causes calibrators to over-predict extreme probabilities on outer test sets.

---

## 4. Comprehensive Benchmark Results

### A. Participant-Level LOGO Benchmark ($N = 32$ participants: 24 PD, 8 HC)

| Model | Calibration Method | Brier Score | ECE (Uniform 5) | ECE (Quantile 5) | ROC-AUC | Bal Acc ($\tau=0.50$) | Sensitivity | Specificity | F1-Score | TP | TN | FP | FN |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline LR (22 feat)** | **Uncalibrated** | 0.1881 | 0.1500 | 0.1938 | 0.7031 | **0.6667** | 0.8333 | **0.5000** | 0.8333 | 20 | 4 | 4 | 4 |
| | Sigmoid (Platt) | 0.1793 | 0.0443 | 0.1162 | 0.6615 | 0.4583 | 0.9167 | **0.0000** | 0.8148 | 22 | 0 | 8 | 2 |
| | Isotonic | 0.1568 | 0.0642 | 0.0996 | 0.7135 | 0.6458 | 0.9167 | 0.3750 | 0.8627 | 22 | 3 | 5 | 2 |
| | Prevalence-Adjusted | 0.1688 | 0.1445 | 0.1675 | 0.6667 | **0.7083** | 0.9167 | **0.5000** | 0.8800 | 22 | 4 | 4 | 2 |
| **Research ET (13 feat)** | **Uncalibrated** | 0.1863 | 0.1778 | 0.1778 | **0.7865** | **0.7917** | 0.7083 | **0.8750** | **0.8095** | 17 | 7 | 1 | 7 |
| | Sigmoid (Platt) | 0.1694 | 0.0741 | 0.0913 | 0.7188 | 0.4792 | 0.9583 | **0.0000** | 0.8364 | 23 | 0 | 8 | 1 |
| | Isotonic | 0.1665 | 0.0482 | 0.0742 | 0.7083 | 0.5000 | 1.0000 | **0.0000** | 0.8571 | 24 | 0 | 8 | 0 |

---

### B. Recording-Level LOGO Benchmark ($N = 195$ recordings: 147 PD, 48 HC)

| Model | Calibration Method | Brier Score | ECE (Uniform 5) | ECE (Quantile 5) | ROC-AUC | Bal Acc ($\tau=0.50$) | Sensitivity | Specificity | F1-Score | TP | TN | FP | FN |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline LR (22 feat)** | **Uncalibrated** | 0.2174 | 0.2120 | 0.2105 | 0.7238 | **0.6310** | 0.7619 | **0.5000** | 0.7915 | 112 | 24 | 24 | 35 |
| | Sigmoid (Platt) | 0.1825 | 0.0732 | 0.0811 | 0.6671 | 0.4800 | 0.9184 | 0.0417 | 0.8232 | 135 | 2 | 46 | 12 |
| | Isotonic | 0.1665 | 0.0855 | 0.0967 | 0.6881 | 0.5808 | 0.9116 | 0.2500 | 0.8454 | 134 | 12 | 36 | 13 |
| | Prevalence-Adjusted | 0.1873 | 0.1608 | 0.1760 | 0.7072 | **0.6818** | 0.8844 | **0.4792** | 0.8609 | 130 | 23 | 25 | 17 |
| **Research ET (13 feat)** | **Uncalibrated** | 0.1941 | 0.1793 | 0.1793 | **0.7483** | **0.7260** | 0.6395 | **0.8125** | 0.7520 | 94 | 39 | 9 | 53 |
| | Sigmoid (Platt) | 0.1715 | 0.0718 | 0.0800 | 0.6942 | 0.4762 | 0.9524 | **0.0000** | 0.8358 | 140 | 0 | 48 | 7 |
| | Isotonic | 0.1727 | 0.0346 | 0.1030 | 0.6750 | 0.4388 | 0.8776 | **0.0000** | 0.7963 | 129 | 0 | 48 | 18 |

---

### C. Repeated StratifiedGroupKFold Benchmark (5 repeats $\times$ 5 folds = 25 evaluations; Mean $\pm$ Std)

| Model Configuration | Calibration Method | Brier Score | ECE (Uniform 5) | ROC-AUC | Balanced Accuracy | Sensitivity | Specificity |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline LR (22 feat)** | **Uncalibrated** | $0.1877 \pm 0.0294$ | $0.1903 \pm 0.0604$ | $0.7219 \pm 0.0589$ | $0.6708 \pm 0.0228$ | $0.8417 \pm 0.0457$ | $0.5000 \pm 0.0000$ |
| | Sigmoid (Platt) | $0.1625 \pm 0.0104$ | $0.0788 \pm 0.0457$ | $0.7094 \pm 0.0578$ | $0.5917 \pm 0.0349$ | $0.9583 \pm 0.0417$ | $0.2250 \pm 0.1046$ |
| | Isotonic | $0.1595 \pm 0.0192$ | $0.1375 \pm 0.0566$ | $0.7083 \pm 0.0578$ | $0.6375 \pm 0.0480$ | $0.9500 \pm 0.0543$ | $0.3250 \pm 0.1118$ |
| | Prevalence-Adjusted | $0.1625 \pm 0.0240$ | $0.1656 \pm 0.0433$ | $0.7052 \pm 0.0528$ | $0.7083 \pm 0.0255$ | $0.9167 \pm 0.0510$ | $0.5000 \pm 0.0000$ |
| **Research ET (13 feat)** | **Uncalibrated** | $0.1850 \pm 0.0038$ | $0.1793 \pm 0.0026$ | $\mathbf{0.7990 \pm 0.0155}$ | $\mathbf{0.7792 \pm 0.0114}$ | $0.6833 \pm 0.0228$ | $\mathbf{0.8750 \pm 0.0000}$ |
| | Sigmoid (Platt) | $0.1589 \pm 0.0065$ | $0.0555 \pm 0.0321$ | $0.7729 \pm 0.0358$ | $0.5000 \pm 0.0295$ | $0.9500 \pm 0.0186$ | $0.0500 \pm 0.0685$ |
| | Isotonic | $0.1628 \pm 0.0060$ | $0.0689 \pm 0.0242$ | $0.7625 \pm 0.0395$ | $0.5208 \pm 0.0295$ | $0.9167 \pm 0.0589$ | $0.1250 \pm 0.0884$ |

---

## 5. Critical Diagnostic Subgroup Tracking

Tracking individual subject probability shifts under LOGO:

| Subject ID | True Class | Clinical Profile | Uncalibrated $P(\text{PD})$ | Sigmoid $P(\text{PD})$ | Isotonic $P(\text{PD})$ | Pre-Adjusted $P(\text{PD})$ | Impact of Calibration |
|:---:|:---:|---|:---:|:---:|:---:|:---:|---|
| **S13** | HC (0) | Healthy Male ($F_0 \approx 116$ Hz) | **0.428 (TN)** | **0.703 (FP)** | **0.742 (FP)** | N/A | Flipped to False Positive |
| **S43** | HC (0) | Healthy Male ($F_0 \approx 116$ Hz) | **0.350 (TN)** | **0.649 (FP)** | **0.558 (FP)** | N/A | Flipped to False Positive |
| **S49** | HC (0) | Healthy Male ($F_0 \approx 117$ Hz) | **0.414 (TN)** | **0.666 (FP)** | **0.630 (FP)** | N/A | Flipped to False Positive |
| **S50** | HC (0) | Healthy Female (Acoustic outlier) | 0.748 (FP) | 0.915 (FP) | 0.939 (FP) | N/A | Severe probability inflation |
| **S08** | PD (1) | Mild / Atypical PD | 0.316 (FN) | 0.538 (TP) | 0.529 (TP) | N/A | Flipped to TP solely via upward shift |
| **S26** | PD (1) | Borderline PD | 0.505 (TP) | 0.732 (TP) | 0.728 (TP) | N/A | Shifted higher |
| **S34** | PD (1) | Moderate PD | 0.602 (TP) | 0.777 (TP) | 0.786 (TP) | N/A | Shifted higher |
| **S35** | PD (1) | Advanced Dysphonic PD | 0.941 (TP) | 0.982 (TP) | 1.000 (TP) | N/A | Saturated to 1.000 |

### Clinical Interpretation:
- Under the **Uncalibrated 13-feature ExtraTrees model**, the low-pitch healthy males (`S13`, `S43`, `S49`) were cleanly classified as Healthy ($P \in [0.350, 0.428]$).
- When Sigmoid or Isotonic calibration was applied, **all three healthy male subjects flipped to False Positives** ($P \in [0.558, 0.742]$).
- The apparent conversion of `S08` from false negative to true positive is an illusion of global threshold elevation: the calibrator simply shifted the entire distribution upward, catching `S08` at the expense of misclassifying every single healthy person.

---

## 6. Evaluation of Specific Calibration Methods

### A. Platt / Sigmoid Calibration
- **Theoretical promise:** Fits a monotonic logistic mapping with low parameter complexity (2 parameters, $A, B$).
- **Observed empirical failure:** Learns the 75% positive class balance from inner folds. The intercept $B$ forces the output probability to center near 0.75, driving false positives to 100% on healthy controls at $\tau = 0.50$.
- **ROC-AUC Degradation:** Because logistic scaling was applied to decision scores whose variance differed between inner and outer folds (score mismatch), ranking ties and small non-linear shifts reduced participant ROC-AUC from **0.7865 down to 0.7188**.

### B. Isotonic Calibration
- **Theoretical risk:** Non-parametric step function prone to severe overfitting on small cohorts.
- **Observed empirical failure:** With only 8 healthy controls (and only ~6 in each inner training fold), isotonic regression produced severe piecewise plateau artifacts. Probabilities saturated to 0.0 or 1.0. ROC-AUC dropped to **0.7083** and Specificity dropped to **0.0000**.
- **Verdict:** Unconditionally rejected for small sample cohorts.

### C. Prevalence / Intercept Adjustment
- **Theoretical formulation:** Corrects logistic regression's decision offset caused by `class_weight='balanced'` by subtracting $\ln(N_{\text{HC}} / N_{\text{PD}})$.
- **Empirical result:** Maintained specificity at 0.5000 and balanced accuracy at 0.7083 on the 22-feature baseline.
- **Critical Caveat:** This adjustment aligns probabilities to the **75% UCI sample prevalence**, which is not the true general population prevalence (~1–2%). Documented as an exploratory mathematical adjustment only; it must not be interpreted as true patient risk.

---

## 7. Artifacts Generated and Verified

All files were produced in `ml/pathway_a/experiments/results/phase2_calibration/`:
1. `calibration_metrics_summary.csv`: Full recording and participant metrics across all methods.
2. `participant_calibrated_predictions.csv`: Out-of-fold subject predictions for all 32 participants.
3. `oof_calibrated_predictions_recordings.csv`: Out-of-fold recording predictions for all 195 samples.
4. `fold_level_metrics.csv`: Detailed metrics across all 32 LOGO folds and 25 SGKF folds.
5. `score_mismatch_audit.json`: Kolmogorov-Smirnov score mismatch statistics.
6. `reliability_diagram_baseline_lr_22.png`: 4-panel reliability diagram for the 22-feature baseline.
7. `reliability_diagram_research_et_13.png`: 4-panel reliability diagram for the 13-feature pitch-free model.

### Protected Artifact Hashes (Verified 100% Identical):
- `champion_pipeline.joblib`: `b2d052eed9c2515506a244b401dfb78e26acd8d24fa0e8ed68dfadc5aae7a414` (Matches)
- `baseline_pipeline.joblib`: `81dc11f063a7bd52b404f36621d2a37bb33655456f209df1401eefd56b6b884f` (Matches)
- `audio_pipeline.joblib`: `7024bc988db633bddea41970a0985d5cb4a779add54e3b3d08f54af52ff5e01f` (Matches)

---

## 8. Final Decision & Recommendation

### Formal Decision Category:
> **"Reject calibration because the results are unstable or unsupported"**

### Scientific Justification:
1. **Destruction of Screening Specificity:** Post-hoc calibration (both Sigmoid and Isotonic) shifts probabilities upward to mirror the 75% dataset prevalence, completely destroying specificity (collapsing from **87.5% down to 0.0%** at reference threshold 0.50).
2. **Artificial Metric Improvement:** The reduction in ECE (0.1778 $\to$ 0.0741) is a statistical artifact of matching an unrepresentative cohort prior; it does not reflect improved diagnostic utility.
3. **Clinical Discordance:** Real-world Parkinson's prevalence is ~1–2%. Calibrating model outputs to a 75% case-control ratio provides fabricated probability figures that would mislead clinicians.
4. **Superiority of Raw Ensemble Probabilities:** The uncalibrated, balanced-weight ExtraTrees ensemble natively produces symmetric, rank-preserving probabilities that achieve **0.7917 Balanced Accuracy**, **0.7865 ROC-AUC**, and **87.5% Specificity** across all 32 participants under strict Leave-One-Group-Out validation.
5. **Action:** Retain raw ensemble probabilities from the 13-feature pitch-free model. Do not deploy post-hoc calibrators.
