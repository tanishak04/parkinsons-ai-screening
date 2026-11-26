# Stage 2A: Preprocessing Experiments Report

**Module:** `ml/pathway_a/experiments/experiment_preprocessing.py`  
**Execution Date:** 2026-11-25  
**Dataset:** UCI Parkinson's Dataset (`parkinsons.data`, 195 recordings, 32 participants)  
**Status:** Complete — Experimental Only (No Production Promotion)

---

## 1. Executive Summary

Stage 2A evaluated whether replacing the baseline's `StandardScaler` with outlier-resistant (`RobustScaler`) or variance-stabilizing (`PowerTransformer` Yeo-Johnson) preprocessing, combined with alternative regularization penalties (`L1`, `ElasticNet`), improves generalization performance on the 22 vocal biomarkers.

All 9 configurations were benchmarked using the verified isolated framework across:
- **Repeated 4-Fold StratifiedGroupKFold (10 seeds = 40 folds per configuration)**
- **32-Fold Leave-One-Group-Out (LOGO) cross-validation**
- **Dual-level aggregation:** Recording-level ($N=195$) and Participant-level ($N=32$)

### Key Takeaway
**No decisive winner emerged.** Differences between `StandardScaler`, `RobustScaler`, and `PowerTransformer` under Logistic Regression are modest and well within cross-validation variance ($\sigma_{\text{BalAcc}} \approx 0.10$–$0.13$). Crucially, **all 9 linear configurations stalled at 50.0% specificity at the participant level** (consistently misclassifying 4 of the 8 healthy control subjects). Preprocessing transformations alone cannot overcome the fundamental linear separability limits on this dataset.

---

## 2. Tested Configurations

All pipelines preserved the frozen 22-feature input contract and were fitted **strictly within training folds**:

| Config ID | Preprocessor | Penalty / Solver | Regularization Hyperparameters |
| :--- | :--- | :--- | :--- |
| `std_l2_reference` | `StandardScaler()` | L2 / `lbfgs` | $C=1.0$, `class_weight='balanced'`, `max_iter=2000` *(Baseline Reference)* |
| `std_l1` | `StandardScaler()` | L1 / `saga` | $C=1.0$, `class_weight='balanced'`, `max_iter=5000` |
| `std_elasticnet` | `StandardScaler()` | ElasticNet / `saga` | $C=1.0$, $\text{l1\_ratio}=0.5$, `class_weight='balanced'`, `max_iter=5000` |
| `robust_l2` | `RobustScaler()` | L2 / `lbfgs` | $C=1.0$, `class_weight='balanced'`, `max_iter=2000` |
| `robust_l1` | `RobustScaler()` | L1 / `saga` | $C=1.0$, `class_weight='balanced'`, `max_iter=5000` |
| `robust_elasticnet` | `RobustScaler()` | ElasticNet / `saga` | $C=1.0$, $\text{l1\_ratio}=0.5$, `class_weight='balanced'`, `max_iter=5000` |
| `power_l2` | `PowerTransformer(yeo-johnson)` | L2 / `lbfgs` | $C=1.0$, `class_weight='balanced'`, `max_iter=2000` |
| `power_l1` | `PowerTransformer(yeo-johnson)` | L1 / `saga` | $C=1.0$, `class_weight='balanced'`, `max_iter=5000` |
| `power_elasticnet` | `PowerTransformer(yeo-johnson)` | ElasticNet / `saga` | $C=1.0$, $\text{l1\_ratio}=0.5$, `class_weight='balanced'`, `max_iter=5000` |

---

## 3. Results Matrix

### A. Recording-Level Performance (Repeated 4×10 SGKF, 10-Seed Out-Of-Fold Mean)

| Config ID | Balanced Acc | Recall (Sens) | Specificity | ROC-AUC | PR-AUC | F1 | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`std_l2_reference`** | 0.6399 | 0.7653 | 0.5146 | **0.7365** | **0.8982** | 0.7955 | 0.2164 |
| `std_l1` | **0.6556** | 0.7653 | **0.5458** | 0.7342 | 0.8921 | 0.7997 | 0.2143 |
| `std_elasticnet` | 0.6507 | 0.7680 | 0.5333 | 0.7367 | 0.8964 | 0.7997 | 0.2142 |
| `robust_l2` | 0.6413 | 0.7639 | 0.5188 | 0.7277 | 0.8882 | 0.7951 | 0.2133 |
| `robust_l1` | 0.6480 | 0.7626 | 0.5333 | 0.7175 | 0.8738 | 0.7964 | 0.2153 |
| `robust_elasticnet` | 0.6455 | 0.7660 | 0.5250 | 0.7247 | 0.8832 | 0.7973 | **0.2128** |
| `power_l2` | 0.6446 | 0.7850 | 0.5042 | 0.7157 | 0.8838 | 0.8062 | 0.2135 |
| `power_l1` | 0.6480 | 0.7898 | 0.5062 | 0.6990 | 0.8684 | 0.8093 | 0.2175 |
| `power_elasticnet` | 0.6480 | 0.7878 | 0.5083 | 0.7132 | 0.8811 | 0.8084 | 0.2136 |

---

### B. Participant-Level Performance (Primary Clinical Unit)

Probabilities aggregated across each participant's recordings ($\bar{P}_{\text{PD}} \ge 0.50$):

#### 1. Repeated 4×10 SGKF (Mean across 10 Seeds, $N=32$ subjects)
| Config ID | Balanced Acc | Recall (Sens) | Specificity | ROC-AUC | PR-AUC | F1 | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`std_l2_reference`** | 0.6562 | 0.8125 | 0.5000 | 0.7260 | **0.8916** | 0.8207 | 0.1865 |
| `std_l1` | 0.6542 | 0.8083 | 0.5000 | **0.7286** | 0.8877 | 0.8181 | **0.1835** |
| `std_elasticnet` | 0.6583 | 0.8167 | 0.5000 | 0.7255 | 0.8885 | 0.8232 | 0.1843 |
| `robust_l2` | 0.6583 | 0.8167 | 0.5000 | 0.7198 | 0.8868 | 0.8231 | 0.1864 |
| `robust_l1` | 0.6521 | 0.8042 | 0.5000 | 0.7141 | 0.8797 | 0.8155 | 0.1871 |
| `robust_elasticnet` | 0.6542 | 0.8083 | 0.5000 | 0.7214 | 0.8866 | 0.8180 | 0.1858 |
| `power_l2` | 0.6729 | 0.8458 | 0.5000 | 0.7135 | 0.8830 | 0.8404 | 0.1858 |
| `power_l1` | **0.6750** | **0.8500** | 0.5000 | 0.7016 | 0.8765 | **0.8425** | 0.1895 |
| `power_elasticnet` | **0.6750** | **0.8500** | 0.5000 | 0.7172 | 0.8850 | **0.8427** | 0.1861 |

#### 2. 32-Fold Leave-One-Group-Out (LOGO Pooled OOF, $N=32$ subjects)
| Config ID | Balanced Acc | Recall (Sens) | Specificity | ROC-AUC | PR-AUC | F1 | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`std_l2_reference`** | 0.6667 | 0.8333 | 0.5000 | 0.7031 | 0.8803 | 0.8333 | 0.1881 |
| `std_l1` | 0.6667 | 0.8333 | 0.5000 | **0.7188** | 0.8839 | 0.8333 | **0.1855** |
| `std_elasticnet` | 0.6667 | 0.8333 | 0.5000 | **0.7188** | **0.8847** | 0.8333 | 0.1877 |
| `robust_l2` | 0.6667 | 0.8333 | 0.5000 | 0.7031 | 0.8736 | 0.8333 | 0.1907 |
| `robust_l1` | 0.6667 | 0.8333 | 0.5000 | 0.6927 | 0.8586 | 0.8333 | 0.1920 |
| `robust_elasticnet` | 0.6667 | 0.8333 | 0.5000 | 0.7031 | 0.8724 | 0.8333 | 0.1913 |
| `power_l2` | **0.6875** | **0.8750** | 0.5000 | 0.6615 | 0.8525 | **0.8571** | 0.1912 |
| `power_l1` | **0.6875** | **0.8750** | 0.5000 | 0.6510 | 0.8400 | **0.8571** | 0.2063 |
| `power_elasticnet` | **0.6875** | **0.8750** | 0.5000 | 0.6562 | 0.8493 | **0.8571** | 0.1937 |

---

## 4. Detailed Analysis

### 1. RobustScaler vs. StandardScaler
`RobustScaler` (median / IQR scaling) performed virtually identically to `StandardScaler` across all metrics.
- In recording SGKF, Balanced Accuracy changed by less than $0.007$ (`0.6413` vs `0.6399`).
- In participant-level LOGO, ROC-AUC was slightly lower (`0.6927`–`0.7031` vs `0.7031`–`0.7188`).
- **Conclusion:** While voice pitch and shimmer have high variance, `RobustScaler` provides no empirical advantage over standard z-score normalization for linear decision boundaries on this dataset.

### 2. PowerTransformer (Yeo-Johnson) Trade-off
`PowerTransformer` yielded an apparent increase in participant-level Recall (`0.8750` vs `0.8333` in LOGO, $+4.2\%$) and Balanced Accuracy (`0.6875` vs `0.6667`).
- **However, ranking performance degraded significantly:** Participant LOGO ROC-AUC dropped from `0.7031` to `0.6615` (L2), `0.6510` (L1), and `0.6562` (ElasticNet).
- **Mechanism:** Yeo-Johnson shifts skewed feature tails toward symmetry, compressing distances in the positive feature space. Under `class_weight='balanced'`, this pushes more samples above the default threshold $\tau=0.50$, artificially boosting sensitivity while flattening probability discrimination across positive samples.

### 3. Regularization Comparison (L1 vs. L2 vs. ElasticNet)
- **`std_l1` and `std_elasticnet`** demonstrated small, consistent improvements in recording-level Specificity (`0.5458` and `0.5333` vs `0.5146`) and LOGO Participant ROC-AUC (`0.7188` vs `0.7031`), alongside lower Brier scores (`0.1835`–`0.1855` vs `0.1881`).
- L1 sparsity partially neutralizes the 19 pairs of collinear features ($r \ge 0.95$), stabilizing coefficient weights without hurting recall.
- However, the delta in Balanced Accuracy ($+0.0157$ in recording SGKF) is well within the 40-fold standard deviation ($\sigma \approx 0.11$).

### 4. The 50.0% Specificity Bottleneck
Across all 9 configurations under both SGKF and LOGO, participant-level Specificity remained invariant at exactly `0.5000` (4 true negatives, 4 false positives out of 8 healthy participants).
- Regardless of scaling or L1/L2 penalties, linear hyperplanes fail to cleanly separate the remaining 4 healthy subjects from early-stage PD phonations without shifting the decision threshold or introducing non-linear feature combinations.

---

## 5. Decision & Next Steps

1. **No Winner Declared:** In accordance with experiment constraints, no configuration is declared a definitive winner because score differences between `StandardScaler` and `RobustScaler`/`PowerTransformer` are within noise and do not solve the specificity bottleneck.
2. **Preprocessing Selection for Stage 2B:**
   - Retain `StandardScaler` as the primary preprocessor due to its superior ROC-AUC preservation (`0.7365` recording, `0.7188` subject).
   - Keep `PowerTransformer` as an alternate pipeline for non-linear models that may benefit from Gaussianized input without suffering linear ranking collapse.
3. **Transition to Stage 2B (Model Families):**
   - Proceed to evaluate non-linear candidate models (**SVM RBF**, **Random Forest**, **Extra Trees**, **Gradient Boosting**) under the same 40-fold SGKF and 32-fold LOGO harness.
