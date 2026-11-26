# Stage 2B: Nonlinear Model Family Evaluation Report

**Module:** `ml/pathway_a/experiments/model_comparison.py`  
**Execution Date:** 2026-11-25  
**Dataset:** UCI Parkinson's Dataset (`parkinsons.data`, 195 recordings, 32 participants)  
**Status:** Complete — Experimental Benchmark (No Production Model Promotion)

---

## 1. Executive Summary

Stage 2B evaluated whether nonlinear machine learning algorithms (Kernel SVM, Bagged Ensembles, and Gradient Boosters) improve generalization, participant-level discrimination, and calibration over the linear Logistic Regression baseline.

Evaluations were performed across:
- **Repeated 4-Fold StratifiedGroupKFold (10 random seeds = 40 folds per model)**
- **32-Fold Leave-One-Group-Out (LOGO) cross-validation**
- **Dual Granularity:** Recording-level ($N=195$) and Participant-level ($N=32$)
- **Zero Speaker Leakage:** Strict disjointness verification enforced on every fold.

### Core Scientific Findings
1. **Tree Ensembles Substantially Outperform Linear & Kernel Baselines in Discrimination:**  
   `ExtraTreesClassifier` achieved the highest ranking discrimination of all models: **ROC-AUC = 0.8113** (recording SGKF) and **0.8073** (participant LOGO), improving upon Logistic Regression by **+10.4% ROC-AUC** in subject-level LOGO (`0.8073` vs `0.7031`).
2. **Extreme Trees Overcomes the Specificity Collapse:**  
   While standard gradient boosters collapsed to sub-40% specificity, `ExtraTrees` achieved the highest recording-level specificity (**0.5938** vs `0.5146` baseline) while maintaining balanced sensitivity (**0.8125**–**0.8333** subject level).
3. **Calibrated Probability Quality (Brier Score):**  
   Both `ExtraTrees` (Brier: `0.1559` subject LOGO) and `XGBoost` (Brier: `0.1398` subject LOGO) substantially improved probability reliability over Logistic Regression (`0.1881` subject LOGO, `0.2164` recording).
4. **Kernel SVM (RBF) Underperformed:**  
   `SVM (RBF)` suffered from the curse of dimensionality across the 22 collinear voice measures, with participant LOGO ROC-AUC dropping to **0.5833** and specificity collapsing to **0.3750**.
5. **Hardware Acceleration Reality:**  
   NVIDIA RTX 5050 Laptop GPU (8GB VRAM) was detected and successfully ran `XGBoost (CUDA)`. However, on $N=195$ samples, CPU execution was **2.7× faster** (`3.94s` CPU vs `10.58s` CUDA) due to PCIe transfer overhead.

---

## 2. Phase 0: Stage 2A Output Reconciliation

### The 1,008 Rows Reconciliation in `per_fold_metrics.csv`
During Phase 0 inspection, the row count of `per_fold_metrics.csv` from Stage 2A was mathematically verified against the evaluation architecture:

- **Stage 2A Configurations:** 9 distinct configurations (3 scalers $\times$ 3 regularizers).
- **Repeated 4-Fold SGKF (10 seeds = 40 folds per config):**
  - Evaluated at both `recording` level (40 rows) and `participant` level (40 rows) = **80 rows per config**.
- **32-Fold Leave-One-Group-Out (LOGO):**
  - 32 folds per config holding out 1 participant each = **32 rows per config**.
- **Total Rows per Configuration:** $80 + 32 = 112$ rows.
- **Total CSV Rows:** $112 \times 9 \text{ configs} = \mathbf{1,008 \text{ rows}}$.

### OOF Predictions Reconciliation
- 10 seeds $\times 195$ recordings = 1,950 SGKF rows per config.
- 1 repeat $\times 195$ recordings = 195 LOGO rows per config.
- Total per config = $2,145$ rows; across 9 configs = $\mathbf{19,305 \text{ rows}}$ in `pooled_oof_predictions.csv`.
- **Verdict:** Zero missing data, zero duplicate rows. Stage 2A outputs are completely reconciled and mathematically verified.

---

## 3. Hardware & Compute Environment (Phase 5)

| Component | Status / Specification |
| :--- | :--- |
| **CUDA Driver & GPU** | Detected: `NVIDIA GeForce RTX 5050 Laptop GPU` |
| **VRAM Available** | 8,151 MB (8.0 GB) |
| **Driver Version** | 621.01 |
| **Scikit-Learn Models** | Executed on **CPU** (`n_jobs=-1`) — standard scikit-learn algorithms are CPU-based. |
| **XGBoost Execution** | Both **CPU** (`xgboost_cpu`, 3.94s) and **CUDA** (`xgboost_cuda`, 10.58s) were evaluated under identical seeds. |
| **Acceleration Assessment** | For small tabular datasets ($N < 10,000$), CPU execution is substantially more efficient than GPU kernel offloading. |

---

## 4. Evaluated Model Configurations

| Model ID | Family | Preprocessing | Key Hyperparameters | Device |
| :--- | :--- | :--- | :--- | :--- |
| `lr_baseline_reference` | Logistic Regression | `StandardScaler()` | $C=1.0$, `class_weight='balanced'`, `lbfgs`, `max_iter=2000` | CPU |
| `svm_rbf` | Support Vector Machine | `StandardScaler()` | `CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, balanced))` | CPU |
| `random_forest` | Random Forest | None (Tree invariant) | `n_estimators=100`, `max_depth=4`, `min_samples_leaf=3`, `balanced` | CPU |
| `extra_trees` | Extra Trees | None (Tree invariant) | `n_estimators=100`, `max_depth=4`, `min_samples_leaf=3`, `balanced` | CPU |
| `gradient_boosting` | Gradient Boosting | None (Tree invariant) | `n_estimators=100`, `max_depth=3`, `lr=0.05`, `subsample=0.8` | CPU |
| `hist_gradient_boosting`| HistGradientBoosting | None (Tree invariant) | `max_iter=100`, `max_depth=3`, `lr=0.05`, `balanced` | CPU |
| `xgboost_cpu` | XGBoost | None (Tree invariant) | `n_est=100`, `depth=3`, `lr=0.05`, `scale_pos_weight=0.326`, `tree_method='hist'` | CPU |
| `xgboost_cuda` | XGBoost | None (Tree invariant) | `n_est=100`, `depth=3`, `lr=0.05`, `scale_pos_weight=0.326`, `device='cuda'` | GPU |

---

## 5. Comparative Results

### A. Recording-Level Performance (Repeated 4×10 SGKF, 10-Seed OOF Mean, $N=195$)

| Model ID | Device | Balanced Acc | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lr_baseline_reference` | CPU | 0.6399 | 0.7653 | 0.5146 | 0.8284 | 0.7955 | 0.7365 | 0.8982 | 0.2164 |
| `svm_rbf` | CPU | 0.6023 | 0.8837 | 0.3208 | 0.7995 | 0.8391 | 0.6706 | 0.8472 | 0.1827 |
| `random_forest` | CPU | 0.6327 | 0.8259 | 0.4396 | 0.8203 | 0.8222 | 0.7804 | 0.9228 | 0.1675 |
| **`extra_trees`** | CPU | **0.6758** | 0.7578 | **0.5938** | **0.8529** | 0.8018 | **0.8113** | **0.9349** | **0.1603** |
| `gradient_boosting` | CPU | 0.6588 | **0.9197** | 0.3979 | 0.8242 | **0.8691** | 0.7657 | 0.8939 | 0.1739 |
| `hist_gradient_boosting`| CPU | 0.6472 | 0.8735 | 0.4208 | 0.8222 | 0.8466 | 0.7440 | 0.8718 | 0.1834 |
| `xgboost_cpu` | CPU | **0.6767** | 0.8701 | 0.4833 | 0.8384 | 0.8534 | 0.7880 | 0.9128 | 0.1618 |
| `xgboost_cuda` | GPU | 0.6743 | 0.8714 | 0.4771 | 0.8373 | 0.8534 | 0.7835 | 0.9080 | 0.1630 |

---

### B. Participant-Level Performance (Primary Clinical Unit)

Probabilities aggregated per participant ($\bar{P}_{\text{PD}} \ge 0.50$):

#### 1. Repeated 4×10 SGKF (Mean across 10 Seeds, $N=32$ subjects)
| Model ID | Device | Balanced Acc | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lr_baseline_reference` | CPU | 0.6562 | 0.8125 | 0.5000 | 0.8300 | 0.8207 | 0.7260 | 0.8916 | 0.1865 |
| `svm_rbf` | CPU | 0.6312 | 0.9500 | 0.3125 | 0.8066 | 0.8716 | 0.6682 | 0.8446 | 0.1720 |
| `random_forest` | CPU | 0.6250 | 0.8625 | 0.3875 | 0.8089 | 0.8343 | 0.7943 | 0.9308 | 0.1513 |
| **`extra_trees`** | CPU | 0.6750 | 0.8125 | **0.5375** | **0.8419** | 0.8262 | **0.8385** | **0.9451** | 0.1529 |
| `gradient_boosting` | CPU | **0.6813** | **0.9625** | 0.4000 | 0.8286 | **0.8900** | 0.7750 | 0.9115 | 0.1501 |
| `hist_gradient_boosting`| CPU | 0.6604 | 0.9208 | 0.4000 | 0.8229 | 0.8679 | 0.7536 | 0.8953 | 0.1543 |
| `xgboost_cpu` | CPU | 0.6792 | 0.9208 | 0.4375 | 0.8344 | 0.8733 | 0.8089 | 0.9273 | **0.1382** |
| `xgboost_cuda` | GPU | 0.6812 | 0.9125 | 0.4500 | 0.8368 | 0.8705 | 0.8047 | 0.9240 | 0.1392 |

#### 2. 32-Fold Leave-One-Group-Out (LOGO Pooled OOF, $N=32$ subjects)
| Model ID | Device | Balanced Acc | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lr_baseline_reference` | CPU | 0.6667 | 0.8333 | **0.5000** | 0.8333 | 0.8333 | 0.7031 | 0.8803 | 0.1881 |
| `svm_rbf` | CPU | 0.6667 | 0.9583 | 0.3750 | 0.8214 | 0.8846 | 0.5833 | 0.7999 | 0.1832 |
| `random_forest` | CPU | 0.6250 | 0.8750 | 0.3750 | 0.8077 | 0.8400 | 0.7396 | 0.9098 | 0.1554 |
| **`extra_trees`** | CPU | 0.6667 | 0.8333 | **0.5000** | **0.8333** | 0.8333 | **0.8073** | **0.9323** | 0.1559 |
| `gradient_boosting` | CPU | **0.6875** | **1.0000** | 0.3750 | 0.8276 | **0.9057** | 0.7188 | 0.8945 | 0.1532 |
| `hist_gradient_boosting`| CPU | 0.6458 | 0.9167 | 0.3750 | 0.8148 | 0.8627 | 0.7135 | 0.8707 | 0.1554 |
| `xgboost_cpu` | CPU | 0.6458 | 0.9167 | 0.3750 | 0.8148 | 0.8627 | 0.7969 | 0.9265 | **0.1398** |
| `xgboost_cuda` | GPU | 0.6667 | 0.9583 | 0.3750 | 0.8214 | 0.8846 | 0.7969 | 0.9258 | 0.1422 |

---

## 6. Deep Technical Analysis

### 1. Why ExtraTrees Outperformed All Other Ensembles
Random Forest and Gradient Boosters greedily select optimal feature splits. Because the UCI dataset contains 19 pairs of features with $r \ge 0.95$, standard decision trees repeatedly split on collinear jitter and shimmer variants, over-indexing on correlated measurement noise and degrading test specificity.  
In contrast, `ExtraTreesClassifier` draws random thresholds for each candidate feature. This acts as an implicit regularizer that de-correlates individual trees, yielding the smoothest probability estimates and achieving:
- **Participant ROC-AUC of 0.8073–0.8385** (highest of all models).
- **Specificity of 0.5938** (recording SGKF), the only model to break past 55% specificity without threshold tuning.

### 2. The Gradient Boosting Sensitivity vs. Specificity Trade-off
Both `GradientBoosting` (100% recall in LOGO) and `XGBoost` (91.7%–95.8% recall in LOGO) demonstrated extraordinary sensitivity to Parkinsonian vocal traits. However, under the default decision threshold ($\tau=0.50$), this came at the expense of specificity (dropping to 37.5% at the participant level, misclassifying 5 of 8 healthy subjects).  
Because `XGBoost` has a very low Brier score (`0.1382`–`0.1398`) and high ROC-AUC (`0.7969`–`0.8089`), its thresholded specificity drop is an operating point issue rather than a ranking issue; threshold tuning in Stage 2D will be able to restore balanced specificity.

### 3. Failure of SVM with RBF Kernel
The RBF kernel mapped the 22 continuous features into an infinite-dimensional Hilbert space, but with only 8 healthy training participants, the radial basis functions over-clustered around the dominant 24 PD subjects. Consequently, `svm_rbf` suffered the lowest ROC-AUC of all candidates (`0.5833` in LOGO, lower than the linear baseline's `0.7031`), proving that unconstrained kernel methods are unsuited for this specific dataset.

---

## 7. Conclusions & Next Steps

1. **Shortlist for Stage 2C & 2D:**
   - **Champion Candidate:** `ExtraTreesClassifier` (highest overall ROC-AUC, best specificity retention, fast CPU execution).
   - **Challenger Candidate:** `XGBoost (CPU)` (lowest Brier score, excellent PR-AUC, high recall, highly responsive to threshold tuning).
2. **Models Eliminated:**
   - `SVM (RBF)`: Eliminated due to severe ranking collapse (`ROC-AUC = 0.5833`).
   - `GradientBoosting` & `HistGradientBoosting`: Eliminated in favor of `XGBoost`, which demonstrated superior probability calibration and higher ROC-AUC.
3. **Hardware Recommendation:** Run tree ensembles on CPU; on this small dataset, CPU avoids device transfer overhead.
