# Pathway A Model Experimentation Plan

**Document Version:** 1.0.0  
**Target Module:** `ml/pathway_a/`  
**Evaluation Scope:** UCI Parkinson's Dataset (`parkinsons.data`)  
**Status:** Approved for Implementation (Documentation Only)

---

## 1. Research Objective

The primary objective of Pathway A model experimentation is to develop, evaluate, and select a statistically sound, clinically aligned machine learning pipeline for binary Parkinson's Disease (PD) screening using vocal biomarkers. 

The pipeline must:
- Discriminate PD subjects from healthy controls using 22 biomedical voice measures.
- Maximize screening sensitivity (Recall) without suffering from the severe specificity collapse observed in uncalibrated baselines.
- Prevent speaker leakage and optimistic bias caused by repeated vocal recordings per subject.
- Strictly adhere to the existing frozen 22-feature input contract consumed by the FastAPI backend and React frontend.
- Maintain a parsimonious, transparent architecture that avoids overfitting the limited sample cohort.

---

## 2. Dataset Characteristics and Limitations

### Cohort & Dimensions
- **File:** `ml/pathway_a/data/raw/parkinsons.data` (verified SHA256: `a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5`).
- **Instances:** 195 voice recordings across 32 unique human participants (`phon_R01_S01_1` through `phon_R01_S50_6`).
- **Features:** 22 continuous numerical acoustic features (`float64`), 1 subject/recording identifier column (`name`), and 1 binary label column (`status`).
- **Data Quality:** Zero missing values (`NaN`), zero non-finite values (`inf`), and zero exact duplicate feature vectors.

### Severe Sample Limitations
1. **Participant Imbalance & Small Control Cohort:**
   - 24 participants diagnosed with Parkinson's Disease (75.0% of cohort; 147 recordings).
   - Only 8 healthy control participants (25.0% of cohort; 48 recordings).
   - In any validation fold, each held-out healthy individual exerts disproportionate leverage over specificity estimates.
2. **Repeated Correlated Measurements:**
   - 29 participants contributed 6 sustained vowel phonations; 3 participants (`S21`, `S27`, `S35`) contributed 7 phonations.
   - Intrasubject correlation across vocal sessions is high. Treating individual recordings as independent samples introduces immediate, catastrophic data leakage.
3. **Severe Multicollinearity:**
   - 19 feature pairs exhibit Pearson correlation $\|r\| \ge 0.95$.
   - Perfect linear redundancies exist: `Jitter:DDP` is identical to $3 \times \text{MDVP:RAP}$ ($r = 1.0$), and `Shimmer:DDA` is identical to $3 \times \text{Shimmer:APQ3}$ ($r = 1.0$).
4. **Distributional Skewness:**
   - 8 features show extreme positive skewness ($\|\text{skew}\| > 2.0$), most notably `NHR` (skewness = 4.22), `MDVP:RAP` (3.36), and `MDVP:Jitter(%)` (3.08).

---

## 3. Baseline Benchmark Reference

### The Legacy Fold-0 Limitation
The original baseline implementation in `train_baseline.py` executed `next(splitter.split(X, y, groups))` using an arbitrary 5-fold split (`seed=42`).
- **Critical Flaw:** It evaluated test metrics on **only 1 healthy control participant** (Subject `S01`, contributing 6 recordings: 2 true negatives, 4 false positives) alongside 5 PD participants (31 recordings).
- **Misleading Generalization:** The resulting metrics (Accuracy: 0.7027, Precision: 0.8571, Recall: 0.7742, Specificity: 0.3333, ROC-AUC: 0.7151) reflected the vocal idiosyncrasies of one healthy subject rather than true model performance.
- When evaluated across all 5 folds, fold specificity ranged from 0.0000 to 1.0000, and ROC-AUC ranged from 0.5056 to 0.9839. Single-fold reporting is scientifically indefensible.

### Verified Reference Benchmarks
Using the isolated test harness in `ml/pathway_a/experiments/run_experiments.py`, the existing baseline pipeline (`StandardScaler` + `LogisticRegression(class_weight="balanced", random_state=42)`) established the following verified benchmarks:

| Evaluation Scheme | Level | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | Balanced Acc | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Repeated 4×10 SGKF** (Mean over 40 folds) | Recording | 0.7036 | 0.8390 | 0.7649 | 0.5146 | 0.7916 | 0.7448 | 0.9075 | 0.6398 | 0.2164 |
| **Repeated 4×10 SGKF** (Std over 40 folds) | Recording | ±0.1076 | ±0.0925 | ±0.1298 | ±0.3243 | ±0.0855 | ±0.1401 | ±0.0509 | ±0.1570 | ±0.0761 |
| **Repeated 4×10 SGKF** (10-Seed OOF Mean) | Recording | 0.7036 | 0.8284 | 0.7653 | 0.5146 | 0.7955 | 0.7365 | 0.8982 | 0.6399 | 0.2164 |
| **32-Fold LOGO** (Pooled OOF) | Recording | 0.6974 | 0.8235 | 0.7619 | 0.5000 | 0.7915 | 0.7238 | 0.8937 | 0.6310 | 0.2174 |
| **Repeated 4×10 SGKF** (Mean over 40 folds) | Participant | 0.7344 | 0.8390 | 0.8125 | 0.5000 | 0.8136 | 0.7417 | 0.9134 | 0.6562 | 0.1865 |
| **32-Fold LOGO** (Pooled OOF) | Participant | 0.7500 | 0.8333 | 0.8333 | 0.5000 | 0.8333 | 0.7031 | 0.8803 | 0.6667 | 0.1881 |

All candidate models must be compared against these 40-fold and 32-fold reference figures.

---

## 4. Candidate Models

Given that $N_{\text{participants}} = 32$ and $N_{\text{recordings}} = 195$, models must exhibit low parameter complexity or strong structural regularization. Deep neural networks or unconstrained gradient boosters with hundreds of leaves are prohibited.

### 1. Regularized Linear Classifiers (Baseline & Extensions)
- **Logistic Regression (L1 / Lasso):** Induces sparsity, automatically zeroing redundant collinear acoustic features.
- **Logistic Regression (L2 / Ridge):** Shrunk coefficients under multicollinearity; stable parameter variance.
- **Logistic Regression (ElasticNet):** Balances sparsity and grouping of correlated vocal measures (e.g., jitter and shimmer clusters).

### 2. Support Vector Machines (Kernel-based)
- **SVM (RBF Kernel):** Effective for nonlinear classification boundaries in compact biological spaces; bounded margin optimization resists small-sample overfitting. Must be paired with Platt scaling (`probability=True`) for downstream probability estimation.
- **SVM (Linear Kernel):** Benchmark for maximum-margin linear separation under balanced class weighting.

### 3. Tree Ensembles
- **Random Forest:** Bagging reduces estimation variance; invariant to monotonic feature scaling and skewness; robust against outlier recording bursts. Hyperparameters must restrict depth (`max_depth=3..5`, `min_samples_leaf=3..5`, `max_features='sqrt'`).
- **Extra Trees (Extremely Randomized Trees):** Introduces random threshold cuts, often yielding smoother decision boundaries and lower variance on small tabular biomedical datasets.

### 4. Gradient Boosting
- **Gradient Boosting Classifier (`GradientBoostingClassifier` / `HistGradientBoostingClassifier`):** Sequentially fits shallow decision stumps (`max_depth=2..3`, `learning_rate=0.03..0.1`). Regularized with `subsample=0.8` and early stopping.
- **XGBoost:** Permitted only if constrained to conservative regularization (`max_depth=2..3`, `colsample_bytree=0.7`, `subsample=0.8`, `reg_alpha > 0`, `reg_lambda > 1`). Must justify whether it offers any empirical advantage over standard scikit-learn tree ensembles on 32 subjects.

---

## 5. Preprocessing Experiments

All preprocessing transformations must be fitted **strictly inside training folds** via `sklearn.pipeline.Pipeline`. No statistics (means, standard deviations, quantiles) may be calculated across the full dataset prior to fold splitting.

### Transformations to Evaluate
1. **StandardScaler (Current Baseline):** Zero-mean, unit-variance scaling. Sensitive to heavy-tailed voice measurements (e.g., `NHR`, `Jitter`).
2. **RobustScaler:** Uses median and interquartile range (IQR). Prevents extreme pitch fluctuations and vocal tremors from distorting normalization centers.
3. **PowerTransformer (Yeo-Johnson):** Applies parametric power transforms to stabilize variance and minimize skewness across the 8 highly skewed vocal features. Valid for positive and negative values (e.g., `spread1`).
4. **QuantileTransformer (Uniform / Normal):** Non-linear mapping to Gaussian distribution; suppresses outlier distortion but risks flattening subtle biomarker gradations.

### Model-Specific Applicability
- **Linear Models & SVM:** Require robust scaling or power transformation due to sensitivity to feature magnitude and non-normality.
- **Tree-Based Models (RF, Extra Trees, Boosting):** Invariant to monotonic scaling. Scaling experiments on tree models serve as a control to ensure transformations do not introduce numerical quantization artifacts.

---

## 6. Feature Analysis & Redundancy Reduction

### Multicollinearity Structure
- **Jitter cluster:** `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`.
- **Shimmer cluster:** `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`.
- **Harmonics / Noise:** `NHR`, `HNR`.
- **Nonlinear Dynamics:** `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`.

### Feature Selection Rules
- **Prohibited:** Any global feature selection (e.g., computing mutual information or dropping features on the 195 rows before cross-validation). This introduces selection bias and invalidates error estimates.
- **Permitted Inside Fold Pipelines:**
  1. **Deterministic Redundancy Dropping:** Eliminating exact mathematical identities (`Jitter:DDP`, `Shimmer:DDA`) within the transformer step while still accepting the full 22-feature API input.
  2. **Variance Thresholding:** Removing near-zero variance features (none currently present, but standard sanity check).
  3. **L1 Penalty / SelectFromModel:** Fitting Lasso within the training fold to select active feature subsets.
  4. **PCA / Dimensionality Reduction:** Orthogonal projection of correlated acoustic clusters (e.g., reducing the 5 jitter features to 1 principal component).

---

## 7. Evaluation Strategy

To guarantee that performance claims reflect true clinical generalization across unseen speakers, evaluation must strictly prevent speaker leakage.

```
Full Cohort: 32 Participants (24 PD, 8 Healthy) — 195 Recordings
   │
   ├── Outer Loop: Repeated 4-Fold StratifiedGroupKFold (10 Seeds = 40 Folds)
   │      Each Fold: Exactly 6 PD Participants + 2 Healthy Participants in Test Fold
   │      Train: 18 PD + 6 Healthy Participants (~146 recordings)
   │      Test:   6 PD + 2 Healthy Participants (~49 recordings)
   │      Assert: train_participants ∩ test_participants == ∅
   │
   └── Confirmatory Loop: 32-Fold Leave-One-Group-Out (LOGO)
          Each Fold: 31 Participants in Train, 1 Participant in Test
          Produces: 32 pure Out-Of-Sample subject predictions
```

### Evaluation Levels
1. **Recording Level:** Evaluates performance on every individual vocal phonation ($N=195$).
2. **Participant Level (Primary Clinical Target):** Aggregates probabilities for each speaker:
   $$\bar{P}_{\text{PD}}(S_i) = \frac{1}{K_i} \sum_{k=1}^{K_i} P_{\text{PD}}(\text{recording}_{i,k})$$
   A patient is screened positive if $\bar{P}_{\text{PD}}(S_i) \ge \tau$.

### Nested Cross-Validation (Model Selection)
When selecting hyperparameters or tuning decision thresholds, a **Nested CV** design must be used:
- **Outer Loop:** 4-Fold StratifiedGroupKFold (estimates generalization error).
- **Inner Loop:** 3-Fold StratifiedGroupKFold or LOGO on outer training data only (selects optimal hyperparameters $\theta^*$).
- Test fold predictions are generated using $\theta^*$ fitted on the outer training fold.

---

## 8. Evaluation Metrics

Metrics must be reported with their mean and standard deviation across folds, as well as on pooled out-of-fold (OOF) predictions.

1. **Recall / Sensitivity ($\text{TP} / [\text{TP} + \text{FN}]$):** Clinical priority. Missing a patient with Parkinson's Disease carries severe clinical costs.
2. **Specificity ($\text{TN} / [\text{TN} + \text{FP}]$):** Must exceed 0.60 to avoid overwhelming healthcare providers with false positive referrals.
3. **Balanced Accuracy ($[\text{Sensitivity} + \text{Specificity}] / 2$):** Primary thresholded metric, neutralizing the 3:1 class imbalance.
4. **ROC-AUC:** Threshold-independent ranking ability across the entire operating spectrum.
5. **PR-AUC (Average Precision):** Measures precision-recall tradeoff on the positive (PD) class under class imbalance.
6. **F1-Score:** Harmonic mean of precision and recall.
7. **Brier Score Loss ($\frac{1}{N} \sum (y_i - p_i)^2$):** Measures calibration quality of output probabilities. Lower is better.

---

## 9. Decision Threshold Selection

The default threshold $\tau = 0.50$ is suboptimal when models use `class_weight="balanced"` or when class prevalence is asymmetric.

### Methodological Constraints
- **Strictly In-Fold Tuning:** Thresholds must be optimized on out-of-fold validation predictions *within the training fold* (or via `TunedThresholdClassifierCV` fit on the training split).
- **Zero Test Leakage:** Never select a threshold by inspecting outer test-fold confusion matrices.
- **Objective Function:** Select $\tau^*$ to maximize Youden's Index ($J = \text{Sensitivity} + \text{Specificity} - 1$) subject to a minimum specificity constraint:
  $$\tau^* = \arg\max_{\tau} (\text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1) \quad \text{s.t.} \quad \text{Specificity}(\tau) \ge 0.60$$

---

## 10. Probability Calibration

The web application displays predicted probabilities (`pd_probability`, `healthy_probability`) and a derived `confidence` score. Uncalibrated models output distorted risk scores.

### Calibration Protocol
1. Model comparisons will first be conducted using raw/uncalibrated probabilities to evaluate inherent ranking performance (ROC-AUC, PR-AUC).
2. For top-performing candidates, evaluate:
   - **Platt Scaling (Sigmoid / Logistic):** Suitable for SVM and small linear models.
   - **Isotonic Regression:** Non-parametric; prone to severe overfitting on small datasets ($N=32$ subjects) and therefore discouraged unless validated under nested CV.
3. Verify calibration using Brier score and reliability curves.
4. Do not overclaim probability confidence: an output of 0.98 must reflect empirical likelihood, not an artifact of unregularized margin maximization.

---

## 11. Staged Experiment Order

To maintain scientific rigor and avoid combinatorial explosion, experiments will execute in 4 discrete stages:

```
Stage 1: Preprocessing & Scaling Benchmark
├── Models: Logistic Regression (L2), SVM (RBF), Random Forest (default depth)
├── Preprocessors: StandardScaler vs. RobustScaler vs. PowerTransformer
└── Decision: Identify best scaling strategy per model family.

Stage 2: Model Family Exploration & Regularization
├── Candidates: Linear (L1, L2, ElasticNet), SVM (RBF, Lin), Tree (RF, ExtraTrees), Boosting (GradBoost, HistGradBoost)
├── Preprocessor: Optimal pipeline from Stage 1
└── Decision: Shortlist top 2 model families based on Balanced Accuracy and ROC-AUC.

Stage 3: Feature Redundancy & Selection
├── Test: Explicit duplicate removal (DDP, DDA) & L1 feature pruning inside fold
├── Candidates: Top 2 shortlisted model families
└── Decision: Determine whether feature selection improves participant-level generalization.

Stage 4: Calibration & Decision Threshold Tuning
├── Methods: Platt scaling & Youden-optimized decision thresholds
├── Candidate: Champion pipeline from Stage 3
└── Decision: Final model candidate ready for validation and promotion review.
```

---

## 12. Model Selection Criteria

To qualify as the champion model replacing the baseline, a candidate must satisfy all of the following:

1. **Superior Generalization:** Demonstrates higher Balanced Accuracy and ROC-AUC than the baseline (`Balanced Acc > 0.64`, `ROC-AUC > 0.74`) on Repeated 4-fold SGKF and 32-fold LOGO.
2. **Balanced Operating Point:** Achieves Sensitivity $\ge 0.80$ while maintaining Specificity $\ge 0.60$ at the participant level.
3. **Fold Stability:** Demonstrates low standard deviation across SGKF seeds ($\sigma_{\text{BalAcc}} < 0.15$).
4. **Parsimony & Anti-Overfitting:** Preference is given to simpler architectures (regularized linear or shallow tree ensembles) over complex models with identical performance.
5. **Calibrated Output:** Brier score lower than baseline benchmark ($< 0.20$).

---

## 13. Risk Management & Clinical Context

| Risk | Nature | Mitigation |
| :--- | :--- | :--- |
| **Small Control Cohort** | Only 8 healthy individuals. High risk that model memorizes control speaker quirks. | Strict 4-fold SGKF (always 2 controls per test split); confirmatory 32-fold LOGO; participant-level evaluation. |
| **Speaker Leakage** | Same subject appearing in train and test splits leads to false 95%+ accuracy. | Automated disjoint participant assertions (`assert train.isdisjoint(test)`) enforced on every fold. |
| **Model Overfitting** | Excessive tree depth or loose SVM margins memorizing noise. | Hard constraints on tree depth (`max_depth <= 4`), minimum leaf samples ($\ge 4$), and compact hyperparameter grids. |
| **Collinearity Distortion** | Inflated coefficient variance in linear models. | L1/L2 regularization; evaluating explicit removal of redundant features ($r=1.0$). |
| **Dataset Bias** | UCI dataset collected under specific acoustic lab conditions in 2007. | Document in model metadata that performance cannot be assumed to generalize to varied microphones or web audio. |
| **Clinical Misuse** | User interpreting screening risk as definitive diagnosis. | Preserve existing mandatory disclaimer and mock flag contracts across all API responses. |

---

## 14. Final Model Promotion Checklist

Before any experimental model is promoted to replace `ml/pathway_a/models/baseline_pipeline.joblib`:

- [ ] **Contract Compatibility:** Pipeline accepts a `pandas.DataFrame` or `dict` with the exact 22 feature names in original order and produces `predict()` and `predict_proba()` output.
- [ ] **API Invariance:** Zero changes required in `backend/` or `frontend/`.
- [ ] **Reproducibility:** Training script (`train_baseline.py` or replacement) sets all random seeds and produces identical artifacts across reruns.
- [ ] **Versioned Metadata:** Joblib artifact contains `metadata` dictionary detailing algorithm, hyperparameter configuration, scikit-learn version, training date, and CV metrics.
- [ ] **Metrics Documentation:** Replacement `baseline_metrics.json` records 4-fold SGKF and LOGO metrics (recording-level and participant-level).
- [ ] **Automated Tests Passing:** All backend test suites (`pytest backend/tests`) and ML contract tests pass with 100% compliance.
- [ ] **No Regression:** Champion model strictly outperforms the baseline on both sensitivity and specificity under identical CV evaluation.
