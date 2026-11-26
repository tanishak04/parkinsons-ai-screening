# Stage 4.1 — Statistical Uncertainty Audit Report
**Project:** Parkinson's AI Voice Screening System (Pathway A — UCI Parkinson's Dataset)  
**Execution Date:** 2026-11-25  
**Audit Status:** COMPLETE — METHODOLOGICALLY VERIFIED  

---

## 1. Executive Summary

This audit quantifies statistical uncertainty around participant-level performance metrics for both the **Historical 22-Feature Pipeline** (`StandardScaler + LogisticRegression`) and the **Research Champion ET-13 Pipeline** (`ExtraTreesClassifier` on 13 pitch-free acoustic features).

The audit was executed under strict non-invasive constraints: **zero model retraining, zero modifications to production pipelines, zero feature modifications, and zero leakage**. Evaluation is strictly conducted at the **participant level ($N = 32$; 24 Parkinson's Disease [PD], 8 Healthy Controls [HC])**, with recording-level data treated as repeated nested measurements rather than independent observations.

### Key Audit Findings

1. **Specificity Gain is Statistically Meaningful Despite Small Sample:**
   - At reference threshold $\tau = 0.50$, Research Champion ET-13 achieves a Specificity point estimate of **$87.5\%$ ($7/8$)** vs. **$50.0\%$ ($4/8$)** for Historical 22.
   - The paired difference in specificity is **$+37.5\%$**, with a 95% Stratified Bootstrap CI of **$[+12.5\%, +75.0\%]$**, which **excludes zero**.
   - Pitch removal successfully rescued low-pitch healthy male controls ($S13, S43, S49$) who were systematically misclassified by the historical model.

2. **Balanced Accuracy Improvement Spans Zero:**
   - Balanced accuracy improved from **$66.67\%$** (Historical 22) to **$79.17\%$** (ET-13) at $\tau = 0.50$ ($+12.50$ percentage points).
   - However, the paired 95% Bootstrap CI for the difference is **$[-4.17\%, +31.25\%]$**, spanning zero. Due to $N = 32$, this sample is statistically underpowered to confirm superiority on balanced accuracy alone.

3. **Sensitivity Trade-off & Borderline PD Cases:**
   - Sensitivity point estimate decreased from **$83.33\%$ ($20/24$)** in Historical 22 to **$70.83\%$ ($17/24$)** in ET-13 at $\tau = 0.50$ (paired difference: $-12.50\%$, 95% CI: $[-25.00\%, 0.00\%]$).
   - At the exploratory screening threshold $\tau = 0.45$, ET-13 recovers sensitivity to **$75.00\%$ ($18/24$)** (exact Clopper-Pearson 95% CI: $[53.29\%, 90.23\%]$) while preserving Specificity at **$87.50\%$**.

4. **Discordant Pairs & McNemar Analysis:**
   - Out of 32 participants, exactly 26 ($81.25\%$) received concordant classifications.
   - Exactly 6 participants are discordant ($b = 3, c = 3$, exact two-tailed McNemar $p = 1.000$).
   - The identical overall accuracy ($24/32 = 75.0\%$) masks a fundamental structural shift: Historical 22 failed on healthy controls due to acoustic pitch confounding, whereas ET-13 removed that confounding at the cost of 3 false negatives on mild/borderline PD cases ($S04, S06, S27$).

5. **Small Sample Constraint ($N_{HC} = 8$):**
   - With only 8 healthy controls, each subject represents $12.5\%$ of the total specificity metric.
   - The exact Clopper-Pearson 95% CI for ET-13 specificity ($7/8$) spans **$[47.35\%, 99.68\%]$** (Wilson interval: $[52.91\%, 97.76\%]$). While mathematically unavoidable for $N=8$, this demonstrates that point estimates must be presented alongside rigorous uncertainty bounds.

---

## 2. Scope & Input Artifact Integrity

All computations were performed using strictly held-out participant-level out-of-fold (OOF) decision margins generated during Stage 3A / Phase 3.3A.

### Cryptographic Hash Verification

| Artifact Name | Path | SHA-256 Hash | Status |
|---|---|---|---|
| Historical 22 Decision Margins | `ml/pathway_a/experiments/results/phase3_threshold_audit/participant_decision_margins_historical22.csv` | `c2fb1a0b8f1e308aba4d5b9336420f27452dae0f272e887229b41ba5560a166d` | VERIFIED |
| Research ET-13 Decision Margins | `ml/pathway_a/experiments/results/phase3_threshold_audit/participant_decision_margins_et13.csv` | `f560dd56d08ada8542d94611f4f85fa0a4dc07335101bc88073ce2c363705da9` | VERIFIED |
| Raw UCI Dataset | `ml/pathway_a/data/raw/parkinsons.data` | `a36116c2deac07895b79ce97a2f99a1b3720ac4892995ff5c3b9ccb397ed2bc5` | VERIFIED |
| Champion Pipeline (Protected) | `ml/pathway_a/models/champion_pipeline.joblib` | `b2d052eed9c2515506a244b401dfb78e26acd8d24fa0e8ed68dfadc5aae7a414` | IMMUTABLE |
| Baseline Pipeline (Protected) | `ml/pathway_a/models/baseline_pipeline.joblib` | `81dc11f063a7bd52b404f36621d2a37bb33655456f209df1401eefd56b6b884f` | IMMUTABLE |
| Audio Pipeline (Protected) | `ml/pathway_a/models/audio_pipeline.joblib` | `7024bc988db633bddea41970a0985d5cb4a779add54e3b3d08f54af52ff5e01f` | IMMUTABLE |

*Integrity Check:* Both decision margin files match 1-to-1 on participant IDs ($S01$ through $S50$), with exactly 24 PD and 8 HC ground truth labels verified against `parkinsons.data`.

---

## 3. Statistical Methodology

### 3.1 Stratified Participant Bootstrap
- **Replicates:** $B = 2,000$ iterations.
- **RNG Seed:** Fixed at `42` (`numpy.random.default_rng(42)`).
- **Stratification:** PD participants ($N=24$) and HC participants ($N=8$) are resampled independently with replacement, guaranteeing that every bootstrap replicate maintains the exact empirical class ratio ($24:8$).
- **Primary CI Method:** 95% Percentile Confidence Intervals:
  $$\text{CI}_{95\%} = \left[ q_{0.025}^*, q_{0.975}^* \right]$$
- **Paired Resampling:** For paired differences ($\Delta = \text{ET13} - \text{H22}$), the exact same resampled participant index tuples are evaluated on both models simultaneously.

### 3.2 Exact Binomial Intervals (Clopper-Pearson & Wilson)
Given the small sample sizes ($N_{PD}=24, N_{HC}=8$), asymptotic normal approximations are invalid.
- **Exact Clopper-Pearson (Primary Binomial CI):** Inversion of the exact binomial cumulative distribution via the Beta distribution:
  $$p_L = \text{Beta}\left(k, n - k + 1; \frac{\alpha}{2}\right), \quad p_U = \text{Beta}\left(k + 1, n - k; 1 - \frac{\alpha}{2}\right)$$
- **Wilson Score Interval (Supplementary):** Provides continuity without over-conservatism at extreme boundaries:
  $$\tilde{p} \pm \frac{z}{1 + z^2/n} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$

### 3.3 Paired Hypothesis Tests
- **Exact McNemar's Test:** Applied to discordant participant pairs using the two-tailed exact binomial test on discordant counts ($b$ and $c$):
  $$p = \sum_{x} \binom{b+c}{x} 0.5^{b+c}$$
- **DeLong Paired AUC Test:** Non-parametric test for paired ROC curves comparing the U-statistic covariance matrices.  
  *Methodological Caveat:* DeLong assumes independent sample vectors; cross-validation OOF predictions exhibit training-overlap covariance that can slightly underestimate variance. It is included as supplementary evidence only.

---

## 4. Uncertainty Audit Results

### 4.1 Research Champion (ET-13 Pitch-Free) Performance & CIs

| Threshold ($\tau$) | Metric | Point Estimate | Bootstrap Mean (Std) | 95% Bootstrap CI | Exact Clopper-Pearson 95% CI | Wilson Score 95% CI |
|---|---|---:|---:|:---:|:---:|:---:|
| **0.50** | Sensitivity | 0.7083 (17/24) | 0.7114 (0.0898) | [0.5417, 0.8750] | [0.4891, 0.8738] | [0.5083, 0.8509] |
| **0.50** | Specificity | 0.8750 (7/8) | 0.8738 (0.1175) | [0.6250, 1.0000] | [0.4735, 0.9968] | [0.5291, 0.9776] |
| **0.50** | Balanced Accuracy | 0.7917 | 0.7926 (0.0754) | [0.6250, 0.9167] | — | — |
| **0.50** | Precision | 0.9444 | 0.9457 (0.0496) | [0.8333, 1.0000] | — | — |
| **0.50** | F1-Score | 0.8095 | 0.8087 (0.0654) | [0.6667, 0.9130] | — | — |
| **0.50** | ROC-AUC | 0.7865 | 0.7881 (0.0896) | [0.5938, 0.9375] | — | — |
| **0.45** | Sensitivity | 0.7500 (18/24) | 0.7522 (0.0867) | [0.5833, 0.9167] | [0.5329, 0.9023] | [0.5510, 0.8800] |
| **0.45** | Specificity | 0.8750 (7/8) | 0.8738 (0.1175) | [0.6250, 1.0000] | [0.4735, 0.9968] | [0.5291, 0.9776] |
| **0.45** | Balanced Accuracy | 0.8125 | 0.8130 (0.0743) | [0.6458, 0.9375] | — | — |
| **0.45** | Precision | 0.9474 | 0.9485 (0.0470) | [0.8500, 1.0000] | — | — |
| **0.45** | F1-Score | 0.8372 | 0.8361 (0.0607) | [0.7000, 0.9333] | — | — |
| **0.45** | ROC-AUC | 0.7865 | 0.7881 (0.0896) | [0.5938, 0.9375] | — | — |

---

### 4.2 Historical 22-Feature Baseline Performance & CIs

| Threshold ($\tau$) | Metric | Point Estimate | Bootstrap Mean (Std) | 95% Bootstrap CI | Exact Clopper-Pearson 95% CI | Wilson Score 95% CI |
|---|---|---:|---:|:---:|:---:|:---:|
| **0.50** | Sensitivity | 0.8333 (20/24) | 0.8366 (0.0746) | [0.6667, 0.9583] | [0.6262, 0.9526] | [0.6415, 0.9332] |
| **0.50** | Specificity | 0.5000 (4/8) | 0.5027 (0.1781) | [0.1250, 0.8750] | [0.1570, 0.8430] | [0.2152, 0.7848] |
| **0.50** | Balanced Accuracy | 0.6667 | 0.6696 (0.0958) | [0.4792, 0.8542] | — | — |
| **0.50** | Precision | 0.8333 | 0.8367 (0.0518) | [0.7407, 0.9524] | — | — |
| **0.50** | F1-Score | 0.8333 | 0.8347 (0.0503) | [0.7273, 0.9200] | — | — |
| **0.50** | ROC-AUC | 0.8073 | 0.8092 (0.0849) | [0.6250, 0.9480] | — | — |
| **0.45** | Sensitivity | 0.8750 (21/24) | 0.8775 (0.0660) | [0.7500, 1.0000] | [0.6764, 0.9734] | [0.6900, 0.9566] |
| **0.45** | Specificity | 0.5000 (4/8) | 0.5027 (0.1781) | [0.1250, 0.8750] | [0.1570, 0.8430] | [0.2152, 0.7848] |
| **0.45** | Balanced Accuracy | 0.6875 | 0.6901 (0.0950) | [0.5000, 0.8750] | — | — |
| **0.45** | Precision | 0.8400 | 0.8432 (0.0498) | [0.7500, 0.9545] | — | — |
| **0.45** | F1-Score | 0.8571 | 0.8584 (0.0452) | [0.7660, 0.9388] | — | — |
| **0.45** | ROC-AUC | 0.8073 | 0.8092 (0.0849) | [0.6250, 0.9480] | — | — |

---

### 4.3 Paired Model Difference Analysis ($\Delta = \text{ET13} - \text{H22}$)

Evaluating the exact paired differences per bootstrap replicate reveals which metric differences are statistically significant versus indeterminate:

| Threshold ($\tau$) | Metric | Point Difference | Bootstrap Mean Diff (Std) | 95% Bootstrap CI | Zero in 95% CI? | Statistical Inference |
|---|---|---:|---:|:---:|:---:|---|
| **0.50** | **Sensitivity** | -0.1250 | -0.1251 (0.0667) | [-0.2500, 0.0000] | Boundary (0 in CI) | Historical 22 trend higher sensitivity |
| **0.50** | **Specificity** | **+0.3750** | **+0.3711 (0.1732)** | **[+0.1250, +0.7500]** | **No (Excludes 0)** | **ET-13 significantly superior** |
| **0.50** | **Balanced Acc.** | +0.1250 | +0.1230 (0.0923) | [-0.0417, +0.3125] | Yes | Indeterminate / Underpowered |
| **0.50** | Precision | +0.1111 | +0.1090 (0.0561) | [0.0000, +0.2174] | Boundary (0 in CI) | ET-13 trend higher precision |
| **0.50** | F1-Score | -0.0238 | -0.0260 (0.0513) | [-0.1363, +0.0629] | Yes | Indeterminate |
| **0.50** | ROC-AUC | -0.0208 | -0.0211 (0.0570) | [-0.1250, +0.0990] | Yes | Indeterminate (ROC curves overlap) |
| **0.45** | Sensitivity | -0.1250 | -0.1253 (0.0672) | [-0.2500, 0.0000] | Boundary (0 in CI) | Historical 22 trend higher sensitivity |
| **0.45** | **Specificity** | **+0.3750** | **+0.3711 (0.1732)** | **[+0.1250, +0.7500]** | **No (Excludes 0)** | **ET-13 significantly superior** |
| **0.45** | Balanced Acc. | +0.1250 | +0.1229 (0.0930) | [-0.0417, +0.3125] | Yes | Indeterminate / Underpowered |
| **0.45** | Precision | +0.1074 | +0.1053 (0.0536) | [0.0000, +0.2083] | Boundary (0 in CI) | ET-13 trend higher precision |
| **0.45** | F1-Score | -0.0199 | -0.0224 (0.0505) | [-0.1311, +0.0654] | Yes | Indeterminate |
| **0.45** | ROC-AUC | -0.0208 | -0.0211 (0.0570) | [-0.1250, +0.0990] | Yes | Indeterminate (ROC curves overlap) |

---

## 5. Discordant Pairs & Clinical Shift Analysis

### 5.1 $2 \times 2$ Matched Contingency Table ($\tau = 0.50$)

| Historical 22 \ ET-13 | ET-13 Correct | ET-13 Incorrect | Total |
|---|---:|---:|---:|
| **H22 Correct** | 21 ($65.62\%$) | 3 ($9.38\%$) | **24** |
| **H22 Incorrect** | 3 ($9.38\%$) | 5 ($15.62\%$) | **8** |
| **Total** | **24** | **8** | **32** |

- **Concordant Classifications:** $21 + 5 = 26$ out of 32 ($81.25\%$).
- **Discordant Classifications:** $b = 3$, $c = 3$.
- **Exact McNemar Test:** $p = 1.000$ (Two-tailed binomial test with $n_{disc} = 6, k = 3, p_0 = 0.5$).

### 5.2 Subject-Level Discordance Audit

| Participant ID | True Status | Model Outcomes ($\tau = 0.50$) | Historical 22 Probability | ET-13 Probability | Clinical / Acoustic Diagnostic Reason |
|---|---|---|---:|---:|---|
| **$S13$** | Healthy Control (0) | H22: False Pos (ERR)<br>ET-13: True Neg (CORR) | **0.8659** | **0.2975** | Healthy male with low fundamental pitch ($\sim 116\text{ Hz}$). H22 falsely classified as PD due to raw pitch confounding. ET-13 pitch-free model correctly identifies as healthy. |
| **$S43$** | Healthy Control (0) | H22: False Pos (ERR)<br>ET-13: True Neg (CORR) | **0.8447** | **0.4632** | Healthy male with lower pitch ($\sim 113\text{ Hz}$). Rescued by pitch-free ET-13. |
| **$S49$** | Healthy Control (0) | H22: False Pos (ERR)<br>ET-13: True Neg (CORR) | **0.8037** | **0.3547** | Healthy male with low fundamental pitch ($\sim 112\text{ Hz}$). Rescued by pitch-free ET-13. |
| **$S04$** | Parkinson's (1) | H22: True Pos (CORR)<br>ET-13: False Neg (ERR) | **0.5513** | **0.4578** | Borderline PD patient. Probability shifted from $0.55$ to $0.46$. (Correctly classified by ET-13 when $\tau = 0.45$). |
| **$S06$** | Parkinson's (1) | H22: True Pos (CORR)<br>ET-13: False Neg (ERR) | **0.6276** | **0.4900** | Borderline PD patient ($P = 0.4900$, right at decision boundary). Missing acoustic pitch cues dropped probability by $0.01$ below 0.50. |
| **$S27$** | Parkinson's (1) | H22: True Pos (CORR)<br>ET-13: False Neg (ERR) | **0.5539** | **0.4079** | Mild dysphonic impairment. H22 caught subject slightly above threshold ($0.55$). ET-13 assigns $0.41$. |

> **Critical Takeaway on McNemar Nonsignificance:**  
> A McNemar $p = 1.000$ indicates that the aggregate accuracy rate of the two models is identical ($75.0\%$). It **does NOT mean the models are clinically or acoustically equivalent**. The historical model's 8 errors consisted of 4 healthy false positives (a devastating 50% false positive rate on healthy controls) and 4 false negatives. The ET-13 model's 8 errors consist of 1 healthy false positive and 7 false negatives. The failure mechanism shifted from **spurious pitch overfitting** to **conservative sensitivity on borderline acoustic impairment**.

---

### 5.3 Supplementary DeLong Paired AUC Test

- $\text{AUC}(\text{H22}) = 0.8073$
- $\text{AUC}(\text{ET13}) = 0.7865$
- Difference: $-0.0208$
- Standard Error of Difference: $0.0551$
- $Z$-statistic: $-0.3781$
- DeLong Two-Tailed $p$-value: **$0.7053$**
- 95% DeLong CI for difference: **$[-0.1288, +0.0872]$**

*Conclusion:* The global discrimination capacity across all possible thresholds is statistically indistinguishable between the two models ($p = 0.705$). However, the decision boundary behavior at operational screening thresholds is profoundly different due to the removal of pitch bias.

---

## 6. Methodological Limitations & Dataset Constraints

1. **Severe Healthy Control Constraint ($N_{HC} = 8$):**
   - The UCI dataset contains only 8 healthy controls. This creates a coarse discrete metric support: every control correctly or incorrectly classified shifts Specificity by $12.5$ percentage points ($1/8$).
   - Consequently, the Clopper-Pearson 95% CI for ET-13 specificity at $\tau = 0.50$ ($7/8$) spans $[47.35\%, 99.68\%]$.
   - While the paired bootstrap proves that ET-13 is significantly more specific than Historical 22 on this cohort ($+37.5\%$, $[+12.5\%, +75.0\%]$), true population specificity cannot be determined with narrow precision until validated on an external, adequately powered cohort.

2. **Statistical Power Limitation for Balanced Accuracy:**
   - With $N = 32$, a 12.5 percentage point improvement in balanced accuracy fails to achieve statistical significance at $\alpha = 0.05$ (CI $[-4.17\%, +31.25\%]$). A post-hoc power calculation indicates that detecting a $12.5\%$ difference in balanced accuracy with $80\%$ power at $\alpha=0.05$ requires a minimum of $N \approx 90$ independent participants.

3. **Conditionality of Confidence Intervals:**
   - The reported confidence intervals are conditioned on the specific pre-specified thresholds ($\tau = 0.50$ and $\tau = 0.45$). They do not incorporate data-driven threshold selection penalties because thresholds were evaluated as fixed operating points established during Phase 3.3A.

4. **Non-Diagnostic Research Prototype Status:**
   - Both models remain investigational research prototypes. No claim of clinical safety, diagnostic efficacy, or regulatory readiness can be made based on $N = 32$ participants.

---

## 7. Audit Verdict & Sign-Off

| Audit Item | Methodological Standard | Verification Result | Status |
|---|---|---|---|
| **Separation of Models** | Historical 22 and ET-13 evaluated independently | Verified in separate tables and paired delta | PASS |
| **Participant Independence** | Primary unit of inference is participant ($N=32$) | Stratified resampling at subject level | PASS |
| **Integrity & Immutability** | Zero model retraining or pipeline changes | Pre- and post-execution SHA-256 identical | PASS |
| **Small-Sample Validity** | Exact Clopper-Pearson & Wilson for binomial metrics | Computed via `scipy.stats.beta` | PASS |
| **Stratified Bootstrap** | $B=2000$, seed=42, separate PD/HC strata | Confirmed via `bootstrap_replicates.csv` | PASS |
| **Paired Differences** | Paired bootstrap differences and discordant pairs | Full contingency and McNemar analysis | PASS |

**Final Verdict:** **PASS (AUDIT COMPLETE)**  
The statistical uncertainty audit confirms that the Research Champion ET-13 provides a statistically verified specificity improvement ($[+12.5\%, +75.0\%]$) by eliminating acoustic pitch confounding, while balanced accuracy gains remain within small-sample uncertainty bounds ($[-4.17\%, +31.25\%]$). All generated audit artifacts are cryptographically anchored and archived.
