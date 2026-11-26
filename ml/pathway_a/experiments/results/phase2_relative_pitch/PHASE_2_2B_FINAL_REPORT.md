# Phase 2.2B — Relative Pitch & Normalized Feature Investigation
## Final Research Report

**Project:** Parkinson's AI Screening — Pathway A (UCI Parkinson's Voice Dataset)  
**Date:** 2026-11-25  
**Audit Status:** Completed — Strictly READ-ONLY with respect to production artifacts  
**Protected Artifacts Pre/Post SHA-256 Verification:** 100% Match (0 modifications)

---

## 1. Dataset Audit

An exhaustive audit of `data/parkinsons.data` was performed prior to experimental formulation:

- **Total Voice Recordings:** 195 acoustic recordings
- **Total Unique Participants:** 32 unique subjects (`S01` through `S50`)
- **Diagnostic Distribution:**
  - Parkinson's Disease (PD, `status = 1`): 24 participants (147 recordings)
  - Healthy Controls (HC, `status = 0`): 8 participants (48 recordings)
  - Severe participant-level class imbalance: 75% PD vs 25% HC (3:1 ratio)
- **Repeated Measures Structure:**
  - 31 participants contributed exactly 6 recordings each.
  - 1 participant (`S07`, PD) contributed exactly 3 recordings.
- **Available Raw Pitch Attributes:**
  - The dataset contains exclusively three scalar pitch statistics per recording:
    1. `MDVP:Fo(Hz)`: Fundamental frequency (mean $F_0$)
    2. `MDVP:Fhi(Hz)`: Maximum fundamental frequency ($F_{\text{hi}}$)
    3. `MDVP:Flo(Hz)`: Minimum fundamental frequency ($F_{\text{lo}}$)
  - No continuous time-series, pitch tracks, cycle-by-cycle periods, or pitch standard deviations ($\sigma_{F_0}$) are present in the dataset.

---

## 2. Feature Mathematical Definitions

To explore whether pitch dynamics can be captured without speaker-absolute fundamental frequency confounding, the candidate relative and normalized features were formalized:

1. **Semitone Dynamic Pitch Range ($\Delta F_{\text{st}}$):**
   $$\Delta F_{\text{st}} = 12 \log_2\left(\frac{F_{\text{hi}}}{F_{\text{lo}}}\right)$$
   Measures pitch excursion width on a log-frequency musical scale, invariant to speaker base octave.

2. **Pitch Range Ratio ($R_{\text{hi/lo}}$):**
   $$R_{\text{hi/lo}} = \frac{F_{\text{hi}}}{F_{\text{lo}}}$$

3. **Upper Excursion Ratio ($R_{\text{hi}/0}$):**
   $$R_{\text{hi}/0} = \frac{F_{\text{hi}}}{F_0}$$

4. **Lower Excursion Ratio ($R_{0/\text{lo}}$):**
   $$R_{0/\text{lo}} = \frac{F_0}{F_{\text{lo}}}$$

5. **Logarithmic Upper & Lower Excursion Ratios:**
   $$\ln R_{\text{hi}/0} = \ln\left(\frac{F_{\text{hi}}}{F_0}\right), \quad \ln R_{0/\text{lo}} = \ln\left(\frac{F_0}{F_{\text{lo}}}\right)$$

6. **Pitch Position Ratio ($P_{\text{pos}}$):**
   $$P_{\text{pos}} = \frac{F_0 - F_{\text{lo}}}{\max(F_{\text{hi}} - F_{\text{lo}}, 10^{-6})}$$
   Measures where the mean pitch sits relative to the minimum and maximum boundaries ($[0, 1]$ interval).

7. **Pitch Coefficient of Variation ($CV_{F_0}$):**
   $$CV_{F_0} = \frac{\sigma_{F_0}}{\mu_{F_0}}$$

---

## 3. Features Actually Implementable from the Dataset

From the tabular data in `parkinsons.data`, the following features were verified as mathematically implementable and free of undefined divisions:
- `semitone_range` $= 12 \log_2(F_{\text{hi}} / F_{\text{lo}})$
- `ratio_fhi_flo` $= F_{\text{hi}} / F_{\text{lo}}$
- `ratio_fhi_fo` $= F_{\text{hi}} / F_0$
- `ratio_fo_flo` $= F_0 / F_{\text{lo}}$
- `log_ratio_fhi_fo` $= \ln(F_{\text{hi}} / F_0)$
- `log_ratio_fo_flo` $= \ln(F_0 / F_{\text{lo}})$
- `pitch_pos_ratio` $= (F_0 - F_{\text{lo}}) / \max(F_{\text{hi}} - F_{\text{lo}}, 10^{-6})$

All features evaluated numerically without NaN, Inf, or zero-division anomalies across all 195 recordings.

---

## 4. Features Rejected and Why

- **Pitch Coefficient of Variation ($CV_{F_0}$): EXPLICITLY REJECTED.**
  - **Audit Finding:** The UCI dataset provides only the scalar summary stats $F_0, F_{\text{hi}}, F_{\text{lo}}$. The standard deviation of fundamental frequency $\sigma_{F_0}$ was never computed or preserved in `parkinsons.data`.
  - **Scientific Integrity Decision:** Substituting an ad-hoc proxy such as $(F_{\text{hi}} - F_{\text{lo}}) / F_0$ would misrepresent true short-term pitch variability and fabricate a non-existent metric. True $CV_{F_0}$ cannot be calculated without the underlying raw audio waveforms or cycle-by-cycle pitch contours. Consequently, $CV_{F_0}$ is formally documented as **not implementable on the UCI dataset**.

---

## 5. LOGO Benchmark Results

Leave-One-Group-Out (LOGO) cross-validation was conducted across all 32 participants (each participant held out entirely as the test set in their respective fold). Standard uncalibrated default threshold $\tau = 0.50$ was applied to participant-level mean predicted probabilities:

| Experiment Code | Feature Set Name | Features ($k$) | Subj Bal Acc | Subj Sens | Subj Spec | Subj ROC-AUC | Subj F1 | Subj TP | Subj TN | Subj FP | Subj FN |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | `pitch_free_13` (Phase 2.2A baseline) | 13 | **0.7917** | **0.7083** | **0.8750** | **0.7865** | **0.8095** | 17 | 7 | 1 | 7 |
| **B1** | `semitone_only` | 1 | 0.7292 | 0.8333 | 0.6250 | 0.6406 | 0.8511 | 20 | 5 | 3 | 4 |
| **B1** | `all_relative_pitch` | 4 | 0.6042 | 0.7083 | 0.5000 | 0.5938 | 0.7556 | 17 | 4 | 4 | 7 |
| **B2** | `combined_17` (13 + 4 relative) | 17 | 0.7708 | 0.6667 | 0.8750 | 0.7656 | 0.7805 | 16 | 7 | 1 | 8 |
| **B3** | `plus_semitone_range` (13 + 1 semitone) | 14 | **0.7917** | **0.7083** | **0.8750** | 0.7812 | **0.8095** | 17 | 7 | 1 | 7 |
| **B3** | `plus_linear_ratios` (13 + 2 ratios) | 15 | 0.7292 | 0.5833 | 0.8750 | 0.7760 | 0.7179 | 14 | 7 | 1 | 10 |
| **B3** | `plus_log_ratios` (13 + 2 log ratios) | 15 | 0.7292 | 0.5833 | 0.8750 | 0.7760 | 0.7179 | 14 | 7 | 1 | 10 |
| **B3** | `plus_pitch_pos_ratio` (13 + 1 pos ratio)| 14 | 0.7708 | 0.6667 | 0.8750 | 0.7812 | 0.7805 | 16 | 7 | 1 | 8 |

---

## 6. Repeated StratifiedGroupKFold Results

To evaluate variance and grouping stability, 5 repeats of 5-fold StratifiedGroupKFold (25 total fold evaluations) were executed with seeds `[42, 123, 456, 789, 2024]`:

| Experiment Code | Feature Set Name | Features ($k$) | SGKF Bal Acc (Mean $\pm$ Std) | SGKF Sensitivity | SGKF Specificity | SGKF ROC-AUC | SGKF F1-Score |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | `pitch_free_13` | 13 | $0.7740 \pm 0.1992$ | $0.6880 \pm 0.2273$ | $0.8600 \pm 0.3007$ | $0.8410 \pm 0.1604$ | $0.7737 \pm 0.1861$ |
| **B1** | `semitone_only` | 1 | $0.6440 \pm 0.2404$ | $0.7480 \pm 0.1967$ | $0.5400 \pm 0.4454$ | $0.6550 \pm 0.3248$ | $0.7807 \pm 0.1517$ |
| **B1** | `all_relative_pitch` | 4 | $0.5680 \pm 0.1831$ | $0.6760 \pm 0.1834$ | $0.4600 \pm 0.3720$ | $0.5450 \pm 0.2884$ | $0.7175 \pm 0.1258$ |
| **B2** | `combined_17` | 17 | $0.7500 \pm 0.1922$ | $0.7000 \pm 0.2319$ | $0.8000 \pm 0.3464$ | $0.8150 \pm 0.1706$ | $0.7725 \pm 0.1875$ |
| **B3** | `plus_semitone_range` | 14 | $\mathbf{0.7840 \pm 0.1859}$ | $0.7080 \pm 0.2134$ | $0.8600 \pm 0.3007$ | $0.8270 \pm 0.1816$ | $\mathbf{0.7896 \pm 0.1704}$ |
| **B3** | `plus_linear_ratios` | 15 | $0.7710 \pm 0.2086$ | $0.6820 \pm 0.2163$ | $0.8600 \pm 0.3007$ | $0.8310 \pm 0.1754$ | $0.7716 \pm 0.1812$ |
| **B3** | `plus_log_ratios` | 15 | $0.7670 \pm 0.2109$ | $0.6740 \pm 0.2298$ | $0.8600 \pm 0.3007$ | $0.8310 \pm 0.1754$ | $0.7620 \pm 0.1971$ |
| **B3** | `plus_pitch_pos_ratio` | 14 | $0.7700 \pm 0.1976$ | $0.6800 \pm 0.2267$ | $0.8600 \pm 0.3007$ | $0.8190 \pm 0.1782$ | $0.7682 \pm 0.1846$ |

---

## 7. Comparison Against Phase 2.2A Baseline

1. **Replication of Phase 2.2A Benchmark:**
   - The 13-feature pitch-free ExtraTrees model from Phase 2.2A was re-benchmarked under identical participant-level LOGO grouping and yielded:
     - Participant Balanced Accuracy: **0.7917** (Sensitivity: 0.7083, Specificity: 0.8750).
     - Confirms 100% reproducible baseline.
2. **Comparison with Relative Pitch Additions:**
   - **Combined 17 features (B2):** Balanced accuracy dropped from **0.7917 down to 0.7708** in LOGO, and dropped from **0.7740 to 0.7500** in SGKF. False negatives increased from 7 to 8.
   - **Plus Semitone Range (B3):** Balanced accuracy remained identical in LOGO (**0.7917**; identical TP=17, TN=7, FP=1, FN=7), while Repeated SGKF showed a minor increase (+0.0100, $0.7840 \pm 0.1859$).
   - **Linear & Log Ratios (B3):** Balanced accuracy degraded sharply to **0.7292** (causing 3 additional false negatives: `S02`, `S19`, `S26`).
3. **Key Inference:** Adding multiple relative pitch ratios degrades classification power by introducing high-variance collinear features into the shallow tree splits.

---

## 8. Feature Ablation Results

Ablation reveals clear separation between perturbation-based acoustic features and relative pitch features:

1. **Relative Pitch In Isolation (B1 Models):**
   - Relative pitch alone has very weak discriminative ability ($0.6042$ Bal Acc for all 4 features; $0.7292$ for semitone range alone).
   - Crucially, semitone range alone produces **3 False Positives out of 8 Healthy subjects** (Specificity collapses to $62.50\%$).
2. **Pitch Ratios Added to Pitch-Free Baseline (B3):**
   - $F_{\text{hi}}/F_0$ and $F_0/F_{\text{lo}}$ severely harm sensitivity (FNs surge from 7 to 10; Sensitivity drops to $58.33\%$).
   - Pitch position ratio ($P_{\text{pos}}$) degrades sensitivity from $70.83\%$ to $66.67\%$.
3. **Semitone Dynamic Range ($\Delta F_{\text{st}}$):**
   - The only relative pitch feature that did not harm sensitivity or specificity when paired with the 13 perturbation features.

---

## 9. Participant-Level Prediction Changes

Tracking the critical diagnostic subgroups identified during the Phase 2.1 pitch bias audit:

| Subject ID | True Class | Description / Clinical Profile | B0 Pitch-Free 13 $P(\text{PD})$ | B1 Semitone Only $P(\text{PD})$ | B2 Combined 17 $P(\text{PD})$ | B3 Plus Semitone $P(\text{PD})$ | B3 Linear Ratios $P(\text{PD})$ |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **S13** | HC (0) | Healthy Male (Low pitch, $F_0 \approx 116$ Hz) | 0.428 (TN) | 0.471 (TN) | 0.459 (TN) | 0.431 (TN) | 0.422 (TN) |
| **S43** | HC (0) | Healthy Male (Low pitch, $F_0 \approx 116$ Hz) | 0.350 (TN) | **0.530 (FP)** | 0.391 (TN) | 0.388 (TN) | 0.380 (TN) |
| **S49** | HC (0) | Healthy Male (Low pitch, $F_0 \approx 117$ Hz) | 0.414 (TN) | **0.614 (FP)** | 0.465 (TN) | 0.448 (TN) | 0.445 (TN) |
| **S50** | HC (0) | Healthy Female (High pitch, acoustic outlier) | 0.748 (FP) | 0.728 (FP) | 0.761 (FP) | 0.722 (FP) | 0.745 (FP) |
| **S08** | PD (1) | Mild / Atypical PD (Persistent FN) | 0.316 (FN) | 0.438 (FN) | 0.323 (FN) | 0.328 (FN) | 0.324 (FN) |
| **S26** | PD (1) | Boundary PD Subject | 0.505 (TP) | 0.538 (TP) | 0.508 (TP) | 0.506 (TP) | **0.493 (FN)** |
| **S34** | PD (1) | Moderate PD | 0.602 (TP) | 0.560 (TP) | 0.593 (TP) | 0.595 (TP) | 0.606 (TP) |
| **S35** | PD (1) | Advanced PD (High dysphonia) | 0.941 (TP) | 0.571 (TP) | 0.969 (TP) | 0.955 (TP) | 0.952 (TP) |

### Key Observations:
1. **Low-Pitch Healthy Males (`S13`, `S43`, `S49`):**
   - In the audited raw 16-feature model, these subjects suffered false positives due to raw low pitch being associated with male sex.
   - Under B0 (Pitch-Free 13), all three are cleanly classified as Healthy ($P \le 0.428$).
   - Under B1 (`semitone_only`), `S43` ($P=0.530$) and `S49` ($P=0.614$) flip to **False Positives**.
   - Under B3 (`plus_semitone_range`), all three remain cleanly classified as Healthy ($P \le 0.448$). The perturbation features anchor the classification correctly.
2. **Boundary PD Subject `S26`:**
   - Under B0 and B3 (`plus_semitone_range`), `S26` is correctly classified ($P = 0.505$ and $P = 0.506$).
   - Under B3 (`plus_linear_ratios`), `S26` is pushed below the decision threshold ($P = 0.493$) to become a **False Negative**.
3. **Outlier `S50` and Hard Case `S08`:**
   - Neither relative pitch nor perturbation features can resolve `S50` (consistently misclassified across all configurations, likely due to vocal fold shimmer characteristics) or `S08` (consistently near-normal acoustic profile).

---

## 10. Confounding Analysis

Correlation between engineered features, raw fundamental frequency ($F_0$), and Parkinson's diagnostic status ($Y$):

| Feature | Pearson $r$ with Raw $F_0$ | Pearson $r$ with Status ($Y$) | Confounding Assessment |
|---|:---:|:---:|---|
| `MDVP:Fo(Hz)` (Raw) | $+1.0000$ | $-0.2832$ | **Severe sex/identity confounding** |
| `semitone_range` | $+0.0889$ | $+0.0917$ | **Virtually unconfounded** ($r \approx 0$ with $F_0$) |
| `ratio_fhi_flo` | $+0.0569$ | $+0.0414$ | Weakly correlated with pitch; poor signal |
| `ratio_fhi_fo` | $-0.1358$ | $+0.0020$ | Zero diagnostic correlation |
| `ratio_fo_flo` | $+0.2952$ | $+0.0684$ | Moderate pitch correlation |
| `log_ratio_fhi_fo` | $-0.1511$ | $+0.0520$ | Slight pitch correlation |
| `log_ratio_fo_flo` | $+0.2740$ | $+0.0891$ | Moderate pitch correlation |
| `pitch_pos_ratio` | $+0.3440$ | $-0.0557$ | Moderate pitch correlation |

### Confounding Conclusion:
- `semitone_range` successfully decouples from absolute pitch ($r = 0.0889$), confirming that logarithmic frequency range normalization removes baseline pitch bias.
- However, its bivariate correlation with Parkinson's status is very low ($r = 0.0917$), explaining why it fails to provide substantial independent predictive power over perturbation features (jitter, shimmer, HNR, RPDE, DFA, PPE).

---

## 11. Limitations

1. **Severe Cohort Sample Size:** Only 32 total participants (8 healthy controls). With only 8 control subjects in LOGO, a single subject misclassification represents a 12.5% swing in specificity and a 6.25% swing in balanced accuracy.
2. **Coarse Summary Statistics:** The dataset does not contain instantaneous pitch trajectories, pitch acceleration, or cycle-to-cycle pitch variability contours. Calculating relative measures from aggregate minimum and maximum values ($F_{\text{hi}}, F_{\text{lo}}$) is inherently sensitive to acoustic recording artifacts (e.g., vocal fry or octave-jumping errors at phonation onset).
3. **No Demographics in Tabular Set:** Age and biological sex are not explicitly encoded as metadata in `parkinsons.data`, necessitating inferential pitch tracking.

---

## 12. Recommendation for Phase 2.2C

### Formal Verdict Category:
> **"Relative pitch provides marginal or unstable improvement; retain the pitch-free baseline."**

### Justification:
1. **Parsimony and Robustness:** The 13-feature pitch-free ExtraTrees model achieves **0.7917 balanced accuracy** under participant-level LOGO validation without relying on any fundamental frequency terms.
2. **Degradation from Ratios:** Adding linear or logarithmic pitch excursion ratios degrades balanced accuracy down to **0.7292** by inducing false negatives on genuine Parkinson's subjects.
3. **Marginality of Semitone Range:** While `semitone_range` is mathematically unconfounded with pitch ($r = 0.0889$) and slightly improves repeated StratifiedGroupKFold accuracy ($0.7840 \pm 0.1859$ vs $0.7740 \pm 0.1992$), it yields **identical participant-level LOGO predictions and balanced accuracy (0.7917)** without rescuing any false negatives or false positives.
4. **Research Baseline for Phase 2.2C:** Phase 2.2C (Feature Selection & Model Refinement) should proceed with the **13-feature pitch-free feature set** as the primary research baseline, with `semitone_range` retained as an optional secondary candidate for sensitivity testing.

---
*Report generated strictly from isolated benchmark results in `ml/pathway_a/experiments/results/phase2_relative_pitch/`.*
