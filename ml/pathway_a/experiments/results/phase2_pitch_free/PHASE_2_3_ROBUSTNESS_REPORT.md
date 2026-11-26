# Phase 2.3 — Robustness Evaluation & Final Candidate Selection
## Pathway A: UCI Parkinson's Voice Dataset

**Document Version:** 1.0.0  
**Date:** 2026-11-25  
**Candidate Architecture:** `ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight='balanced', random_state=42)`  
**Feature Space:** 13 Pitch-Free Vocal Perturbation Features (Excludes $F_0, F_{\text{hi}}, F_{\text{lo}}$)  
**Status:** Final Candidate Evaluation Complete  

---

### 1. Robustness & Stability Summary

The candidate model was rigorously evaluated across 32-Fold Leave-One-Group-Out (LOGO) and 5-Repeat 5-Fold StratifiedGroupKFold (25 total split evaluations across seeds `[42, 123, 456, 789, 2024]`):

| Metric | Participant Level (LOGO) | Participant Level (SGKF 5-Seed Mean $\pm$ Std) | Recording Level (LOGO) | Recording Level (SGKF 5-Seed Mean $\pm$ Std) |
|---|:---:|:---:|:---:|:---:|
| **Balanced Accuracy** | **0.7917** | **$0.7792 \pm 0.0114$** | **0.7260** | **$0.7231 \pm 0.0069$** |
| **Sensitivity (Recall)** | **0.7083** (17/24) | **$0.6833 \pm 0.0228$** | **0.6395** (94/147) | **$0.6504 \pm 0.0108$** |
| **Specificity** | **0.8750** (7/8) | **$0.8750 \pm 0.0000$** | **0.8125** (39/48) | **$0.7958 \pm 0.0175$** |
| **ROC-AUC** | **0.7865** | **$0.7990 \pm 0.0155$** | **0.7483** | **$0.7634 \pm 0.0084$** |
| **PR-AUC** | **0.9228** | **$0.9301 \pm 0.0079$** | **0.9097** | **$0.9160 \pm 0.0048$** |
| **F1-Score** | **0.8095** | **$0.7921 \pm 0.0159$** | **0.7520** | **$0.7555 \pm 0.0063$** |
| **Brier Score** | **0.1863** | **$0.1850 \pm 0.0038$** | **0.1941** | **$0.1918 \pm 0.0035$** |

---

### 2. Fold-Wise Performance & Stability Findings

1. **Seed Invariance & Extreme Specificity Stability:**
   - Across all 5 independent SGKF seeds, participant specificity is **100% constant at 0.8750** ($\text{std} = 0.0000$).
   - Participant balanced accuracy exhibits an exceptionally narrow dispersion ($\text{std} = \pm 0.0114$, ranging between $0.7708$ and $0.7917$).
2. **High Individual Prediction Consistency:**
   - **29 out of 32 participants (90.6%) receive 100% identical binary classifications** across all 6 evaluation settings (LOGO and all 5 SGKF seeds).
   - Only 3 borderline PD subjects (`S02`, `S06`, `S19`, with mean predicted probabilities between $0.47$ and $0.56$) shift across the $0.50$ decision threshold depending on fold composition.
3. **Participant Driver Analysis:**
   - **Healthy Controls (8 subjects):** 7 out of 8 controls (`S07`, `S10`, `S13`, `S17`, `S42`, `S43`, `S49`) are consistently identified as Healthy across every split ($P \in [0.291, 0.431]$).
   - **Outlier Subject `S50`:** Represents the single persistent false positive ($P = 0.748$) across all baseline and candidate architectures. Because $N_{\text{HC}} = 8$, this single subject sets the empirical specificity ceiling at $7/8 = 87.50\%$.
   - **Persistent False Negatives (6 subjects):** `S04`, `S08`, `S25`, `S27`, `S32`, `S44` are mild or atypical PD cases with near-normal vocal stability (high HNR, low jitter). Sustained phonation alone lacks motor activation to detect their impairment.

---

### 3. Weaknesses & Limitations

1. **Sample Cohort Size:** 32 participants total (only 8 healthy controls). While participant-level grouping was strictly enforced, future external validation on an independent cohort is essential.
2. **Atypical Phonation Blindspot:** Mild PD cases without overt laryngeal tremor or dysphonia (such as `S08` and `S25`) cannot be differentiated from healthy voices using sustained vowels alone.
3. **Threshold Boundary Sensitivity:** Sensitivity varies slightly ($66.7\%$ to $70.8\%$) because 3 borderline subjects sit near $\tau = 0.50$.

---

### 4. Final Recommendation

### **Verdict:**
> **RETAIN the 13-feature pitch-free ExtraTrees model as the final research champion candidate for Pathway A.**

### **Justification:**
- Outperforms the legacy 22-feature Stage 1 baseline in balanced accuracy ($0.7917$ vs $0.6667$) and specificity ($0.8750$ vs $0.5000$).
- Successfully eliminates demographic and vocal pitch confounding (healthy males `S13`, `S43`, `S49` are correctly classified as healthy).
- Demonstrates near-perfect stability across seeds ($\pm 0.0114$ Balanced Accuracy std) and 90.6% participant prediction consistency.
- Parsimonious tree architecture (`max_depth=3`, `min_samples_leaf=3`) prevents overfitting the 32-subject cohort.
