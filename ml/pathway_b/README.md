# Pathway B — Acoustic Feature Extraction & Model Comparison

This module (Stage 2) handles:
1. **Feature extraction** from raw `.wav` recordings using PRAAT and librosa
2. **5-fold cross-validated model comparison** (LR, RF, XGBoost, SVM)
3. **SHAP explainability** for the best model

## Directory Structure

```
pathway_b/
├── data/
│   ├── raw/
│   │   └── italian_pd/
│   │       ├── pd/          ← PD speaker .wav files (gitignored)
│   │       └── control/     ← Healthy control .wav files (gitignored)
│   ├── processed/
│   │   └── features.csv     ← Extracted acoustic features (gitignored)
│   └── features/            ← Reserved for feature-engineering outputs
├── models/
│   ├── best_pipeline_stage2.joblib   ← Best trained model (gitignored)
│   └── comparison_results.json       ← CV results for all models
├── plots/
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── shap_waterfall_sample_*.png
│   └── shap_feature_importance.json
├── extract_features.py    ← Step 1: .wav → features.csv
├── train_compare.py       ← Step 2: 5-fold CV comparison
├── explain_shap.py        ← Step 3: SHAP analysis
└── README.md
```

## Usage

```bash
# Step 1: Extract features from .wav files
python ml/pathway_b/extract_features.py \
    --input  ml/pathway_b/data/raw/italian_pd \
    --output ml/pathway_b/data/processed/features.csv

# Step 2: Train & compare models
python ml/pathway_b/train_compare.py \
    --features ml/pathway_b/data/processed/features.csv \
    --output   ml/pathway_b/models/

# Step 3: SHAP explainability
python ml/pathway_b/explain_shap.py \
    --model    ml/pathway_b/models/best_pipeline_stage2.joblib \
    --features ml/pathway_b/data/processed/features.csv \
    --output   ml/pathway_b/plots/
```

## Dependencies

```bash
pip install praat-parselmouth librosa xgboost shap soundfile antropy
```
