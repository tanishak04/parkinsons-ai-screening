"""
ml/pathway_b/explain_shap.py
SHAP explainability for the best Stage 2 model.

Usage:
    python ml/pathway_b/explain_shap.py \
        --model    ml/pathway_b/models/best_pipeline_stage2.joblib \
        --features ml/pathway_b/data/processed/features.csv \
        --output   ml/pathway_b/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[ERROR] shap not installed. Run: pip install shap")
    raise SystemExit(1)

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH   = BASE_DIR / "models" / "best_pipeline_stage2.joblib"
DEFAULT_FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
DEFAULT_PLOTS_DIR    = BASE_DIR / "plots"
META_COLS = {"filename", "status", "speaker_id"}


def load_data(artifact: dict, features_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(features_csv).dropna(subset=["status"])
    y = df["status"].astype(int)
    feature_names: list[str] = artifact["feature_names"]
    X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y


def get_base_estimator(pipeline):
    """Extract the raw classifier from a Pipeline."""
    return pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]


def build_explainer(pipeline, X_background: pd.DataFrame):
    """Return the most appropriate SHAP explainer for the pipeline's classifier."""
    # Apply all transformers except the final step
    steps = list(pipeline.steps)
    if len(steps) > 1:
        from sklearn.pipeline import Pipeline as SKPipeline
        transformer = SKPipeline(steps[:-1])
        X_transformed = transformer.transform(X_background)
    else:
        X_transformed = X_background.values

    clf = get_base_estimator(pipeline)
    clf_name = type(clf).__name__

    if "XGB" in clf_name or "Forest" in clf_name or "Tree" in clf_name:
        explainer = shap.TreeExplainer(clf)
    elif "Linear" in clf_name or "Logistic" in clf_name or "SVC" in clf_name:
        # Use KernelExplainer with a small background sample for SVC
        background = shap.sample(X_transformed, min(50, len(X_transformed)))
        explainer = shap.KernelExplainer(
            lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=X_background.columns))[:, 1],
            background,
        )
        return explainer, None  # No transform needed; predict_proba handles it

    return explainer, X_transformed


def run_shap_analysis(
    model_path: Path,
    features_csv: Path,
    output_dir: Path,
    max_waterfall_samples: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    model_name = artifact["metadata"]["algorithm"]
    print(f"Model: {model_name}")

    print(f"Loading features: {features_csv}")
    X, y = load_data(artifact, features_csv)
    feature_names: list[str] = artifact["feature_names"]
    print(f"Records: {len(X)} | Features: {X.shape[1]}\n")

    # --- Build explainer ---
    result = build_explainer(pipeline, X)
    if isinstance(result, tuple):
        explainer, X_transformed = result
    else:
        explainer = result
        X_transformed = None

    # Decide what to pass to shap_values
    if X_transformed is not None:
        shap_input = X_transformed
    else:
        shap_input = X

    print("Computing SHAP values (this may take a moment) ...")
    shap_values = explainer.shap_values(shap_input)

    # For binary classifiers that return a list (class 0, class 1), take class 1
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # --- Save SHAP values CSV ---
    shap_df = pd.DataFrame(sv, columns=feature_names)
    shap_df["true_label"] = y.values
    shap_csv = output_dir / "shap_values.csv"
    shap_df.to_csv(shap_csv, index=False)
    print(f"✓ Saved SHAP values to {shap_csv}")

    # --- Summary plot ---
    if HAS_MPL:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X, feature_names=feature_names, show=False)
        summary_path = output_dir / "shap_summary_plot.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150)
        plt.close()
        print(f"✓ Saved summary plot to {summary_path}")

        # Bar plot (mean |SHAP|)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X, feature_names=feature_names, plot_type="bar", show=False)
        bar_path = output_dir / "shap_bar_plot.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=150)
        plt.close()
        print(f"✓ Saved bar plot to {bar_path}")

        # Waterfall plots for top PD-positive predictions
        pd_indices = np.where(y.values == 1)[0][:max_waterfall_samples]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        for idx in pd_indices:
            expl = shap.Explanation(
                values=sv[idx],
                base_values=float(base_value),
                data=X.iloc[idx].values,
                feature_names=feature_names,
            )
            plt.figure()
            shap.waterfall_plot(expl, show=False)
            wf_path = output_dir / f"shap_waterfall_sample_{idx}.png"
            plt.tight_layout()
            plt.savefig(wf_path, dpi=150)
            plt.close()
        print(f"✓ Saved {len(pd_indices)} waterfall plots to {output_dir}")

    # --- Top features report ---
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    report = {
        "model": model_name,
        "top_features_by_mean_abs_shap": [
            {"feature": f, "mean_abs_shap": round(float(v), 6)}
            for f, v in top_features
        ],
    }
    report_path = output_dir / "shap_feature_importance.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved feature importance to {report_path}")

    print("\n--- Top 10 Features by SHAP Importance ---")
    for f, v in top_features[:10]:
        print(f"  {f:<30} {v:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability for Stage 2 model.")
    parser.add_argument("--model",    type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--output",   type=Path, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()
    run_shap_analysis(args.model, args.features, args.output)


if __name__ == "__main__":
    main()
