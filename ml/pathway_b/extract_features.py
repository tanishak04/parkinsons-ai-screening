"""
ml/pathway_b/extract_features.py
Acoustic feature extraction from sustained-vowel .wav recordings.

Usage:
    python ml/pathway_b/extract_features.py \
        --input  ml/pathway_b/data/raw/italian_pd \
        --output ml/pathway_b/data/processed/features.csv

The input directory must contain two sub-folders: pd/ and control/
Each .wav file = one recording from one speaker.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully for unit testing without audio libs
# ---------------------------------------------------------------------------
try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("[WARN] praat-parselmouth not found — Praat features will be NaN")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[WARN] librosa not found — MFCC/spectral features will be NaN")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_praat_features(wav_path: Path) -> dict[str, float]:
    """Extract F0, Jitter, Shimmer, HNR via Praat/parselmouth."""
    if not HAS_PARSELMOUTH:
        return {k: float("nan") for k in [
            "fo_mean", "fo_std", "fo_range",
            "jitter_local_pct", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp",
            "shimmer_local_pct", "shimmer_local_db",
            "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "shimmer_dda",
            "hnr_mean", "nhr",
        ]}

    snd = parselmouth.Sound(str(wav_path))
    # --- Pitch (F0) ---
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]  # voiced frames only

    fo_mean = float(np.mean(pitch_values)) if len(pitch_values) else 0.0
    fo_std  = float(np.std(pitch_values))  if len(pitch_values) else 0.0
    fo_range = float(np.ptp(pitch_values)) if len(pitch_values) else 0.0

    # --- PointProcess for jitter/shimmer ---
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    jitter_local  = call(point_process, "Get jitter (local)",    0, 0, 0.0001, 0.02, 1.3)
    jitter_abs    = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap    = call(point_process, "Get jitter (rap)",      0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5   = call(point_process, "Get jitter (ppq5)",     0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp    = call(point_process, "Get jitter (ddp)",      0, 0, 0.0001, 0.02, 1.3)

    shimmer_local    = call([snd, point_process], "Get shimmer (local)",       0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local_db = call([snd, point_process], "Get shimmer (local_dB)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3     = call([snd, point_process], "Get shimmer (apq3)",        0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5     = call([snd, point_process], "Get shimmer (apq5)",        0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11    = call([snd, point_process], "Get shimmer (apq11)",       0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda      = call([snd, point_process], "Get shimmer (dda)",         0, 0, 0.0001, 0.02, 1.3, 1.6)

    # --- HNR ---
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_mean = call(harmonicity, "Get mean", 0, 0)
    nhr = 1.0 / hnr_mean if hnr_mean and hnr_mean > 0 else float("nan")

    return {
        "fo_mean": fo_mean,
        "fo_std": fo_std,
        "fo_range": fo_range,
        "jitter_local_pct": jitter_local,
        "jitter_abs": jitter_abs,
        "jitter_rap": jitter_rap,
        "jitter_ppq5": jitter_ppq5,
        "jitter_ddp": jitter_ddp,
        "shimmer_local_pct": shimmer_local,
        "shimmer_local_db": shimmer_local_db,
        "shimmer_apq3": shimmer_apq3,
        "shimmer_apq5": shimmer_apq5,
        "shimmer_apq11": shimmer_apq11,
        "shimmer_dda": shimmer_dda,
        "hnr_mean": hnr_mean,
        "nhr": nhr,
    }


def extract_librosa_features(wav_path: Path, n_mfcc: int = 13) -> dict[str, float]:
    """Extract MFCCs and spectral features with librosa."""
    if not HAS_LIBROSA:
        feats: dict[str, float] = {}
        for i in range(1, n_mfcc + 1):
            feats[f"mfcc{i}_mean"] = float("nan")
            feats[f"mfcc{i}_std"] = float("nan")
        for k in ["spectral_centroid", "spectral_rolloff", "zcr"]:
            feats[k] = float("nan")
        return feats

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats: dict[str, float] = {}
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfccs[i]))

    feats["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["spectral_rolloff"]  = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats["zcr"]               = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    return feats


def extract_features(wav_path: Path) -> dict[str, float]:
    """Full feature extraction for one .wav file."""
    feats: dict[str, float] = {
        "filename": wav_path.name,
    }
    feats.update(extract_praat_features(wav_path))
    feats.update(extract_librosa_features(wav_path))
    return feats


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    output_csv: Path,
    pd_subdir: str = "pd",
    control_subdir: str = "control",
) -> pd.DataFrame:
    """
    Walk input_dir/pd/ and input_dir/control/, extract features, save CSV.

    Parameters
    ----------
    input_dir   : root dir containing pd/ and control/ subdirs
    output_csv  : where to write features.csv
    pd_subdir   : subfolder name for Parkinson's recordings
    control_subdir: subfolder name for healthy recordings
    """
    rows: list[dict] = []

    for label, subdir_name in [(1, pd_subdir), (0, control_subdir)]:
        subdir = input_dir / subdir_name
        wav_files = list(subdir.glob("*.wav")) + list(subdir.glob("*.WAV"))
        print(f"[{subdir_name}] Found {len(wav_files)} .wav files (label={label})")

        for wav in sorted(wav_files):
            try:
                feats = extract_features(wav)
                feats["status"] = label
                feats["speaker_id"] = wav.stem.split("_")[0]  # adjust if naming differs
                rows.append(feats)
                print(f"  ✓ {wav.name}")
            except Exception as exc:
                print(f"  ✗ {wav.name}: {exc}")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} records to {output_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract acoustic features from .wav files.")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent / "data" / "raw" / "italian_pd",
        help="Root dir with pd/ and control/ subdirs",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "data" / "processed" / "features.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print(f"Input dir : {args.input}")
    print(f"Output CSV: {args.output}\n")

    df = process_directory(args.input, args.output)
    print("\nFeature shape:", df.shape)
    print("Label distribution:\n", df["status"].value_counts().to_string())
    print("\nFirst row preview:")
    print(df.iloc[0].to_string())


if __name__ == "__main__":
    main()
