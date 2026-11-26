"""
backend/app/services/audio_extractor.py
Acoustic feature extractor using Praat / Parselmouth.

Extracts the 16 standard voice perturbation biomarkers from live audio
and audits the availability of the 6 non-linear dynamical features
required by the frozen UCI model.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

logger = logging.getLogger(__name__)

# The 22 features expected by the frozen Pathway A model
REQUIRED_FEATURES: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# The 16 features extractable via standard acoustic analysis (Praat / Parselmouth)
EXTRACTABLE_PRAAT_FEATURES: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
]

# The 6 non-linear dynamical features developed by Max Little (Little et al., 2007)
NONLINEAR_UNAVAILABLE_FEATURES: list[str] = [
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

ACCEPTED_AUDIO_MIME_TYPES: list[str] = [
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp4",
]


class AudioExtractionError(Exception):
    """Raised when audio decoding or acoustic analysis fails."""


class AudioFeatureExtractor:
    """Service for extracting clinical vocal biomarkers from audio recordings."""

    @staticmethod
    def load_sound(audio_bytes: bytes, filename: str = "recording.wav") -> parselmouth.Sound:
        """Decode raw audio bytes into a Parselmouth Sound object."""
        if not audio_bytes:
            raise AudioExtractionError("Empty audio payload received.")

        # Try in-memory decoding via soundfile
        try:
            buffer = io.BytesIO(audio_bytes)
            data, sr = sf.read(buffer)
            # If multi-channel, convert to mono by averaging channels
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return parselmouth.Sound(data.astype(np.float64), sampling_frequency=float(sr))
        except Exception:
            # Fallback to temporary file decoding (for WebM/OGG containers)
            pass

        try:
            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            return parselmouth.Sound(tmp_path)
        except Exception as exc:
            logger.exception("Failed to decode audio file into Parselmouth Sound")
            raise AudioExtractionError(f"Audio decoding failed: {exc}") from exc

    @classmethod
    def extract_features(
        cls,
        audio_bytes: bytes,
        filename: str = "recording.wav",
        min_duration_sec: float = 0.8,
    ) -> dict[str, Any]:
        """
        Extract acoustic features from voice recording.

        Returns a dictionary containing:
          - extracted_features: dict mapping 16 feature names to float values
          - extracted_count: int (16)
          - missing_features: list of 6 non-linear features
          - is_fully_compatible: bool (False)
          - duration_seconds: float
          - blocker_reason: detailed diagnostic explanation
        """
        sound = cls.load_sound(audio_bytes, filename)
        duration = float(sound.duration)
        if duration < min_duration_sec:
            raise AudioExtractionError(
                f"Recording duration ({duration:.2f}s) is too short. "
                f"Minimum required is {min_duration_sec:.1f}s of sustained vowel phonation."
            )

        # 1. Pitch / Fundamental Frequency
        pitch = sound.to_pitch()
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        f0_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        f0_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

        if np.isnan(f0_mean) or f0_mean <= 0:
            raise AudioExtractionError(
                "No periodic pitch detected. Please ensure you record a clear sustained vowel sound."
            )

        # 2. Point Process for Jitter & Shimmer Perturbation Analysis
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

        # 3. Shimmer Measures
        shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # 4. Harmonics-to-Noise Ratio (HNR) and Noise-to-Harmonics Ratio (NHR)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        if np.isnan(hnr):
            hnr = 0.0
        # NHR approximation from HNR in dB: 10^(-HNR/10)
        nhr = float(10 ** (-hnr / 10.0)) if hnr > 0 else 1.0

        extracted_features = {
            "MDVP:Fo(Hz)": round(float(f0_mean), 3),
            "MDVP:Fhi(Hz)": round(float(f0_max), 3),
            "MDVP:Flo(Hz)": round(float(f0_min), 3),
            "MDVP:Jitter(%)": round(float(jitter_local), 6),
            "MDVP:Jitter(Abs)": round(float(jitter_abs), 6),
            "MDVP:RAP": round(float(jitter_rap), 6),
            "MDVP:PPQ": round(float(jitter_ppq5), 6),
            "Jitter:DDP": round(float(jitter_ddp), 6),
            "MDVP:Shimmer": round(float(shimmer_local), 6),
            "MDVP:Shimmer(dB)": round(float(shimmer_db), 4),
            "Shimmer:APQ3": round(float(shimmer_apq3), 6),
            "Shimmer:APQ5": round(float(shimmer_apq5), 6),
            "MDVP:APQ": round(float(shimmer_apq11), 6),
            "Shimmer:DDA": round(float(shimmer_dda), 6),
            "NHR": round(float(nhr), 6),
            "HNR": round(float(hnr), 3),
        }

        blocker_reason = (
            "Standard acoustic analysis (Praat/Parselmouth) successfully extracts the 16 linear voice perturbation "
            "biomarkers. However, the frozen UCI model strictly requires 22 features, including 6 non-linear dynamical "
            "features (spread1, spread2, PPE, RPDE, DFA, D2) developed by Max Little. These non-linear measures require "
            "specialized phase-space delay reconstruction and semitone entropy algorithms that cannot be reliably "
            "reproduced from live browser microphone audio. Because spread1, PPE, and spread2 represent 3 of the top 4 "
            "most influential features in the champion model, live inference is blocked to prevent clinical distortion."
        )

        return {
            "status": "partial_extraction",
            "duration_seconds": round(duration, 2),
            "sampling_frequency": sound.sampling_frequency,
            "extracted_feature_count": len(extracted_features),
            "required_feature_count": len(REQUIRED_FEATURES),
            "extracted_features": extracted_features,
            "missing_features": NONLINEAR_UNAVAILABLE_FEATURES,
            "is_fully_compatible": False,
            "blocker_code": "NONLINEAR_FEATURES_UNAVAILABLE",
            "blocker_reason": blocker_reason,
        }
