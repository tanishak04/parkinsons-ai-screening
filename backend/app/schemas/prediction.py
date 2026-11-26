"""Schemas for feature-based prediction requests and model responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """The 22 numerical acoustic features expected by the existing ML server."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mdvp_fo_hz: float = Field(alias="MDVP:Fo(Hz)")
    mdvp_fhi_hz: float = Field(alias="MDVP:Fhi(Hz)")
    mdvp_flo_hz: float = Field(alias="MDVP:Flo(Hz)")
    mdvp_jitter_pct: float = Field(alias="MDVP:Jitter(%)")
    mdvp_jitter_abs: float = Field(alias="MDVP:Jitter(Abs)")
    mdvp_rap: float = Field(alias="MDVP:RAP")
    mdvp_ppq: float = Field(alias="MDVP:PPQ")
    jitter_ddp: float = Field(alias="Jitter:DDP")
    mdvp_shimmer: float = Field(alias="MDVP:Shimmer")
    mdvp_shimmer_db: float = Field(alias="MDVP:Shimmer(dB)")
    shimmer_apq3: float = Field(alias="Shimmer:APQ3")
    shimmer_apq5: float = Field(alias="Shimmer:APQ5")
    mdvp_apq: float = Field(alias="MDVP:APQ")
    shimmer_dda: float = Field(alias="Shimmer:DDA")
    nhr: float = Field(alias="NHR")
    hnr: float = Field(alias="HNR")
    rpde: float = Field(alias="RPDE")
    dfa: float = Field(alias="DFA")
    spread1: float
    spread2: float
    d2: float = Field(alias="D2")
    ppe: float = Field(alias="PPE")

    def to_model_payload(self) -> dict[str, float]:
        """Serialize using the exact field names expected by the Flask model server."""
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    """Normalized response returned by the backend to its frontend client."""

    prediction: Literal[0, 1]
    label: Literal["PARKINSONS", "HEALTHY"]
    confidence: float = Field(ge=0, le=1)
    pd_probability: float = Field(ge=0, le=1)
    healthy_probability: float = Field(ge=0, le=1)
    model_version: str
    feature_count: int | None = None
    mock: bool = False
    disclaimer: str = "Research prototype only; this output is not a diagnosis."


class AudioCapabilities(BaseModel):
    """Audio processing and model compatibility declaration."""

    audio_feature_extraction: bool = True
    audio_input_compatible: bool = False
    accepted_audio_formats: list[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/webm",
        "audio/webm;codecs=opus",
        "audio/ogg",
        "audio/ogg;codecs=opus",
        "audio/mp4",
        "audio/mpeg",
    ]
    supported_features_count: int = 16
    required_features_count: int = 22
    incompatibility_reason: str | None = (
        "Live audio screening is disabled because the experimental 16-feature acoustic model "
        "did not satisfy clinical validation criteria (balanced accuracy 0.6667 < target 0.70 "
        "under participant-level LOGO cross-validation). Automated screening is unavailable."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_server: Literal["reachable", "unreachable", "unknown"]
    model_server_detail: str | None = None
    capabilities: AudioCapabilities = Field(default_factory=AudioCapabilities)


class AudioScreeningDisabledResponse(BaseModel):
    """Explicit response returned when audio is processed while screening prediction is disabled."""

    status: Literal["screening_disabled", "validation_failed"] = "screening_disabled"
    screening_available: Literal[False] = False
    message: str = (
        "Automated audio screening is temporarily disabled. The experimental 16-feature acoustic "
        "model did not satisfy project validation criteria (balanced accuracy 0.6667 < 0.70 target). "
        "Acoustic feature extraction is preserved for research purposes only."
    )
    validation_summary: str = (
        "Evaluated under participant-level Leave-One-Group-Out (LOGO) cross-validation; achieved "
        "balanced accuracy 0.6667 with participant instability. Automated predictions are disabled."
    )
    duration_seconds: float
    sampling_frequency: float
    extracted_feature_count: int
    extracted_features: dict[str, float]
    missing_features: list[str] = [
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    disclaimer: str = (
        "Research status notification only. This system does not provide medical screening, "
        "clinical diagnosis, or health assessments."
    )


class AudioExtractionResponse(BaseModel):
    """Response returned when audio is processed for feature extraction."""

    status: Literal["partial_extraction", "incompatible"]
    duration_seconds: float
    sampling_frequency: float
    extracted_feature_count: int
    required_feature_count: int
    extracted_features: dict[str, float]
    missing_features: list[str]
    is_fully_compatible: bool
    blocker_code: str
    blocker_reason: str


class ExploratoryClassification(BaseModel):
    """Exploratory high-sensitivity screening operating point at threshold 0.45."""

    threshold: float = 0.45
    prediction: Literal[0, 1]
    label: Literal["PARKINSONS", "HEALTHY"]
    note: str = "Exploratory high-sensitivity screening operating point; non-diagnostic."


class AudioPredictionResponse(BaseModel):
    """Inference response from the ET-13 pitch-free research candidate via audio analysis."""

    prediction: Literal[0, 1]
    label: Literal["PARKINSONS", "HEALTHY"]
    confidence: float = Field(ge=0, le=1)
    pd_probability: float = Field(ge=0, le=1)
    healthy_probability: float = Field(ge=0, le=1)
    model_version: str
    model_designation: str = "ET-13 Pitch-Free Research Candidate"
    feature_count: int = 13
    reference_threshold: float = 0.50
    exploratory_classification: ExploratoryClassification | None = None
    duration_seconds: float | None = None
    sampling_frequency: float | None = None
    extracted_features: dict[str, float] | None = None
    disclaimer: str = (
        "Research prototype only. Investigational machine-learning screening based on "
        "13 pitch-free vocal perturbation biomarkers. Not a clinical diagnosis."
    )


