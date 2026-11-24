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


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    model_server: Literal["reachable", "unreachable", "unknown"]
    model_server_detail: str | None = None

