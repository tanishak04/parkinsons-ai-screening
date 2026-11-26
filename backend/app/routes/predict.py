"""Prediction proxy endpoint and audio biomarker extraction."""

import logging
import math

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas.prediction import (
    AudioExtractionResponse,
    AudioPredictionResponse,
    AudioScreeningDisabledResponse,
    ExploratoryClassification,
    PredictionRequest,
    PredictionResponse,
)
from app.services.audio_extractor import (
    ACCEPTED_AUDIO_MIME_TYPES,
    AudioExtractionError,
    AudioFeatureExtractor,
)
from app.services.model_client import (
    InvalidModelResponse,
    ModelClient,
    ModelServerRejectedRequest,
    ModelServerUnavailable,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])

PITCH_FREE_13_FEATURES: list[str] = [
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
]


@router.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    summary="Request a research prediction from the ML server",
    description=(
        "Accepts the 22 precomputed acoustic features required by the current model server. "
        "This endpoint does not accept audio and is not a diagnostic service."
    ),
)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Validate and forward the feature vector, then normalize the model response."""
    client: ModelClient = request.app.state.model_client
    try:
        model_result = await client.predict(payload.to_model_payload())
    except ModelServerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ModelServerRejectedRequest as exc:
        # Do not forward arbitrary upstream error bodies through the public API.
        raise HTTPException(
            status_code=exc.status_code,
            detail="Model server rejected the request",
        ) from exc
    except InvalidModelResponse as exc:
        logger.exception("Invalid response from model server")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        normalized = PredictionResponse(
            prediction=model_result.get("prediction"),
            label=model_result.get("label"),
            confidence=model_result.get("confidence"),
            pd_probability=model_result.get("pd_probability"),
            healthy_probability=model_result.get("healthy_probability"),
            model_version=model_result.get("model_version"),
            feature_count=model_result.get("feature_count"),
            mock=model_result.get("_mock") is True,
        )
    except (ValidationError, TypeError) as exc:
        logger.exception("Model server response did not match prediction schema")
        raise HTTPException(status_code=502, detail="Model server returned an invalid prediction response") from exc
    return normalized


@router.post(
    "/predict/audio",
    response_model=AudioPredictionResponse,
    response_model_exclude_none=True,
    summary="Process voice recording for ET-13 research candidate screening",
    description=(
        "Extracts 16 acoustic vocal biomarkers from sustained vowel phonation recordings via "
        "Praat/Parselmouth, isolates the 13 pitch-free perturbation features, and requests inference "
        "from the ET-13 pitch-free research candidate. This is an investigational prototype and is not a diagnosis."
    ),
    responses={
        200: {"model": AudioPredictionResponse, "description": "ET-13 pitch-free candidate screening inference completed"},
        400: {"description": "Invalid audio file, corrupt container, or no sustained vocal phonation detected"},
        422: {"description": "Extracted acoustic features missing required ET-13 biomarkers or non-finite"},
        502: {"description": "Upstream model server rejected request or returned invalid response"},
        503: {"description": "Upstream model server unavailable or timed out"},
        500: {"description": "Feature extraction or internal processing failed"},
    },
)
async def predict_audio(request: Request, file: UploadFile = File(...)) -> AudioPredictionResponse:
    """Extract acoustic features from audio file and run ET-13 pitch-free research candidate inference."""
    # Basic content-type validation if provided
    if file.content_type:
        base_mime = file.content_type.split(";")[0].strip().lower()
        if base_mime not in ACCEPTED_AUDIO_MIME_TYPES and not base_mime.startswith("audio/"):
            if base_mime != "application/octet-stream":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported audio media type '{file.content_type}'. Supported: {ACCEPTED_AUDIO_MIME_TYPES}",
                )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio payload received.",
        )

    try:
        extraction_result = AudioFeatureExtractor.extract_features(
            audio_bytes=contents,
            filename=file.filename or "recording.wav",
        )
    except AudioExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during audio feature extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {exc}",
        ) from exc

    # Isolate the exact 13 pitch-free features
    extracted_features = extraction_result.get("extracted_features", {})
    missing_et13 = [f for f in PITCH_FREE_13_FEATURES if f not in extracted_features]
    if missing_et13:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Extracted audio features missing required ET-13 biomarkers: {missing_et13}",
        )

    et13_payload: dict[str, float] = {}
    for feat in PITCH_FREE_13_FEATURES:
        val = extracted_features[feat]
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Extracted feature '{feat}' is not a valid number: {val!r}",
            )
        if not math.isfinite(num_val):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Extracted feature '{feat}' has non-finite value: {val!r}",
            )
        et13_payload[feat] = num_val

    # Request inference from the model server's ET-13 endpoint
    client: ModelClient = request.app.state.model_client
    try:
        model_result = await client.predict_et13(et13_payload)
    except ModelServerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ModelServerRejectedRequest as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail="Model server rejected the request",
        ) from exc
    except InvalidModelResponse as exc:
        logger.exception("Invalid response from model server")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        exploratory_data = model_result.get("exploratory_classification")
        exploratory_obj = (
            ExploratoryClassification(**exploratory_data)
            if isinstance(exploratory_data, dict)
            else None
        )
        return AudioPredictionResponse(
            prediction=model_result["prediction"],
            label=model_result["label"],
            confidence=model_result["confidence"],
            pd_probability=model_result["pd_probability"],
            healthy_probability=model_result["healthy_probability"],
            model_version=model_result["model_version"],
            model_designation=model_result.get("model_designation", "ET-13 Pitch-Free Research Candidate"),
            feature_count=model_result.get("feature_count", 13),
            reference_threshold=model_result.get("reference_threshold", 0.50),
            exploratory_classification=exploratory_obj,
            duration_seconds=extraction_result.get("duration_seconds"),
            sampling_frequency=extraction_result.get("sampling_frequency"),
            extracted_features=et13_payload,
            disclaimer=model_result.get(
                "disclaimer",
                "Research prototype only. Investigational machine-learning screening based on "
                "13 pitch-free vocal perturbation biomarkers. Not a clinical diagnosis.",
            ),
        )
    except (ValidationError, KeyError, TypeError) as exc:
        logger.exception("Model server ET-13 response did not match schema: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Model server returned an invalid ET-13 prediction response",
        ) from exc




