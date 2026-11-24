"""Prediction proxy endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_client import (
    InvalidModelResponse,
    ModelClient,
    ModelServerRejectedRequest,
    ModelServerUnavailable,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["prediction"])


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

