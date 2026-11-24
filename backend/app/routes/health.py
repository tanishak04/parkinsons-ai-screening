"""Backend health and optional upstream readiness endpoint."""

from fastapi import APIRouter, Request

from app.schemas.prediction import HealthResponse
from app.services.model_client import ModelClient, ModelServerUnavailable

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Check backend and model server health")
async def health(request: Request) -> HealthResponse:
    """Return backend liveness and best-effort model server connectivity."""
    client: ModelClient = request.app.state.model_client
    try:
        reachable, detail = await client.health()
    except ModelServerUnavailable as exc:
        reachable, detail = False, str(exc)
    return HealthResponse(
        status="ok",
        service="parkinsons-screening-backend",
        model_server="reachable" if reachable else "unreachable",
        model_server_detail=detail,
    )

