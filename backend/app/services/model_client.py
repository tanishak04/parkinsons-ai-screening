"""HTTPX client for the existing Flask ML server."""

import logging
from typing import Any

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ModelServerError(Exception):
    """Base error for failures communicating with the model server."""


class ModelServerUnavailable(ModelServerError):
    """The model server could not be reached or did not respond in time."""


class InvalidModelResponse(ModelServerError):
    """The model server returned a response outside the expected contract."""


class ModelServerRejectedRequest(ModelServerError):
    """The model server rejected a request that passed backend validation."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelClient:
    """Small reusable client that proxies requests to the Flask service."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        timeout = httpx.Timeout(self.settings.model_server_timeout_seconds)
        try:
            if self._client is not None:
                return await self._client.request(method, path, timeout=timeout, **kwargs)
            async with httpx.AsyncClient(base_url=self.settings.model_base_url, timeout=timeout) as client:
                return await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            logger.warning("Model server request timed out")
            raise ModelServerUnavailable("Model server request timed out") from exc
        except httpx.RequestError as exc:
            logger.warning("Model server request failed (%s)", type(exc).__name__)
            raise ModelServerUnavailable("Model server is unavailable") from exc

    async def predict(self, payload: dict[str, float]) -> dict[str, Any]:
        response = await self._request("POST", f"{self.settings.model_base_url}/predict", json=payload)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            logger.warning("Model server rejected prediction request (%s)", response.status_code)
            raise ModelServerRejectedRequest(
                f"Model server rejected the request: {detail}", status_code=502
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidModelResponse("Model server returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise InvalidModelResponse("Model server response must be a JSON object")
        return body

    async def health(self) -> tuple[bool, str | None]:
        response = await self._request("GET", f"{self.settings.model_base_url}/health")
        if response.status_code >= 400:
            return False, f"Model server returned HTTP {response.status_code}"
        try:
            body = response.json()
        except ValueError:
            return False, "Model server returned invalid health JSON"
        if not isinstance(body, dict) or body.get("status") != "ok":
            return False, "Model server health response was unexpected"
        return True, None

