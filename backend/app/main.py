"""FastAPI application factory and application entry point."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes import health, predict
from app.services.model_client import ModelClient


def create_app() -> FastAPI:
    """Create the API app with configured routes, CORS, and shared model client."""
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient() as http_client:
            application.state.model_client = ModelClient(settings, client=http_client)
            yield

    application = FastAPI(
        title="Parkinson's AI Screening Research API",
        description=(
            "A research prototype API that validates precomputed voice features and proxies "
            "them to an existing ML server. Outputs are experimental and are not diagnoses."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.include_router(health.router)
    application.include_router(predict.router)
    return application


app = create_app()

