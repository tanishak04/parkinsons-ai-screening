"""Environment-backed application settings."""

from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_server_url: AnyHttpUrl = "http://localhost:8001"
    model_server_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    frontend_origins: str = "http://localhost:5173,http://localhost:3000"
    log_level: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]

    @property
    def model_base_url(self) -> str:
        return str(self.model_server_url).rstrip("/")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for the current process."""
    return Settings()

