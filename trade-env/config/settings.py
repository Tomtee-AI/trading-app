"""
config/settings.py

Application settings using Pydantic Settings (v2).
Loads from .env file with sensible defaults for local development.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3.1")

    # Data
    FRED_API_KEY: str | None = None
    DATABASE_URL: str = Field(default="sqlite:///zero_human_company.db")

    # Logging & Runtime
    LOG_LEVEL: str = Field(default="INFO")
    ENVIRONMENT: str = Field(default="development")

    # Governance thresholds
    MIN_CONFIDENCE_MARKET_ANALYST: float = 0.65
    MIN_CONFIDENCE_TRADE_ANALYST: float = 0.55
    MIN_CONFIDENCE_PORTFOLIO: float = 0.70


settings = Settings()


# Quick access
def get_settings() -> Settings:
    return settings