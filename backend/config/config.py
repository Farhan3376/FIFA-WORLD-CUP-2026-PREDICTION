"""Pydantic configuration settings for the backend application.

Loads environment variables with default values for local development.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings

# Root directory of the backend/project
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from env variables or defaults."""

    # Project metadata
    PROJECT_NAME: str = "FIFA World Cup 2026 Prediction Engine"
    API_V1_STR: str = "/api"
    
    # Environment
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

    # Database Settings
    # Use SQLite for easy local fallback if PostgreSQL environment variables aren't defined
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "fifa_worldcup")
    
    @property
    def DATABASE_URL(self) -> str:
        """Dynamically construct database URL from variables."""
        # Check if DATABASE_URL is directly supplied
        direct_url = os.getenv("DATABASE_URL")
        if direct_url:
            return direct_url
        
        # Fallback to SQLite if we are in development and postgres connection parameters are not fully supplied
        if self.ENV == "development" and not os.getenv("DB_HOST"):
            sqlite_path = BACKEND_ROOT / "fifa_worldcup.db"
            return f"sqlite:///{sqlite_path}"
            
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # JWT Authentication settings
    # In production, JWT_SECRET_KEY MUST be set via environment variable to a
    # strong, persistent value. When unset (local/dev use), a random key is
    # generated per process start -- this invalidates existing sessions on
    # restart but avoids ever committing a real secret to source control.
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # ML Model & Data paths
    MODEL_PATH: Path = PROJECT_ROOT / "models" / "trained" / "best_model.pkl"
    SELECTOR_PATH: Path = PROJECT_ROOT / "models" / "feature_selector.pkl"
    SCALER_PATH: Path = PROJECT_ROOT / "models" / "scaler.pkl"
    IMPUTER_PATH: Path = PROJECT_ROOT / "models" / "imputer.pkl"
    TEAM_DB_PATH: Path = PROJECT_ROOT / "models" / "team_database.csv"
    
    # Path to historical clean match data (used by validation/analytics)
    CLEAN_MATCHES_PATH: Path = PROJECT_ROOT / "data" / "interim" / "matches_clean.csv"
    ENGINEERED_MATCHES_PATH: Path = PROJECT_ROOT / "data" / "processed" / "matches_engineered.csv"

    # Third-party live fixtures/scores API (worldcup2026 by rezarahiminia, github.com/rezarahiminia/worldcup2026)
    # Public, unauthenticated instance. No official FIFA affiliation or SLA -- treat as best-effort.
    LIVE_FIXTURES_API_URL: str = os.getenv("LIVE_FIXTURES_API_URL", "https://worldcup26.ir")
    LIVE_FIXTURES_TIMEOUT_SECONDS: float = 8.0

    class Config:
        case_sensitive = True
        env_file = ".env"


# Instantiate settings singleton
settings = Settings()
