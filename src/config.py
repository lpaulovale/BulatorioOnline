"""
PharmaBula Configuration Module

Uses Pydantic Settings for type-safe configuration from environment variables.
All settings can be overridden via .env file or environment variables.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # =========================================
    # LLM Configuration
    # =========================================
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key"
    )
    gemini_model: str = Field(
        default="gemini-3-flash",
        description="Gemini model to use"
    )
    
    # =========================================
    # Database Configuration
    # =========================================
    chroma_persist_path: str = Field(
        default="./data/chroma",
        description="Path to ChromaDB persistence directory"
    )
    sqlite_database_path: str = Field(
        default="./data/cache.db",
        description="Path to SQLite cache database"
    )
    
    # =========================================
    # Scheduler Configuration
    # =========================================
    scraper_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours between bulletin update checks"
    )
    enable_scheduler: bool = Field(
        default=True,
        description="Enable/disable background scheduler"
    )
    
    # =========================================
    # API Configuration
    # =========================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API port"
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:8000"],
        description="Allowed CORS origins"
    )
    
    # =========================================
    # Application Mode
    # =========================================
    app_env: Literal["development", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()
