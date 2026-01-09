"""
Global settings for Bulário RAG System.

Centralized configuration using Pydantic Settings.
Supports environment variables and .env file.
"""

from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Framework(str, Enum):
    """Available RAG framework implementations."""
    MCP = "mcp"
    LANGCHAIN = "langchain"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Usage:
        from config import settings
        print(settings.ACTIVE_FRAMEWORK)
    """
    
    # =========================================
    # Framework Selection
    # =========================================
    ACTIVE_FRAMEWORK: Framework = Field(
        default=Framework.OPENAI,
        description="Active RAG framework: mcp, langchain, openai, anthropic"
    )
    
    # =========================================
    # API Keys
    # =========================================
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key (for LangChain)"
    )
    
    # =========================================
    # Model Settings
    # =========================================
    GENERATION_MODEL: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model for response generation"
    )
    JUDGE_MODEL: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model for judge evaluations"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="Model for embeddings"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model for generation"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for LangChain"
    )
    
    # =========================================
    # Context Management
    # =========================================
    MAX_CONTEXT_MESSAGES: int = Field(
        default=10,
        description="Maximum messages in conversation history"
    )
    TOKEN_BUDGET: int = Field(
        default=100000,
        description="Token budget for context"
    )
    USE_PROMPT_CACHING: bool = Field(
        default=True,
        description="Enable Anthropic prompt caching"
    )
    
    # =========================================
    # Vector Database
    # =========================================
    VECTOR_DB_TYPE: str = Field(
        default="chroma",
        description="Vector DB type: chroma, pinecone, qdrant"
    )
    VECTOR_DB_PATH: str = Field(
        default="./data/chroma",
        description="Path to vector database"
    )
    
    # =========================================
    # Judge Configuration
    # =========================================
    SAFETY_WEIGHT: float = Field(default=0.40)
    QUALITY_WEIGHT: float = Field(default=0.30)
    SOURCE_WEIGHT: float = Field(default=0.20)
    FORMAT_WEIGHT: float = Field(default=0.10)
    
    ENABLE_JUDGE_PIPELINE: bool = Field(
        default=True,
        description="Enable judge pipeline for response evaluation"
    )
    MAX_REVISION_ATTEMPTS: int = Field(
        default=2,
        description="Maximum revision attempts on judge rejection"
    )
    
    # =========================================
    # Caching
    # =========================================
    ENABLE_REDIS_CACHE: bool = Field(default=False)
    REDIS_URL: Optional[str] = Field(default=None)
    
    # =========================================
    # Logging
    # =========================================
    LOG_LEVEL: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings
