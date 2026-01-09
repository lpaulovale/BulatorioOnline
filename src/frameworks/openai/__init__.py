"""
OpenAI Framework - Complete Implementation.

Uses OpenAI API with Function Calling and JSON mode.
"""

from src.frameworks.openai.rag_implementation import (
    OpenAIAgent,
    get_openai_agent,
    reset_openai_agent
)
from src.frameworks.openai.judges import OpenAIJudgePipeline
from src.frameworks.openai.router import OpenAIRouter, get_openai_router

__all__ = [
    "OpenAIAgent",
    "get_openai_agent",
    "reset_openai_agent",
    "OpenAIJudgePipeline",
    "OpenAIRouter",
    "get_openai_router",
]
