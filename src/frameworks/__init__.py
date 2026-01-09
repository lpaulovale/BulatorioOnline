"""
Frameworks package.

Contains RAG implementations:
- MCP (Anthropic Claude)
- LangChain (Gemini)
- OpenAI
"""

from src.frameworks.factory import create_rag_instance, get_rag, get_available_frameworks

__all__ = [
    "create_rag_instance",
    "get_rag",
    "get_available_frameworks",
]
