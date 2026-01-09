"""
Framework factory for creating RAG agents.

Usage:
    from src.frameworks.factory import get_rag
    
    rag = get_rag()  # Uses ACTIVE_FRAMEWORK from settings
    response = await rag.query("Para que serve paracetamol?")
"""

import logging
from typing import Optional, Any

from config.settings import Framework, settings

logger = logging.getLogger(__name__)


def get_rag(framework: Optional[Framework] = None) -> Any:
    """
    Get RAG agent instance based on framework selection.
    
    Args:
        framework: Framework to use. If None, uses settings.ACTIVE_FRAMEWORK
    
    Returns:
        Framework-specific agent instance (MCPAgent, LangChainAgent, or OpenAIAgent)
    
    Raises:
        ValueError: If framework is not supported or API key is missing
    """
    framework = framework or settings.ACTIVE_FRAMEWORK
    
    logger.info(f"Getting RAG agent for framework: {framework.value}")
    
    if framework == Framework.MCP:
        from src.frameworks.mcp.rag_implementation import get_mcp_agent
        
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured for MCP framework")
        
        return get_mcp_agent()
    
    elif framework == Framework.LANGCHAIN:
        from src.frameworks.langchain.rag_implementation import get_langchain_agent
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured for LangChain framework")
        
        return get_langchain_agent()
    
    elif framework == Framework.OPENAI:
        from src.frameworks.openai.rag_implementation import get_openai_agent
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured for OpenAI framework")
        
        return get_openai_agent()
    
    elif framework == Framework.ANTHROPIC:
        # Anthropic uses the same as MCP
        from src.frameworks.mcp.rag_implementation import get_mcp_agent
        
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured for Anthropic framework")
        
        return get_mcp_agent()
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def create_rag_instance(framework: Optional[Framework] = None) -> Any:
    """Alias for get_rag for backward compatibility."""
    return get_rag(framework)


def get_available_frameworks() -> list:
    """
    Get list of available frameworks based on configured API keys.
    
    Returns:
        List of Framework enums that have required API keys configured
    """
    available = []
    
    if settings.ANTHROPIC_API_KEY:
        available.append(Framework.MCP)
        available.append(Framework.ANTHROPIC)
    
    if settings.GEMINI_API_KEY:
        available.append(Framework.LANGCHAIN)
    
    if settings.OPENAI_API_KEY:
        available.append(Framework.OPENAI)
    
    return available


def get_current_framework() -> Framework:
    """Get currently active framework."""
    return settings.ACTIVE_FRAMEWORK
