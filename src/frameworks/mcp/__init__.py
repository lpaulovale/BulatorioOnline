"""
MCP Framework - Complete Implementation.

Uses Anthropic Claude with tool calling.
"""

from src.frameworks.mcp.rag_implementation import (
    MCPAgent,
    get_mcp_agent,
    reset_mcp_agent
)
from src.frameworks.mcp.judges import MCPJudgePipeline
from src.frameworks.mcp.router import MCPRouter, get_mcp_router

__all__ = [
    "MCPAgent",
    "get_mcp_agent",
    "reset_mcp_agent",
    "MCPJudgePipeline",
    "MCPRouter",
    "get_mcp_router",
]
