"""
Gemini Router Agent System
==========================
MPC-style router that analyzes requests and routes to appropriate tools.

Components:
- RouterAgent: Core routing and execution logic
- Schemas: Pydantic models for tools, decisions, contexts
- PharmaBula: Domain-specific tools for medication assistance
"""

from .schemas import (
    Priority,
    CostTier,
    Latency,
    Confidence,
    Tool,
    API,
    ToolRegistry,
    UserPreferences,
    UserContext,
    SelectedTool,
    RoutingDecision,
    ExecutionResult,
)
from .router_agent import RouterAgent, get_router_agent, reset_router_agent

# PharmaBula-specific
from .pharmabula_registry import (
    get_pharmabula_tool_registry,
    get_cached_pharmabula_registry,
)
from .pharmabula_executors import get_pharmabula_executors, PHARMABULA_EXECUTORS

__all__ = [
    # Core
    "Priority",
    "CostTier",
    "Latency",
    "Confidence",
    "Tool",
    "API",
    "ToolRegistry",
    "UserPreferences",
    "UserContext",
    "SelectedTool",
    "RoutingDecision",
    "ExecutionResult",
    "RouterAgent",
    "get_router_agent",
    "reset_router_agent",
    # PharmaBula
    "get_pharmabula_tool_registry",
    "get_cached_pharmabula_registry",
    "get_pharmabula_executors",
    "PHARMABULA_EXECUTORS",
]

