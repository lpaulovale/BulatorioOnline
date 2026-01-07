"""
Pydantic schemas for the Router Agent system.

Contains all data models used by the router for:
- Tool and API definitions
- User context and preferences
- Routing decisions and execution results
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# ENUMS
# ============================================================================

class Priority(str, Enum):
    """Request priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class CostTier(str, Enum):
    """Tool/operation cost classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Latency(str, Enum):
    """Expected latency classification"""
    FAST = "fast"
    MODERATE = "moderate"
    SLOW = "slow"


class Confidence(str, Enum):
    """Router confidence in its decision"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# TOOL & API DEFINITIONS
# ============================================================================

class Tool(BaseModel):
    """Definition of a tool that can be used by the router"""
    id: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What this tool does")
    capabilities: list[str] = Field(..., description="List of capabilities")
    input_schema: dict = Field(..., description="Expected input parameters")
    output_schema: dict = Field(..., description="Expected output format")
    cost_tier: CostTier = Field(..., description="Resource cost classification")
    latency: Latency = Field(..., description="Expected response time")
    requirements: list[str] = Field(
        default_factory=list, 
        description="Prerequisites for using this tool"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Example use cases for the tool"
    )


class API(BaseModel):
    """Definition of an external API endpoint"""
    name: str = Field(..., description="API identifier")
    endpoint: str = Field(..., description="Base URL")
    methods: list[str] = Field(..., description="Supported HTTP methods")
    capabilities: list[str] = Field(..., description="What this API can do")
    rate_limits: dict = Field(
        default_factory=dict,
        description="Rate limiting configuration"
    )
    auth_required: bool = Field(
        default=False,
        description="Whether authentication is required"
    )


class ToolRegistry(BaseModel):
    """Registry containing all available tools and APIs"""
    tools: list[Tool] = Field(..., description="List of available tools")
    apis: list[API] = Field(
        default_factory=list,
        description="List of available external APIs"
    )
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by its ID"""
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        return None
    
    def list_tool_ids(self) -> list[str]:
        """Get all tool IDs"""
        return [tool.id for tool in self.tools]


# ============================================================================
# USER CONTEXT
# ============================================================================

class UserPreferences(BaseModel):
    """User preferences for tool selection"""
    prefer_speed: bool = Field(
        default=False,
        description="Prioritize faster tools over accuracy"
    )
    cost_sensitivity: CostTier = Field(
        default=CostTier.MEDIUM,
        description="How sensitive to cost considerations"
    )
    preferred_tools: list[str] = Field(
        default_factory=list,
        description="Tools to prefer when multiple options exist"
    )
    avoided_tools: list[str] = Field(
        default_factory=list,
        description="Tools to avoid if alternatives exist"
    )
    custom: dict = Field(
        default_factory=dict,
        description="Custom preference key-value pairs"
    )


class UserContext(BaseModel):
    """Full context for a user request"""
    message: str = Field(..., description="The user's request")
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Request priority level"
    )
    preferences: UserPreferences = Field(
        default_factory=UserPreferences,
        description="User's tool preferences"
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Previous conversation messages"
    )
    additional_context: dict = Field(
        default_factory=dict,
        description="Any extra context information"
    )


# ============================================================================
# ROUTING DECISION
# ============================================================================

class SelectedTool(BaseModel):
    """A tool selected for execution in the plan"""
    tool_id: str = Field(..., description="ID of the selected tool")
    reason: str = Field(..., description="Why this tool was chosen")
    order: int = Field(..., description="Execution order (1-based)")
    inputs: dict = Field(
        default_factory=dict,
        description="Input parameters for the tool"
    )


class RoutingDecision(BaseModel):
    """Router agent's complete decision output"""
    reasoning: str = Field(..., description="Detailed thought process")
    selected_tools: list[SelectedTool] = Field(
        ...,
        description="Tools selected for execution"
    )
    execution_plan: list[str] = Field(
        ...,
        description="Step-by-step execution plan"
    )
    trade_offs: str = Field(
        ...,
        description="Trade-offs considered in decision"
    )
    confidence: Confidence = Field(
        ...,
        description="Confidence level in the decision"
    )
    fallback_plan: str = Field(
        ...,
        description="Alternative if primary plan fails"
    )
    estimated_cost: CostTier = Field(
        ...,
        description="Overall estimated cost"
    )
    estimated_time: Latency = Field(
        ...,
        description="Overall estimated time"
    )


# ============================================================================
# EXECUTION RESULTS
# ============================================================================

class ToolExecutionResult(BaseModel):
    """Result from executing a single tool"""
    tool_id: str = Field(..., description="ID of the executed tool")
    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(default=None, description="Tool output if successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Execution time in milliseconds"
    )


class ExecutionResult(BaseModel):
    """Complete result from executing a routing plan"""
    decision: RoutingDecision = Field(
        ...,
        description="The original routing decision"
    )
    tool_results: list[ToolExecutionResult] = Field(
        ...,
        description="Results from each tool execution"
    )
    final_response: Optional[str] = Field(
        default=None,
        description="Aggregated final response"
    )
    success: bool = Field(..., description="Overall execution success")
    total_time_ms: Optional[float] = Field(
        default=None,
        description="Total execution time in milliseconds"
    )
