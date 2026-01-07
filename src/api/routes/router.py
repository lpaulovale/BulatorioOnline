"""
Router API Routes for PharmaBula

Provides endpoints for direct router access, debugging, and tool inspection.
"""

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/router", tags=["Router"])


class RouterAnalyzeRequest(BaseModel):
    """Request model for router analysis."""
    
    message: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User's question or request"
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        default="medium",
        description="Request priority level"
    )
    preferences: dict = Field(
        default_factory=dict,
        description="User preferences for tool selection"
    )


class RouterAnalyzeResponse(BaseModel):
    """Response model for router analysis."""
    
    reasoning: str = Field(description="Router's reasoning process")
    selected_tools: list[dict] = Field(description="Tools selected for execution")
    execution_plan: list[str] = Field(description="Step-by-step execution plan")
    confidence: str = Field(description="Confidence level (high/medium/low)")
    estimated_cost: str = Field(description="Estimated cost tier")
    estimated_time: str = Field(description="Estimated execution time")
    fallback_plan: str = Field(description="Fallback if primary plan fails")


class ToolInfo(BaseModel):
    """Information about a single tool."""
    
    id: str
    description: str
    capabilities: list[str]
    cost_tier: str
    latency: str


@router.post("/analyze", response_model=RouterAnalyzeResponse)
async def analyze_request(request: RouterAnalyzeRequest) -> RouterAnalyzeResponse:
    """
    Analyze a request and return routing decision WITHOUT executing.
    
    Useful for debugging and understanding how the router makes decisions.
    
    - **message**: The user's question or request
    - **priority**: Priority level (affects tool selection)
    - **preferences**: User preferences (speed, cost sensitivity, etc.)
    """
    try:
        from src.llm.router import RouterAgent, Priority, UserPreferences, CostTier
        from src.llm.router.pharmabula_registry import get_cached_pharmabula_registry
        
        # Map priority
        priority_map = {
            "low": Priority.LOW,
            "medium": Priority.MEDIUM,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT
        }
        priority = priority_map.get(request.priority, Priority.MEDIUM)
        
        # Build preferences
        prefs = request.preferences
        user_prefs = UserPreferences(
            prefer_speed=prefs.get("prefer_speed", False),
            cost_sensitivity=CostTier(prefs.get("cost_sensitivity", "medium"))
        )
        
        # Create router with PharmaBula registry
        router_agent = RouterAgent(
            tool_registry=get_cached_pharmabula_registry()
        )
        
        # Get routing decision
        decision = router_agent.route_request(
            message=request.message,
            priority=priority,
            preferences=user_prefs
        )
        
        return RouterAnalyzeResponse(
            reasoning=decision.reasoning,
            selected_tools=[
                {
                    "tool_id": t.tool_id,
                    "reason": t.reason,
                    "order": t.order,
                    "inputs": t.inputs
                }
                for t in decision.selected_tools
            ],
            execution_plan=decision.execution_plan,
            confidence=decision.confidence.value,
            estimated_cost=decision.estimated_cost.value,
            estimated_time=decision.estimated_time.value,
            fallback_plan=decision.fallback_plan
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao analisar requisição: {str(e)}"
        )


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools() -> list[ToolInfo]:
    """
    List all available tools in the PharmaBula registry.
    
    Returns tool IDs, descriptions, capabilities, and cost/latency info.
    """
    from src.llm.router.pharmabula_registry import get_cached_pharmabula_registry
    
    registry = get_cached_pharmabula_registry()
    
    return [
        ToolInfo(
            id=tool.id,
            description=tool.description,
            capabilities=tool.capabilities,
            cost_tier=tool.cost_tier.value,
            latency=tool.latency.value
        )
        for tool in registry.tools
    ]


@router.get("/tools/{tool_id}")
async def get_tool_details(tool_id: str):
    """
    Get detailed information about a specific tool.
    """
    from src.llm.router.pharmabula_registry import get_cached_pharmabula_registry
    
    registry = get_cached_pharmabula_registry()
    tool = registry.get_tool(tool_id)
    
    if not tool:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_id}' não encontrada"
        )
    
    return {
        "id": tool.id,
        "description": tool.description,
        "capabilities": tool.capabilities,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
        "cost_tier": tool.cost_tier.value,
        "latency": tool.latency.value,
        "requirements": tool.requirements,
        "examples": tool.examples
    }


@router.get("/health")
async def router_health():
    """Health check for the router service."""
    try:
        from src.llm.router.pharmabula_registry import get_cached_pharmabula_registry
        
        registry = get_cached_pharmabula_registry()
        
        return {
            "status": "healthy",
            "service": "router",
            "tools_count": len(registry.tools),
            "apis_count": len(registry.apis)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
