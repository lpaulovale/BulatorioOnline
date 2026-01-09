"""
Router API Routes for PharmaBula

Provides endpoints for router access using framework-specific implementations.
"""

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings, Framework

router = APIRouter(prefix="/api/router", tags=["Router"])


class RouterAnalyzeRequest(BaseModel):
    """Request model for router analysis."""
    
    message: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User's question or request"
    )
    framework: Literal["mcp", "langchain", "openai"] = Field(
        default="openai",
        description="Framework to use for routing"
    )


class RouterAnalyzeResponse(BaseModel):
    """Response model for router analysis."""
    
    reasoning: str = Field(description="Router's reasoning process")
    selected_tools: list[dict] = Field(description="Tools selected for execution")
    confidence: float = Field(description="Confidence level (0-1)")
    framework: str = Field(description="Framework used")


class ToolInfo(BaseModel):
    """Information about a single tool."""
    
    name: str
    description: str


@router.post("/analyze", response_model=RouterAnalyzeResponse)
async def analyze_request(request: RouterAnalyzeRequest) -> RouterAnalyzeResponse:
    """
    Analyze a request and return routing decision using selected framework.
    
    - **message**: The user's question or request
    - **framework**: Which framework to use (mcp, langchain, openai)
    """
    try:
        framework = request.framework
        
        if framework == "mcp":
            from src.frameworks.mcp.router import get_mcp_router
            
            router_agent = get_mcp_router()
            decision = router_agent.route_request(request.message)
            
            return RouterAnalyzeResponse(
                reasoning=decision.reasoning,
                selected_tools=[{"name": t} for t in decision.selected_tools],
                confidence=decision.confidence,
                framework="mcp"
            )
        
        elif framework == "langchain":
            from src.frameworks.langchain.router import get_langchain_router
            
            router_agent = get_langchain_router()
            decision = router_agent.route_request(request.message)
            
            return RouterAnalyzeResponse(
                reasoning=decision.reasoning,
                selected_tools=[{"name": t} for t in decision.selected_tools],
                confidence=decision.confidence,
                framework="langchain"
            )
        
        elif framework == "openai":
            from src.frameworks.openai.router import get_openai_router
            
            router_agent = get_openai_router()
            decision = await router_agent.route_request(request.message)
            
            return RouterAnalyzeResponse(
                reasoning=decision.reasoning,
                selected_tools=decision.selected_tools,
                confidence=decision.confidence,
                framework="openai"
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown framework: {framework}")
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao analisar requisição: {str(e)}"
        )


@router.get("/tools")
async def list_tools(framework: str = "openai") -> list[ToolInfo]:
    """
    List available tools for the specified framework.
    """
    tools = []
    
    if framework == "mcp":
        from src.frameworks.mcp.router import MCPRouter
        for tool in MCPRouter.PHARMABULA_TOOLS:
            tools.append(ToolInfo(name=tool.name, description=tool.description))
    
    elif framework == "langchain":
        tools = [
            ToolInfo(name="search_drugs", description="Search drug information"),
            ToolInfo(name="get_drug_details", description="Get detailed drug info"),
            ToolInfo(name="check_interactions", description="Check drug interactions")
        ]
    
    elif framework == "openai":
        from src.frameworks.openai.router import PHARMABULA_FUNCTIONS
        for func in PHARMABULA_FUNCTIONS:
            f = func["function"]
            tools.append(ToolInfo(name=f["name"], description=f["description"]))
    
    return tools


@router.get("/health")
async def router_health():
    """Health check for the router service."""
    return {
        "status": "healthy",
        "service": "router",
        "available_frameworks": ["mcp", "langchain", "openai"]
    }
