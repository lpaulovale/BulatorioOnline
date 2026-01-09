"""
MCP Router Agent.

Uses MCP protocol with Anthropic Claude for tool routing.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Tool definition for MCP."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class MCPRoutingDecision:
    """Routing decision from MCP router."""
    selected_tools: List[str]
    reasoning: str
    confidence: float
    execution_plan: List[str]


@dataclass
class MCPExecutionResult:
    """Result of tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class MCPRouter:
    """
    Router using MCP with Anthropic Claude.
    
    Features:
    - Claude for intelligent tool selection
    - MCP-style tool definitions
    - Prompt caching for efficiency
    """
    
    PHARMABULA_TOOLS = [
        MCPTool(
            name="search_drugs",
            description="Search for drug information in the bulletin database",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        ),
        MCPTool(
            name="get_drug_details",
            description="Get detailed information about a specific drug",
            input_schema={
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"}
                },
                "required": ["drug_name"]
            }
        ),
        MCPTool(
            name="check_interactions",
            description="Check for drug-drug interactions",
            input_schema={
                "type": "object",
                "properties": {
                    "drugs": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["drugs"]
            }
        ),
        MCPTool(
            name="generate_response",
            description="Generate a response using RAG",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["query"]
            }
        )
    ]
    
    def __init__(self):
        """Initialize MCP router."""
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.GENERATION_MODEL
        self.tools = self.PHARMABULA_TOOLS
        
        logger.info("MCP Router initialized")
    
    def _build_tools_prompt(self) -> str:
        """Build tools description for prompt."""
        tools_desc = []
        for tool in self.tools:
            props = tool.input_schema.get("properties", {})
            params = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in props.items()])
            tools_desc.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(tools_desc)
    
    def route_request(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> MCPRoutingDecision:
        """
        Route a request to appropriate tools.
        
        Args:
            message: User request
            context: Additional context
        
        Returns:
            MCPRoutingDecision with selected tools
        """
        prompt = f"""You are a tool router for PharmaBula, a drug information system.

Available tools:
{self._build_tools_prompt()}

User request: {message}

Analyze the request and return a JSON with:
{{
    "selected_tools": ["tool1", "tool2"],
    "reasoning": "why these tools",
    "confidence": 0.0-1.0,
    "execution_plan": ["step1", "step2"]
}}

Return ONLY valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        
        # Parse JSON
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text.strip())
            
            return MCPRoutingDecision(
                selected_tools=data.get("selected_tools", []),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                execution_plan=data.get("execution_plan", [])
            )
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return MCPRoutingDecision(
                selected_tools=["generate_response"],
                reasoning="Fallback to default tool",
                confidence=0.5,
                execution_plan=["Generate response"]
            )
    
    async def execute_tools(
        self,
        decision: MCPRoutingDecision,
        executors: Dict[str, Any]
    ) -> List[MCPExecutionResult]:
        """
        Execute selected tools.
        
        Args:
            decision: Routing decision
            executors: Dict of tool_name -> executor function
        
        Returns:
            List of execution results
        """
        import asyncio
        
        results = []
        
        for tool_name in decision.selected_tools:
            executor = executors.get(tool_name)
            
            if not executor:
                results.append(MCPExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"No executor for {tool_name}"
                ))
                continue
            
            try:
                if asyncio.iscoroutinefunction(executor):
                    result = await executor()
                else:
                    result = executor()
                
                results.append(MCPExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=result
                ))
            except Exception as e:
                results.append(MCPExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e)
                ))
        
        return results


# Singleton
_mcp_router: Optional[MCPRouter] = None


def get_mcp_router() -> MCPRouter:
    global _mcp_router
    if _mcp_router is None:
        _mcp_router = MCPRouter()
    return _mcp_router
