"""
OpenAI Router Agent.

Uses OpenAI Function Calling for tool routing.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


# OpenAI Function definitions
PHARMABULA_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_drugs",
            "description": "Search for drug information in the bulletin database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for drugs"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drug_details",
            "description": "Get detailed information about a specific drug",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Name of the drug"
                    }
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_interactions",
            "description": "Check for drug-drug interactions between multiple medications",
            "parameters": {
                "type": "object",
                "properties": {
                    "drugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of drug names to check"
                    }
                },
                "required": ["drugs"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_response",
            "description": "Generate a response using the RAG system",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


@dataclass
class OpenAIRoutingDecision:
    """Routing decision from OpenAI router."""
    selected_tools: List[Dict[str, Any]]  # [{"name": "tool", "arguments": {...}}]
    reasoning: str
    confidence: float


@dataclass
class OpenAIExecutionResult:
    """Result of function execution."""
    function_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class OpenAIRouter:
    """
    Router using OpenAI Function Calling.
    
    Features:
    - Native function calling
    - Parallel function execution
    - Automatic tool selection
    """
    
    def __init__(self):
        """Initialize OpenAI router."""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.tools = PHARMABULA_FUNCTIONS
        
        logger.info("OpenAI Router initialized")
    
    async def route_request(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> OpenAIRoutingDecision:
        """
        Route request using OpenAI function calling.
        
        Args:
            message: User request
            context: Additional context
        
        Returns:
            OpenAIRoutingDecision with selected functions
        """
        messages = [
            {
                "role": "system",
                "content": "You are PharmaBula, a drug information assistant. "
                           "Analyze the user's request and call the appropriate functions."
            },
            {"role": "user", "content": message}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        choice = response.choices[0]
        
        # Extract tool calls
        selected_tools = []
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                selected_tools.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
        
        return OpenAIRoutingDecision(
            selected_tools=selected_tools,
            reasoning=choice.message.content or "Function calls selected",
            confidence=0.9 if selected_tools else 0.5
        )
    
    async def execute_functions(
        self,
        decision: OpenAIRoutingDecision,
        executors: Dict[str, Any]
    ) -> List[OpenAIExecutionResult]:
        """
        Execute selected functions.
        
        Args:
            decision: Routing decision
            executors: Dict of function_name -> executor
        
        Returns:
            List of execution results
        """
        import asyncio
        
        results = []
        
        for tool_call in decision.selected_tools:
            func_name = tool_call["name"]
            arguments = tool_call["arguments"]
            
            executor = executors.get(func_name)
            
            if not executor:
                results.append(OpenAIExecutionResult(
                    function_name=func_name,
                    success=False,
                    error=f"No executor for {func_name}"
                ))
                continue
            
            try:
                if asyncio.iscoroutinefunction(executor):
                    result = await executor(**arguments)
                else:
                    result = executor(**arguments)
                
                results.append(OpenAIExecutionResult(
                    function_name=func_name,
                    success=True,
                    result=result
                ))
            except Exception as e:
                results.append(OpenAIExecutionResult(
                    function_name=func_name,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    async def route_and_execute(
        self,
        message: str,
        executors: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Route and execute in one call with automatic tool result handling.
        
        Uses OpenAI's multi-turn function calling pattern.
        """
        messages = [
            {
                "role": "system",
                "content": "You are PharmaBula, a drug information assistant."
            },
            {"role": "user", "content": message}
        ]
        
        # First call to get function selections
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        choice = response.choices[0]
        
        # If no tool calls, return direct response
        if not choice.message.tool_calls:
            return {
                "response": choice.message.content,
                "tools_used": []
            }
        
        # Execute tool calls
        messages.append(choice.message)
        tools_used = []
        
        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            executor = executors.get(func_name)
            
            if executor:
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(executor):
                        result = await executor(**arguments)
                    else:
                        result = executor(**arguments)
                    result_str = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
            else:
                result_str = json.dumps({"error": f"No executor for {func_name}"})
            
            tools_used.append(func_name)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })
        
        # Second call to get final response
        final_response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return {
            "response": final_response.choices[0].message.content,
            "tools_used": tools_used
        }


# Singleton
_openai_router: Optional[OpenAIRouter] = None


def get_openai_router() -> OpenAIRouter:
    global _openai_router
    if _openai_router is None:
        _openai_router = OpenAIRouter()
    return _openai_router
