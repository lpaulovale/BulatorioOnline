"""
Router Agent - Core implementation with tool execution.

This module provides the main RouterAgent class that:
1. Analyzes user requests using Gemini
2. Selects optimal tools from the registry
3. Executes the selected tools in order
4. Aggregates results into a final response
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Optional, Union

import google.generativeai as genai

from .prompts import ROUTER_SYSTEM_PROMPT, build_router_prompt
from .schemas import (
    Confidence,
    CostTier,
    ExecutionResult,
    Latency,
    Priority,
    RoutingDecision,
    SelectedTool,
    Tool,
    ToolExecutionResult,
    ToolRegistry,
    UserContext,
    UserPreferences,
)
from .pharmabula_registry import get_cached_pharmabula_registry

logger = logging.getLogger(__name__)

# Type alias for tool executor functions
ToolExecutor = Callable[..., Any]
AsyncToolExecutor = Callable[..., Any]  # Coroutine returning Any


class RouterAgent:
    """
    Router Agent that analyzes requests and routes to appropriate tools.
    
    Features:
    - Request analysis and routing decisions via Gemini
    - Tool execution with sync and async support
    - Result aggregation with optional LLM summarization
    
    Usage:
        >>> router = RouterAgent(tool_registry=my_tools)
        >>> decision = router.route_request("Analyze this data")
        >>> result = await router.execute_plan(decision, tool_executors)
    """
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None
    ):
        """
        Initialize the router agent.
        
        Args:
            tool_registry: Custom tool registry (uses PharmaBula default if None)
            system_prompt: Custom system prompt (uses default if None)
            model_name: Gemini model to use
            api_key: Gemini API key (uses env var if None)
        """
        # Configure Gemini
        api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in environment or pass to constructor."
            )
        
        genai.configure(api_key=api_key)
        
        # Set configuration - use PharmaBula registry as default
        self.tool_registry = tool_registry or get_cached_pharmabula_registry()
        self.system_prompt = system_prompt or ROUTER_SYSTEM_PROMPT
        self.model_name = model_name
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )
        
        self.chat = None
        
        logger.info(f"RouterAgent initialized with model: {model_name}")
    
    def start_session(self):
        """Start a new chat session for context continuity."""
        self.chat = self.model.start_chat(history=[])
        logger.debug("Started new chat session")
    
    def _build_prompt(self, context: UserContext) -> str:
        """Build the complete prompt for routing."""
        registry_json = self.tool_registry.model_dump_json(indent=2)
        
        preferences_json = json.dumps({
            "priority": context.priority.value,
            "preferences": context.preferences.model_dump()
        }, indent=2)
        
        history_json = json.dumps(
            context.conversation_history[-5:], 
            indent=2
        ) if context.conversation_history else ""
        
        return build_router_prompt(
            tool_registry_json=registry_json,
            user_context_json=preferences_json,
            conversation_history_json=history_json,
            user_message=context.message
        )
    
    def _parse_response(self, response_text: str) -> RoutingDecision:
        """Parse Gemini response into RoutingDecision."""
        try:
            # Handle markdown code blocks
            text = response_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            # Parse JSON
            data = json.loads(text.strip())
            
            # Convert selected_tools to SelectedTool objects
            selected_tools = []
            for tool in data.get("selected_tools", []):
                if isinstance(tool, dict):
                    selected_tools.append(SelectedTool(**tool))
                else:
                    selected_tools.append(tool)
            
            data["selected_tools"] = selected_tools
            
            # Convert enums
            if isinstance(data.get("confidence"), str):
                data["confidence"] = Confidence(data["confidence"])
            if isinstance(data.get("estimated_cost"), str):
                data["estimated_cost"] = CostTier(data["estimated_cost"])
            if isinstance(data.get("estimated_time"), str):
                data["estimated_time"] = Latency(data["estimated_time"])
            
            return RoutingDecision(**data)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            raise ValueError(f"Failed to create RoutingDecision: {e}")
    
    def route_request(
        self,
        message: str,
        priority: Priority = Priority.MEDIUM,
        preferences: Optional[UserPreferences] = None,
        conversation_history: Optional[list[dict]] = None,
        additional_context: Optional[dict] = None
    ) -> RoutingDecision:
        """
        Route a user request to appropriate tools.
        
        This method analyzes the request using Gemini and returns
        a routing decision with selected tools and execution plan.
        
        Args:
            message: User's request
            priority: Request priority level
            preferences: User preferences for tool selection
            conversation_history: Previous conversation messages
            additional_context: Any additional context
            
        Returns:
            RoutingDecision with selected tools and execution plan
        """
        if self.chat is None:
            self.start_session()
        
        # Build context
        context = UserContext(
            message=message,
            priority=priority,
            preferences=preferences or UserPreferences(),
            conversation_history=conversation_history or [],
            additional_context=additional_context or {}
        )
        
        # Build and send prompt
        prompt = self._build_prompt(context)
        
        logger.debug(f"Routing request: {message[:100]}...")
        
        response = self.chat.send_message(prompt)
        
        # Parse and return decision
        decision = self._parse_response(response.text)
        
        logger.info(
            f"Routed to tools: {[t.tool_id for t in decision.selected_tools]}, "
            f"confidence: {decision.confidence.value}"
        )
        
        return decision
    
    async def execute_plan(
        self,
        decision: RoutingDecision,
        tool_executors: dict[str, Union[ToolExecutor, AsyncToolExecutor]],
        aggregate_results: bool = True
    ) -> ExecutionResult:
        """
        Execute the routing decision using provided tool executors.
        
        Args:
            decision: The routing decision from route_request()
            tool_executors: Dict mapping tool_id to executor function
            aggregate_results: Whether to aggregate results with LLM
            
        Returns:
            ExecutionResult with all tool outputs and final response
        """
        start_time = time.time()
        tool_results: list[ToolExecutionResult] = []
        all_success = True
        
        # Sort tools by execution order
        sorted_tools = sorted(decision.selected_tools, key=lambda t: t.order)
        
        # Execute each tool in order
        for selected_tool in sorted_tools:
            tool_id = selected_tool.tool_id
            inputs = selected_tool.inputs
            
            logger.debug(f"Executing tool: {tool_id}")
            
            # Get executor
            executor = tool_executors.get(tool_id)
            if not executor:
                logger.warning(f"No executor found for tool: {tool_id}")
                tool_results.append(ToolExecutionResult(
                    tool_id=tool_id,
                    success=False,
                    error=f"No executor registered for tool: {tool_id}"
                ))
                all_success = False
                continue
            
            # Execute
            tool_start = time.time()
            try:
                # Handle async executors
                if asyncio.iscoroutinefunction(executor):
                    result = await executor(**inputs)
                else:
                    result = executor(**inputs)
                
                execution_time = (time.time() - tool_start) * 1000
                
                tool_results.append(ToolExecutionResult(
                    tool_id=tool_id,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time
                ))
                
                logger.debug(f"Tool {tool_id} completed in {execution_time:.2f}ms")
                
            except Exception as e:
                execution_time = (time.time() - tool_start) * 1000
                logger.error(f"Tool {tool_id} failed: {e}")
                
                tool_results.append(ToolExecutionResult(
                    tool_id=tool_id,
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time
                ))
                all_success = False
        
        total_time = (time.time() - start_time) * 1000
        
        # Aggregate results if requested
        final_response = None
        if aggregate_results and tool_results:
            final_response = await self._aggregate_results(
                decision, tool_results
            )
        
        return ExecutionResult(
            decision=decision,
            tool_results=tool_results,
            final_response=final_response,
            success=all_success,
            total_time_ms=total_time
        )
    
    async def _aggregate_results(
        self,
        decision: RoutingDecision,
        tool_results: list[ToolExecutionResult]
    ) -> str:
        """
        Aggregate tool results into a coherent response.
        
        Uses the LLM to combine multiple tool outputs into
        a single, user-friendly response.
        """
        # Build results summary
        results_text = []
        for result in tool_results:
            if result.success:
                results_text.append(
                    f"Tool '{result.tool_id}': {json.dumps(result.result, default=str)}"
                )
            else:
                results_text.append(
                    f"Tool '{result.tool_id}' FAILED: {result.error}"
                )
        
        prompt = f"""Based on the following tool execution results, provide a clear, 
helpful response to the user's original request.

Original request context:
{decision.reasoning}

Execution plan:
{chr(10).join(decision.execution_plan)}

Tool results:
{chr(10).join(results_text)}

Provide a clear, concise response that:
1. Summarizes what was accomplished
2. Presents the key findings or results
3. Notes any issues or limitations encountered
4. Suggests next steps if applicable

Response:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to aggregate results: {e}")
            # Fallback to simple concatenation
            return "\n\n".join([
                f"**{r.tool_id}**: {r.result if r.success else r.error}"
                for r in tool_results
            ])
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool from the registry by ID."""
        return self.tool_registry.get_tool(tool_id)
    
    def list_tools(self) -> list[Tool]:
        """List all available tools."""
        return self.tool_registry.tools
    
    def list_tool_ids(self) -> list[str]:
        """List all available tool IDs."""
        return self.tool_registry.list_tool_ids()


# ============================================================================
# SINGLETON SUPPORT
# ============================================================================

_router_instance: Optional[RouterAgent] = None


def get_router_agent(
    tool_registry: Optional[ToolRegistry] = None,
    **kwargs
) -> RouterAgent:
    """
    Get or create the global RouterAgent instance.
    
    Args:
        tool_registry: Tool registry (only used on first call)
        **kwargs: Additional args for RouterAgent (only used on first call)
        
    Returns:
        Global RouterAgent instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = RouterAgent(tool_registry=tool_registry, **kwargs)
    return _router_instance


def reset_router_agent():
    """Reset the global RouterAgent instance (mainly for testing)."""
    global _router_instance
    _router_instance = None
