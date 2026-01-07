"""
Tests for the Router Agent system.

Run with: pytest tests/test_router.py -v
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm.router.schemas import (
    API,
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


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_tool_creation(self):
        """Test Tool model creation."""
        tool = Tool(
            id="test_tool",
            description="A test tool",
            capabilities=["test"],
            input_schema={"input": "string"},
            output_schema={"output": "string"},
            cost_tier=CostTier.LOW,
            latency=Latency.FAST
        )
        
        assert tool.id == "test_tool"
        assert tool.cost_tier == CostTier.LOW
        assert len(tool.requirements) == 0  # Default empty list
    
    def test_tool_registry_get_tool(self):
        """Test ToolRegistry.get_tool method."""
        registry = ToolRegistry(tools=[
            Tool(
                id="tool1",
                description="Tool 1",
                capabilities=["cap1"],
                input_schema={},
                output_schema={},
                cost_tier=CostTier.LOW,
                latency=Latency.FAST
            ),
            Tool(
                id="tool2",
                description="Tool 2",
                capabilities=["cap2"],
                input_schema={},
                output_schema={},
                cost_tier=CostTier.HIGH,
                latency=Latency.SLOW
            )
        ])
        
        tool1 = registry.get_tool("tool1")
        assert tool1 is not None
        assert tool1.id == "tool1"
        
        tool_none = registry.get_tool("nonexistent")
        assert tool_none is None
    
    def test_tool_registry_list_tool_ids(self):
        """Test ToolRegistry.list_tool_ids method."""
        registry = ToolRegistry(tools=[
            Tool(
                id="tool_a",
                description="A",
                capabilities=[],
                input_schema={},
                output_schema={},
                cost_tier=CostTier.LOW,
                latency=Latency.FAST
            ),
            Tool(
                id="tool_b",
                description="B",
                capabilities=[],
                input_schema={},
                output_schema={},
                cost_tier=CostTier.LOW,
                latency=Latency.FAST
            )
        ])
        
        ids = registry.list_tool_ids()
        assert ids == ["tool_a", "tool_b"]
    
    def test_user_preferences_defaults(self):
        """Test UserPreferences default values."""
        prefs = UserPreferences()
        
        assert prefs.prefer_speed is False
        assert prefs.cost_sensitivity == CostTier.MEDIUM
        assert prefs.preferred_tools == []
        assert prefs.avoided_tools == []
    
    def test_user_context_creation(self):
        """Test UserContext model creation."""
        context = UserContext(
            message="Test message",
            priority=Priority.HIGH,
            preferences=UserPreferences(prefer_speed=True)
        )
        
        assert context.message == "Test message"
        assert context.priority == Priority.HIGH
        assert context.preferences.prefer_speed is True
    
    def test_selected_tool_creation(self):
        """Test SelectedTool model creation."""
        tool = SelectedTool(
            tool_id="test_tool",
            reason="Best match for request",
            order=1,
            inputs={"param": "value"}
        )
        
        assert tool.tool_id == "test_tool"
        assert tool.order == 1
        assert tool.inputs["param"] == "value"
    
    def test_routing_decision_creation(self):
        """Test RoutingDecision model creation."""
        decision = RoutingDecision(
            reasoning="Analyzed request",
            selected_tools=[
                SelectedTool(tool_id="tool1", reason="test", order=1)
            ],
            execution_plan=["Step 1", "Step 2"],
            trade_offs="Cost vs speed",
            confidence=Confidence.HIGH,
            fallback_plan="Use alternative",
            estimated_cost=CostTier.LOW,
            estimated_time=Latency.FAST
        )
        
        assert decision.confidence == Confidence.HIGH
        assert len(decision.selected_tools) == 1
        assert len(decision.execution_plan) == 2
    
    def test_execution_result_creation(self):
        """Test ExecutionResult model creation."""
        decision = RoutingDecision(
            reasoning="Test",
            selected_tools=[],
            execution_plan=[],
            trade_offs="None",
            confidence=Confidence.MEDIUM,
            fallback_plan="None",
            estimated_cost=CostTier.LOW,
            estimated_time=Latency.FAST
        )
        
        result = ExecutionResult(
            decision=decision,
            tool_results=[
                ToolExecutionResult(
                    tool_id="tool1",
                    success=True,
                    result={"data": "test"}
                )
            ],
            success=True,
            total_time_ms=100.5
        )
        
        assert result.success is True
        assert result.total_time_ms == 100.5
        assert len(result.tool_results) == 1


class TestToolRegistry:
    """Tests for default tool registry."""
    
    def test_default_registry_loads(self):
        """Test that default registry loads without errors."""
        from src.llm.router.tool_registry import get_default_tool_registry
        
        registry = get_default_tool_registry()
        
        assert registry is not None
        assert len(registry.tools) > 0
    
    def test_default_registry_has_expected_tools(self):
        """Test default registry contains expected tools."""
        from src.llm.router.tool_registry import get_default_tool_registry
        
        registry = get_default_tool_registry()
        tool_ids = registry.list_tool_ids()
        
        # Check for some expected tools
        assert "web_search" in tool_ids
        assert "text_analyzer" in tool_ids
        assert "data_processor" in tool_ids
    
    def test_default_registry_tools_have_required_fields(self):
        """Test all tools have required fields."""
        from src.llm.router.tool_registry import get_default_tool_registry
        
        registry = get_default_tool_registry()
        
        for tool in registry.tools:
            assert tool.id, "Tool ID should not be empty"
            assert tool.description, "Description should not be empty"
            assert tool.cost_tier in CostTier
            assert tool.latency in Latency


class TestRouterAgent:
    """Tests for RouterAgent class."""
    
    @pytest.fixture
    def mock_genai(self):
        """Mock google.generativeai module."""
        with patch('src.llm.router.router_agent.genai') as mock:
            # Mock the model
            mock_model = MagicMock()
            mock_model.generate_content.return_value.text = json.dumps({
                "reasoning": "Test reasoning",
                "selected_tools": [
                    {"tool_id": "web_search", "reason": "test", "order": 1}
                ],
                "execution_plan": ["Step 1"],
                "trade_offs": "None",
                "confidence": "high",
                "fallback_plan": "None",
                "estimated_cost": "low",
                "estimated_time": "fast"
            })
            
            mock_chat = MagicMock()
            mock_chat.send_message.return_value.text = mock_model.generate_content.return_value.text
            mock_model.start_chat.return_value = mock_chat
            
            mock.GenerativeModel.return_value = mock_model
            
            yield mock
    
    def test_router_initialization(self, mock_genai):
        """Test RouterAgent initializes correctly."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            
            assert router.tool_registry is not None
            assert router.model is not None
    
    def test_router_requires_api_key(self):
        """Test RouterAgent raises error without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('src.llm.router.router_agent.genai'):
                from src.llm.router.router_agent import RouterAgent
                
                with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                    RouterAgent()
    
    def test_route_request_returns_decision(self, mock_genai):
        """Test route_request returns RoutingDecision."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            decision = router.route_request("Test message")
            
            assert isinstance(decision, RoutingDecision)
            assert decision.confidence == Confidence.HIGH
            assert len(decision.selected_tools) == 1
    
    def test_route_request_with_preferences(self, mock_genai):
        """Test route_request accepts preferences."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            decision = router.route_request(
                message="Test",
                priority=Priority.HIGH,
                preferences=UserPreferences(prefer_speed=True)
            )
            
            assert isinstance(decision, RoutingDecision)
    
    @pytest.mark.asyncio
    async def test_execute_plan_success(self, mock_genai):
        """Test execute_plan executes tools correctly."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            decision = router.route_request("Test")
            
            # Define a simple executor
            def mock_executor(**kwargs):
                return {"result": "success"}
            
            result = await router.execute_plan(
                decision=decision,
                tool_executors={"web_search": mock_executor},
                aggregate_results=False
            )
            
            assert result.success is True
            assert len(result.tool_results) == 1
            assert result.tool_results[0].success is True
    
    @pytest.mark.asyncio
    async def test_execute_plan_missing_executor(self, mock_genai):
        """Test execute_plan handles missing executor."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            decision = router.route_request("Test")
            
            # No executors provided
            result = await router.execute_plan(
                decision=decision,
                tool_executors={},
                aggregate_results=False
            )
            
            assert result.success is False
            assert result.tool_results[0].error is not None
    
    @pytest.mark.asyncio
    async def test_execute_plan_async_executor(self, mock_genai):
        """Test execute_plan handles async executors."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            decision = router.route_request("Test")
            
            # Async executor
            async def async_executor(**kwargs):
                return {"async_result": True}
            
            result = await router.execute_plan(
                decision=decision,
                tool_executors={"web_search": async_executor},
                aggregate_results=False
            )
            
            assert result.success is True
            assert result.tool_results[0].result["async_result"] is True
    
    def test_parse_json_with_markdown(self, mock_genai):
        """Test JSON parsing handles markdown code blocks."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            
            # Response with markdown code block
            response_text = """```json
{
    "reasoning": "Test",
    "selected_tools": [],
    "execution_plan": [],
    "trade_offs": "None",
    "confidence": "medium",
    "fallback_plan": "None",
    "estimated_cost": "low",
    "estimated_time": "fast"
}
```"""
            
            decision = router._parse_response(response_text)
            
            assert isinstance(decision, RoutingDecision)
            assert decision.confidence == Confidence.MEDIUM
    
    def test_list_tools(self, mock_genai):
        """Test list_tools returns tool list."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            tools = router.list_tools()
            
            assert isinstance(tools, list)
            assert len(tools) > 0
    
    def test_get_tool(self, mock_genai):
        """Test get_tool returns correct tool."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            from src.llm.router.router_agent import RouterAgent
            
            router = RouterAgent()
            tool = router.get_tool("web_search")
            
            assert tool is not None
            assert tool.id == "web_search"


class TestRouterAgentSingleton:
    """Tests for singleton pattern."""
    
    def test_get_router_agent_returns_same_instance(self):
        """Test get_router_agent returns singleton."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            with patch('src.llm.router.router_agent.genai'):
                from src.llm.router.router_agent import (
                    get_router_agent,
                    reset_router_agent,
                )
                
                reset_router_agent()
                
                router1 = get_router_agent()
                router2 = get_router_agent()
                
                assert router1 is router2
                
                reset_router_agent()
    
    def test_reset_router_agent(self):
        """Test reset_router_agent clears singleton."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            with patch('src.llm.router.router_agent.genai'):
                from src.llm.router.router_agent import (
                    get_router_agent,
                    reset_router_agent,
                )
                
                reset_router_agent()
                router1 = get_router_agent()
                
                reset_router_agent()
                router2 = get_router_agent()
                
                assert router1 is not router2
                
                reset_router_agent()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
