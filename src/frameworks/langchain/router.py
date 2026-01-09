"""
LangChain Router Agent.

Uses LangChain Agents with LCEL for tool routing.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_react_agent
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# Tool schemas
class SearchDrugsInput(BaseModel):
    query: str = Field(description="Search query for drugs")
    limit: int = Field(default=5, description="Max results")


class DrugDetailsInput(BaseModel):
    drug_name: str = Field(description="Name of the drug")


class InteractionsInput(BaseModel):
    drugs: List[str] = Field(description="List of drug names")


# LangChain Tools
@tool("search_drugs", args_schema=SearchDrugsInput)
def search_drugs_tool(query: str, limit: int = 5) -> str:
    """Search for drug information in the bulletin database."""
    from src.database.vector_store import get_vector_store
    
    vs = get_vector_store()
    results = vs.search(query, n_results=limit)
    return json.dumps(results, ensure_ascii=False, default=str)


@tool("get_drug_details", args_schema=DrugDetailsInput)
def get_drug_details_tool(drug_name: str) -> str:
    """Get detailed information about a specific drug."""
    from src.database.vector_store import get_vector_store
    
    vs = get_vector_store()
    results = vs.search(drug_name, n_results=3)
    return json.dumps(results, ensure_ascii=False, default=str)


@tool("check_interactions", args_schema=InteractionsInput)
def check_interactions_tool(drugs: List[str]) -> str:
    """Check for drug-drug interactions."""
    from src.database.vector_store import get_vector_store
    
    vs = get_vector_store()
    query = f"interações medicamentosas {' '.join(drugs)}"
    results = vs.search(query, n_results=5)
    return json.dumps(results, ensure_ascii=False, default=str)


@dataclass
class LangChainRoutingDecision:
    """Routing decision from LangChain router."""
    selected_tools: List[str]
    reasoning: str
    confidence: float
    agent_output: Optional[str] = None


class LangChainRouter:
    """
    Router using LangChain Agents.
    
    Features:
    - LCEL-based agent composition
    - ReAct agent for reasoning and acting
    - Tool calling via LangChain
    """
    
    PROMPT_TEMPLATE = """You are a helpful assistant for PharmaBula, a drug information system.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

    def __init__(self):
        """Initialize LangChain router."""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
            convert_system_message_to_human=True
        )
        
        self.tools = [
            search_drugs_tool,
            get_drug_details_tool,
            check_interactions_tool
        ]
        
        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        logger.info("LangChain Router initialized")
    
    async def route_and_execute(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> LangChainRoutingDecision:
        """
        Route and execute request using LangChain agent.
        
        Args:
            message: User request
            context: Additional context
        
        Returns:
            LangChainRoutingDecision with results
        """
        try:
            result = await self.agent_executor.ainvoke({"input": message})
            
            # Extract used tools from intermediate steps
            used_tools = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if hasattr(step[0], "tool"):
                        used_tools.append(step[0].tool)
            
            return LangChainRoutingDecision(
                selected_tools=used_tools or ["generate_response"],
                reasoning="Agent executed tools based on request",
                confidence=0.8,
                agent_output=result.get("output", "")
            )
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return LangChainRoutingDecision(
                selected_tools=[],
                reasoning=f"Error: {str(e)}",
                confidence=0.0,
                agent_output=None
            )
    
    def route_request(
        self,
        message: str,
        context: Optional[Dict] = None
    ) -> LangChainRoutingDecision:
        """
        Analyze request and determine tools (without execution).
        
        Uses LLM to determine which tools would be needed.
        """
        prompt = f"""Analyze this request and determine which tools to use.

Request: {message}

Available tools:
- search_drugs: Search drug information
- get_drug_details: Get detailed drug info
- check_interactions: Check drug interactions

Return JSON:
{{
    "selected_tools": ["tool1"],
    "reasoning": "why",
    "confidence": 0.8
}}"""

        from langchain_core.output_parsers import StrOutputParser
        
        chain = self.llm | StrOutputParser()
        result = chain.invoke(prompt)
        
        try:
            # Parse JSON from response
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            data = json.loads(result.strip())
            
            return LangChainRoutingDecision(
                selected_tools=data.get("selected_tools", []),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5)
            )
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return LangChainRoutingDecision(
                selected_tools=["search_drugs"],
                reasoning="Fallback",
                confidence=0.5
            )


# Singleton
_langchain_router: Optional[LangChainRouter] = None


def get_langchain_router() -> LangChainRouter:
    global _langchain_router
    if _langchain_router is None:
        _langchain_router = LangChainRouter()
    return _langchain_router
