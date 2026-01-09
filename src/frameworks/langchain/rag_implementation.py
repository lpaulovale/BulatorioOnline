"""
LangChain RAG Implementation - Complete Agent.

Fully self-contained implementation using LangChain with LCEL and ReAct agents.
Includes: RAG, Router, Judges, JSON response formatting.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================
# LangChain Prompts (Self-contained)
# ============================================================

SYSTEM_TEMPLATE = """Você é o PharmaBula, um assistente especializado em informações sobre medicamentos do bulário eletrônico brasileiro (ANVISA).

MODO: {mode}

DIRETRIZES:
- Responda APENAS com base nas informações das bulas oficiais fornecidas no contexto
- Seja preciso, factual e baseado em evidências  
- Para modo "patient": use linguagem simples, acessível e empática
- Para modo "professional": use terminologia técnica e científica
- NUNCA invente ou extrapole informações além das fontes
- Inclua disclaimers de segurança apropriados

CONTEXTO DAS BULAS:
{context}"""


REACT_TEMPLATE = """Você é o PharmaBula, assistente de informações sobre medicamentos.

Você tem acesso às seguintes ferramentas:
{tools}

Use o seguinte formato:

Question: a pergunta do usuário
Thought: você deve pensar sobre o que fazer
Action: a ação a tomar, uma das [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (repita Thought/Action/Action Input/Observation quantas vezes necessário)
Thought: Agora sei a resposta final
Final Answer: responda em JSON válido com: {{"response": "...", "confidence": "alta|média|baixa", "sources": [...], "disclaimer": "..."}}

Question: {input}
{agent_scratchpad}"""


# ============================================================
# LangChain Tools
# ============================================================

class SearchInput(BaseModel):
    query: str = Field(description="Termo de busca")
    limit: int = Field(default=5, description="Número máximo de resultados")


class DrugDetailsInput(BaseModel):
    drug_name: str = Field(description="Nome do medicamento")


class InteractionsInput(BaseModel):
    drugs: List[str] = Field(description="Lista de medicamentos")


# We'll create tools dynamically with access to vector store
def create_langchain_tools(vector_store):
    """Create LangChain tools with vector store access."""
    
    @tool("search_drugs", args_schema=SearchInput)
    def search_drugs(query: str, limit: int = 5) -> str:
        """Busca informações sobre medicamentos no banco de bulas."""
        results = vector_store.search(query, n_results=limit)
        return json.dumps(results, ensure_ascii=False, default=str)
    
    @tool("get_drug_details", args_schema=DrugDetailsInput)
    def get_drug_details(drug_name: str) -> str:
        """Obtém informações detalhadas de um medicamento."""
        results = vector_store.search(drug_name, n_results=3)
        return json.dumps(results, ensure_ascii=False, default=str)
    
    @tool("check_interactions", args_schema=InteractionsInput)
    def check_interactions(drugs: List[str]) -> str:
        """Verifica interações entre medicamentos."""
        query = f"interações medicamentosas {' '.join(drugs)}"
        results = vector_store.search(query, n_results=5)
        return json.dumps(results, ensure_ascii=False, default=str)
    
    return [search_drugs, get_drug_details, check_interactions]


# ============================================================
# LangChain Agent Implementation
# ============================================================

class LangChainAgent:
    """
    Complete LangChain Agent using LCEL.
    
    Features:
    - LCEL chain composition
    - ReAct agent for tool use
    - Gemini as LLM backend
    - Structured JSON outputs
    - Conversation memory
    """
    
    def __init__(self):
        """Initialize LangChain Agent."""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        self.history: List[Any] = []  # LangChain messages
        self.max_history = settings.MAX_CONTEXT_MESSAGES
        
        # Lazy loaded components
        self._vector_store = None
        self._tools = None
        self._agent_executor = None
        self._judge_pipeline = None
        
        logger.info(f"LangChain Agent initialized with model: {settings.GEMINI_MODEL}")
    
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            from src.database.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store
    
    @property
    def tools(self):
        """Lazy load tools."""
        if self._tools is None:
            self._tools = create_langchain_tools(self.vector_store)
        return self._tools
    
    @property
    def agent_executor(self):
        """Lazy load agent executor."""
        if self._agent_executor is None:
            prompt = ChatPromptTemplate.from_template(REACT_TEMPLATE)
            agent = create_react_agent(self.llm, self.tools, prompt)
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        return self._agent_executor
    
    @property
    def judge_pipeline(self):
        """Lazy load judge pipeline."""
        if self._judge_pipeline is None:
            from src.frameworks.langchain.judges import LangChainJudgePipeline
            self._judge_pipeline = LangChainJudgePipeline()
        return self._judge_pipeline
    
    def _format_documents(self, docs: List[Dict]) -> str:
        """Format documents for context."""
        if not docs:
            return "Nenhum documento encontrado."
        
        formatted = []
        for doc in docs:
            source = doc.get("source", doc.get("metadata", {}).get("drug_name", "Documento"))
            content = doc.get("content", "")
            formatted.append(f"### {source}\n\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Search vector store for relevant documents."""
        try:
            return self.vector_store.search(query, n_results=limit)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def query(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5,
        use_agent: bool = False
    ) -> str:
        """
        Execute a complete RAG query.
        
        Args:
            question: User question
            mode: Response mode (patient/professional)
            n_context: Number of context documents
            use_agent: Use ReAct agent with tools
        
        Returns:
            JSON string with response
        """
        start_time = time.time()
        
        try:
            if use_agent:
                # Use ReAct agent
                result = await self.agent_executor.ainvoke({"input": question})
                answer = result.get("output", "")
            else:
                # Use simple LCEL chain
                docs = self.search_documents(question, n_context)
                context = self._format_documents(docs)
                
                # Build prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_TEMPLATE),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}\n\nResponda em JSON válido.")
                ])
                
                # Create chain
                chain = prompt | self.llm | StrOutputParser()
                
                # Execute
                answer = await chain.ainvoke({
                    "mode": mode,
                    "context": context,
                    "history": self.history[-self.max_history:],
                    "question": question
                })
                
                docs_for_sources = docs
            
            # Run judge pipeline if enabled
            if settings.ENABLE_JUDGE_PIPELINE:
                try:
                    docs = self.search_documents(question, n_context) if use_agent else docs
                    judgment = await self.judge_pipeline.evaluate(
                        user_query=question,
                        generated_response=answer,
                        retrieved_documents=docs,
                        mode=mode
                    )
                    
                    if judgment.final_response:
                        answer = judgment.final_response
                except Exception as e:
                    logger.warning(f"Judge pipeline error: {e}")
            
            # Ensure valid JSON
            docs = self.search_documents(question, 3) if use_agent else docs
            answer = self._ensure_json_response(answer, docs, mode)
            
            # Update history
            self.history.append(HumanMessage(content=question))
            self.history.append(AIMessage(content=answer))
            
            # Trim history
            if len(self.history) > self.max_history * 2:
                self.history = self.history[-(self.max_history * 2):]
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"Query completed in {latency:.2f}ms")
            
            return answer
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return json.dumps({
                "response": f"Erro ao processar sua pergunta: {str(e)}",
                "confidence": "baixa",
                "sources": [],
                "disclaimer": "Consulte um profissional de saúde."
            }, ensure_ascii=False)
    
    def _ensure_json_response(self, answer: str, docs: List[Dict], mode: str) -> str:
        """Ensure response is valid JSON."""
        try:
            # Extract JSON from markdown
            if "```json" in answer:
                answer = answer.split("```json")[1].split("```")[0]
            elif "```" in answer:
                parts = answer.split("```")
                if len(parts) >= 2:
                    answer = parts[1]
            
            parsed = json.loads(answer.strip())
            
            # Ensure required fields
            if "response" not in parsed:
                parsed["response"] = answer
            if "confidence" not in parsed:
                parsed["confidence"] = "média"
            if "sources" not in parsed:
                parsed["sources"] = [d.get("source", "") for d in docs[:3] if d.get("source")]
            if "disclaimer" not in parsed:
                parsed["disclaimer"] = "Consulte sempre um profissional de saúde."
            
            return json.dumps(parsed, ensure_ascii=False)
            
        except json.JSONDecodeError:
            sources = [d.get("source", "") for d in docs[:3] if d.get("source")]
            return json.dumps({
                "response": answer,
                "confidence": "média",
                "sources": sources,
                "disclaimer": "Consulte sempre um profissional de saúde."
            }, ensure_ascii=False)
    
    async def check_interactions(self, drugs: List[str]) -> str:
        """Check drug interactions."""
        if len(drugs) < 2:
            return json.dumps({
                "error": "Forneça pelo menos dois medicamentos.",
                "drugs_analyzed": drugs
            }, ensure_ascii=False)
        
        query = f"Verifique interações entre: {', '.join(drugs)}"
        return await self.query(query, mode="professional", use_agent=True)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
    
    def get_history(self) -> List[Dict]:
        """Get conversation history as dicts."""
        result = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return result


# ============================================================
# Singleton
# ============================================================

_langchain_agent: Optional[LangChainAgent] = None


def get_langchain_agent() -> LangChainAgent:
    """Get singleton LangChain agent instance."""
    global _langchain_agent
    if _langchain_agent is None:
        _langchain_agent = LangChainAgent()
    return _langchain_agent


def reset_langchain_agent() -> None:
    """Reset singleton instance."""
    global _langchain_agent
    _langchain_agent = None
