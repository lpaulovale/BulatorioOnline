"""
OpenAI RAG Implementation - Complete Agent.

Fully self-contained implementation using OpenAI API with Function Calling.
Includes: RAG, Router, Judges, JSON response formatting.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================
# OpenAI Prompts (Self-contained)
# ============================================================

SYSTEM_PROMPT = """Você é o PharmaBula, um assistente especializado em informações sobre medicamentos do bulário eletrônico brasileiro (ANVISA).

MODO: {mode}

DIRETRIZES:
- Responda APENAS com base nas informações das bulas oficiais
- Seja preciso, factual e baseado em evidências
- Para modo "patient": use linguagem simples e empática
- Para modo "professional": use terminologia técnica
- NUNCA invente informações além das fontes
- Inclua disclaimers de segurança

CONTEXTO DAS BULAS:
{context}

Sempre responda em JSON válido com a estrutura:
{{"response": "...", "confidence": "alta|média|baixa", "sources": [...], "disclaimer": "..."}}"""


# ============================================================
# OpenAI Function Definitions
# ============================================================

FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_drugs",
            "description": "Busca informações sobre medicamentos no banco de bulas",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca"},
                    "limit": {"type": "integer", "description": "Máximo de resultados", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drug_details",
            "description": "Obtém informações detalhadas de um medicamento",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Nome do medicamento"}
                },
                "required": ["drug_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_interactions",
            "description": "Verifica interações entre medicamentos",
            "parameters": {
                "type": "object",
                "properties": {
                    "drugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de medicamentos"
                    }
                },
                "required": ["drugs"]
            }
        }
    }
]


# ============================================================
# OpenAI Agent Implementation
# ============================================================

class OpenAIAgent:
    """
    Complete OpenAI Agent using Function Calling.
    
    Features:
    - AsyncOpenAI client
    - Native function calling
    - JSON mode for structured outputs
    - Multi-turn tool execution
    - Conversation history
    """
    
    def __init__(self):
        """Initialize OpenAI Agent."""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        
        self.history: List[Dict[str, str]] = []
        self.max_history = settings.MAX_CONTEXT_MESSAGES
        
        # Lazy loaded components
        self._vector_store = None
        self._judge_pipeline = None
        
        logger.info(f"OpenAI Agent initialized with model: {self.model}")
    
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            from src.database.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store
    
    @property
    def judge_pipeline(self):
        """Lazy load judge pipeline."""
        if self._judge_pipeline is None:
            from src.frameworks.openai.judges import OpenAIJudgePipeline
            self._judge_pipeline = OpenAIJudgePipeline(
                api_key=settings.OPENAI_API_KEY,
                model=self.model
            )
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
    
    def _execute_function(self, name: str, arguments: Dict) -> str:
        """Execute a function and return result."""
        if name == "search_drugs":
            results = self.search_documents(
                arguments.get("query", ""),
                arguments.get("limit", 5)
            )
            return json.dumps(results, ensure_ascii=False, default=str)
        
        elif name == "get_drug_details":
            results = self.search_documents(arguments.get("drug_name", ""), 3)
            return json.dumps(results, ensure_ascii=False, default=str)
        
        elif name == "check_interactions":
            drugs = arguments.get("drugs", [])
            query = f"interações medicamentosas {' '.join(drugs)}"
            results = self.search_documents(query, 5)
            return json.dumps(results, ensure_ascii=False, default=str)
        
        return json.dumps({"error": f"Unknown function: {name}"})
    
    async def query(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5,
        use_functions: bool = True
    ) -> str:
        """
        Execute a complete RAG query with function calling.
        
        Returns JSON string with response.
        """
        start_time = time.time()
        
        try:
            # Step 1: Get context documents
            docs = self.search_documents(question, n_context)
            context = self._format_documents(docs)
            
            # Step 2: Build messages
            system = SYSTEM_PROMPT.format(mode=mode, context=context)
            
            messages = [{"role": "system", "content": system}]
            messages.extend(self.history[-self.max_history:])
            messages.append({"role": "user", "content": question})
            
            # Step 3: Generate with function calling
            if use_functions:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=FUNCTIONS,
                    tool_choice="auto",
                    response_format={"type": "json_object"}
                )
                
                choice = response.choices[0]
                
                # Handle function calls in a loop
                while choice.message.tool_calls:
                    # Add assistant's message with tool calls
                    messages.append(choice.message)
                    
                    # Execute each function
                    for tool_call in choice.message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        result = self._execute_function(func_name, func_args)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    
                    # Get next response
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=FUNCTIONS,
                        tool_choice="auto",
                        response_format={"type": "json_object"}
                    )
                    choice = response.choices[0]
                
                answer = choice.message.content
            else:
                # Direct generation without functions
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                answer = response.choices[0].message.content
            
            # Step 4: Run judge pipeline if enabled
            if settings.ENABLE_JUDGE_PIPELINE:
                try:
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
            
            # Step 5: Ensure valid JSON
            answer = self._ensure_json_response(answer, docs, mode)
            
            # Step 6: Update history
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            
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
            # Parse JSON (OpenAI json_object mode should already return valid JSON)
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
        """Check drug interactions using function calling."""
        if len(drugs) < 2:
            return json.dumps({
                "error": "Forneça pelo menos dois medicamentos.",
                "drugs_analyzed": drugs
            }, ensure_ascii=False)
        
        query = f"Verifique interações entre: {', '.join(drugs)}"
        return await self.query(query, mode="professional")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()
    
    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.history.copy()


# ============================================================
# Singleton
# ============================================================

_openai_agent: Optional[OpenAIAgent] = None


def get_openai_agent() -> OpenAIAgent:
    """Get singleton OpenAI agent instance."""
    global _openai_agent
    if _openai_agent is None:
        _openai_agent = OpenAIAgent()
    return _openai_agent


def reset_openai_agent() -> None:
    """Reset singleton instance."""
    global _openai_agent
    _openai_agent = None
