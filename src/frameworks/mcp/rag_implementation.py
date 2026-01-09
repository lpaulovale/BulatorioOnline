"""
MCP RAG Implementation - Complete Agent.

Fully self-contained implementation using Model Context Protocol with Anthropic Claude.
Includes: RAG, Router, Judges, JSON response formatting.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================
# MCP Prompts (Self-contained)
# ============================================================

SYSTEM_PROMPT = """Você é o PharmaBula, um assistente especializado em informações sobre medicamentos do bulário eletrônico brasileiro (ANVISA).

MODO: {mode}

DIRETRIZES:
- Responda APENAS com base nas informações das bulas oficiais fornecidas no contexto
- Seja preciso, factual e baseado em evidências
- Para modo "patient": use linguagem simples, acessível e empática
- Para modo "professional": use terminologia técnica e científica
- NUNCA invente ou extrapole informações além das fontes
- Inclua disclaimers de segurança apropriados

CONTEXTO DAS BULAS:
{context}

Responda em JSON válido:
{{
    "response": "sua resposta detalhada",
    "confidence": "alta|média|baixa",
    "sources": ["fonte1", "fonte2"],
    "disclaimer": "aviso de segurança"
}}"""


# ============================================================
# MCP Agent Implementation
# ============================================================

class MCPAgent:
    """
    Complete MCP Agent using Anthropic Claude.
    
    Features:
    - Prompt caching for efficiency
    - Tool use via Claude's native tool calling
    - Structured JSON outputs
    - Conversation history management
    """
    
    # Tool definitions for Claude
    TOOLS = [
        {
            "name": "search_drugs",
            "description": "Busca informações sobre medicamentos no banco de bulas",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_drug_details",
            "description": "Obtém informações detalhadas de um medicamento específico",
            "input_schema": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"}
                },
                "required": ["drug_name"]
            }
        },
        {
            "name": "check_interactions",
            "description": "Verifica interações entre medicamentos",
            "input_schema": {
                "type": "object",
                "properties": {
                    "drugs": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["drugs"]
            }
        }
    ]
    
    def __init__(self):
        """Initialize MCP Agent."""
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.GENERATION_MODEL
        self.history: List[Dict[str, str]] = []
        self.max_history = settings.MAX_CONTEXT_MESSAGES
        
        # Lazy loaded components
        self._vector_store = None
        self._judge_pipeline = None
        
        logger.info(f"MCP Agent initialized with model: {self.model}")
    
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
            from src.frameworks.mcp.judges import MCPJudgePipeline
            self._judge_pipeline = MCPJudgePipeline()
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
            results = self.vector_store.search(query, n_results=limit)
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a tool and return result."""
        if tool_name == "search_drugs":
            results = self.search_documents(
                tool_input.get("query", ""),
                tool_input.get("limit", 5)
            )
            return json.dumps(results, ensure_ascii=False, default=str)
        
        elif tool_name == "get_drug_details":
            results = self.search_documents(tool_input.get("drug_name", ""), 3)
            return json.dumps(results, ensure_ascii=False, default=str)
        
        elif tool_name == "check_interactions":
            drugs = tool_input.get("drugs", [])
            query = f"interações medicamentosas {' '.join(drugs)}"
            results = self.search_documents(query, 5)
            return json.dumps(results, ensure_ascii=False, default=str)
        
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    async def query(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5,
        use_tools: bool = True
    ) -> str:
        """
        Execute a complete RAG query with agent loop.
        
        Returns JSON string with response.
        """
        start_time = time.time()
        
        try:
            # Step 1: Search for relevant documents
            docs = self.search_documents(question, n_context)
            context = self._format_documents(docs)
            
            # Step 2: Build messages with cache_control for system prompt
            # Anthropic prompt caching: cache static parts to reduce latency/cost
            system_with_cache = [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(mode=mode, context=context),
                    "cache_control": {"type": "ephemeral"}  # Cache for session
                }
            ]
            
            # Build conversation messages from history
            messages = []
            for msg in self.history[-self.max_history:]:
                messages.append(msg)
            messages.append({"role": "user", "content": question})
            
            # Step 3: Generate with Claude using prompt caching
            if use_tools and self.TOOLS:
                # Use tool calling with cached system prompt
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    system=system_with_cache,
                    tools=self.TOOLS,
                    messages=messages
                )
                
                # Handle tool use
                while response.stop_reason == "tool_use":
                    tool_calls = [b for b in response.content if b.type == "tool_use"]
                    
                    tool_results = []
                    for tool_call in tool_calls:
                        result = self._execute_tool(tool_call.name, tool_call.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": result
                        })
                    
                    # Continue conversation with tool results
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                    
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        temperature=0.3,
                        system=system_with_cache,
                        tools=self.TOOLS,
                        messages=messages
                    )
                
                answer = response.content[0].text
            else:
                # Direct generation without tools
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    system=system_with_cache,
                    messages=messages
                )
                answer = response.content[0].text
            
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
            
            # Step 5: Ensure valid JSON response
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
                "disclaimer": "Consulte um profissional de saúde para informações precisas."
            }, ensure_ascii=False)
    
    def _ensure_json_response(self, answer: str, docs: List[Dict], mode: str) -> str:
        """Ensure response is valid JSON."""
        try:
            # Try to parse existing JSON
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
            # Wrap plain text in JSON structure
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
                "error": "Forneça pelo menos dois medicamentos para verificar interações.",
                "drugs_analyzed": drugs
            }, ensure_ascii=False)
        
        query = f"Verifique interações medicamentosas entre: {', '.join(drugs)}"
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

_mcp_agent: Optional[MCPAgent] = None


def get_mcp_agent() -> MCPAgent:
    """Get singleton MCP agent instance."""
    global _mcp_agent
    if _mcp_agent is None:
        _mcp_agent = MCPAgent()
    return _mcp_agent


def reset_mcp_agent() -> None:
    """Reset singleton instance."""
    global _mcp_agent
    _mcp_agent = None
