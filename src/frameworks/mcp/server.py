"""
MCP Server for PharmaBula

Implements Model Context Protocol (MCP) server with:
- Resources: Drug bulletins as browsable resources
- Tools: Search, query, and interaction checking
- Prompts: Pre-defined prompts for common queries
- Context management: Semantic message selection
"""

import asyncio
import json
import logging
from typing import Any, Sequence
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
    INTERNAL_ERROR,
    INVALID_PARAMS,
)
import anthropic

from src.config import get_settings
from src.shared.schemas.judges import JudgmentDecision

logger = logging.getLogger(__name__)


# ============================================================
# MCP Server Implementation
# ============================================================

class PharmaBulaMCPServer:
    """
    MCP Server for PharmaBula drug information system.
    
    MCP-Specific Features Used:
    - Resources: Expose bulas as browsable documents
    - Tools: Drug search, query, interaction checking
    - Prompts: Pre-defined prompt templates
    - Server Protocol: Full MCP specification compliance
    - Anthropic Claude: Native integration for generation
    """
    
    def __init__(self):
        """Initialize MCP server with all components."""
        self.app = Server("pharmabula-mcp-server")
        self.settings = get_settings()
        
        # Initialize Anthropic client for Claude
        self.anthropic_client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key
        )
        
        # Lazy-loaded services
        self._drug_service = None
        self._judge_pipeline = None
        
        # Register all MCP handlers
        self._register_handlers()
        
        logger.info("PharmaBula MCP server initialized")
    
    @property
    def drug_service(self):
        """Lazy-load drug service to avoid circular imports."""
        if self._drug_service is None:
            from src.services.drug_service import get_drug_service
            self._drug_service = get_drug_service()
        return self._drug_service
    
    @property
    def judge_pipeline(self):
        """Lazy-load judge pipeline."""
        if self._judge_pipeline is None:
            from src.frameworks.mcp.judges import MCPJudgePipeline
            self._judge_pipeline = MCPJudgePipeline()
        return self._judge_pipeline
    
    def _register_handlers(self):
        """Register all MCP protocol handlers."""
        
        # ==========================================
        # RESOURCES - Expose drug bulletins
        # ==========================================
        
        @self.app.list_resources()
        async def list_resources() -> list[Resource]:
            """
            List available drug bulletin resources.
            
            MCP resources allow clients to browse and retrieve documents.
            """
            return [
                Resource(
                    uri="pharmabula://bulas/profissional",
                    name="Bulas Profissionais",
                    mimeType="application/json",
                    description="Bulas para profissionais de saúde com informações técnicas completas"
                ),
                Resource(
                    uri="pharmabula://bulas/paciente",
                    name="Bulas para Pacientes",
                    mimeType="application/json",
                    description="Bulas simplificadas para pacientes"
                ),
                Resource(
                    uri="pharmabula://medicamentos/lista",
                    name="Lista de Medicamentos",
                    mimeType="application/json",
                    description="Lista de todos os medicamentos disponíveis no sistema"
                ),
                Resource(
                    uri="pharmabula://interacoes/database",
                    name="Base de Interações",
                    mimeType="application/json",
                    description="Base de dados de interações medicamentosas conhecidas"
                )
            ]
        
        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            """
            Read a specific drug bulletin resource.
            
            Parses the URI to determine what content to return.
            """
            if not uri.startswith("pharmabula://"):
                raise ValueError(f"Unknown resource URI scheme: {uri}")
            
            path = uri.replace("pharmabula://", "")
            parts = path.split("/")
            
            if parts[0] == "bulas":
                tipo = parts[1] if len(parts) > 1 else "profissional"
                return await self._get_bulas_list(tipo)
            
            elif parts[0] == "medicamentos":
                if len(parts) > 2:
                    # Specific medication: pharmabula://medicamentos/nome/paracetamol
                    drug_name = parts[2]
                    return await self._get_drug_info(drug_name)
                else:
                    return await self._get_medications_list()
            
            elif parts[0] == "interacoes":
                return await self._get_interactions_database()
            
            raise ValueError(f"Unknown resource path: {path}")
        
        # ==========================================
        # TOOLS - Drug query and analysis
        # ==========================================
        
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """
            List available MCP tools.
            
            Tools are functions that can be called by the MCP client.
            """
            return [
                Tool(
                    name="search_bulas",
                    description="Busca semântica por informações em bulas de medicamentos",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Termo de busca (nome do medicamento, condição, etc)"
                            },
                            "tipo_bula": {
                                "type": "string",
                                "enum": ["profissional", "paciente", "ambos"],
                                "description": "Tipo de bula para buscar",
                                "default": "ambos"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Número de resultados a retornar",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="query_medication",
                    description="Responde perguntas sobre medicamentos com avaliação de qualidade por juízes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Pergunta sobre o medicamento"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["patient", "professional"],
                                "description": "Modo de resposta",
                                "default": "patient"
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="check_interactions",
                    description="Verifica interações entre dois ou mais medicamentos",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "drugs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Lista de medicamentos para verificar interações",
                                "minItems": 2
                            }
                        },
                        "required": ["drugs"]
                    }
                ),
                Tool(
                    name="get_drug_summary",
                    description="Obtém um resumo completo de um medicamento",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "drug_name": {
                                "type": "string",
                                "description": "Nome do medicamento"
                            },
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "indicacoes",
                                        "contraindicacoes",
                                        "posologia",
                                        "efeitos_colaterais",
                                        "interacoes",
                                        "precaucoes"
                                    ]
                                },
                                "description": "Seções específicas a incluir (opcional, todas se não especificado)"
                            }
                        },
                        "required": ["drug_name"]
                    }
                ),
                Tool(
                    name="validate_response",
                    description="Valida uma resposta sobre medicamentos através do pipeline de juízes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Pergunta original"
                            },
                            "response": {
                                "type": "string",
                                "description": "Resposta a ser validada"
                            },
                            "sources": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Fontes utilizadas na resposta"
                            }
                        },
                        "required": ["query", "response"]
                    }
                )
            ]
        
        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
            """
            Execute a tool and return results.
            
            Dispatches to the appropriate handler based on tool name.
            """
            try:
                if name == "search_bulas":
                    result = await self._tool_search_bulas(
                        query=arguments["query"],
                        tipo_bula=arguments.get("tipo_bula", "ambos"),
                        top_k=arguments.get("top_k", 5)
                    )
                
                elif name == "query_medication":
                    result = await self._tool_query_medication(
                        question=arguments["question"],
                        mode=arguments.get("mode", "patient")
                    )
                
                elif name == "check_interactions":
                    result = await self._tool_check_interactions(
                        drugs=arguments["drugs"]
                    )
                
                elif name == "get_drug_summary":
                    result = await self._tool_get_drug_summary(
                        drug_name=arguments["drug_name"],
                        sections=arguments.get("sections")
                    )
                
                elif name == "validate_response":
                    result = await self._tool_validate_response(
                        query=arguments["query"],
                        response=arguments["response"],
                        sources=arguments.get("sources", [])
                    )
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )]
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": name
                    }, ensure_ascii=False)
                )]
        
        # ==========================================
        # PROMPTS - Pre-defined prompt templates
        # ==========================================
        
        @self.app.list_prompts()
        async def list_prompts() -> list[Prompt]:
            """
            List available prompt templates.
            
            MCP prompts provide reusable templates for common queries.
            """
            return [
                Prompt(
                    name="drug_info",
                    description="Obter informações gerais sobre um medicamento",
                    arguments=[
                        PromptArgument(
                            name="drug_name",
                            description="Nome do medicamento",
                            required=True
                        ),
                        PromptArgument(
                            name="mode",
                            description="Modo: patient ou professional",
                            required=False
                        )
                    ]
                ),
                Prompt(
                    name="interaction_check",
                    description="Verificar interações entre medicamentos",
                    arguments=[
                        PromptArgument(
                            name="drug1",
                            description="Primeiro medicamento",
                            required=True
                        ),
                        PromptArgument(
                            name="drug2",
                            description="Segundo medicamento",
                            required=True
                        )
                    ]
                ),
                Prompt(
                    name="dosage_guide",
                    description="Guia de dosagem para um medicamento",
                    arguments=[
                        PromptArgument(
                            name="drug_name",
                            description="Nome do medicamento",
                            required=True
                        ),
                        PromptArgument(
                            name="patient_type",
                            description="Tipo: adulto, criança, idoso",
                            required=False
                        )
                    ]
                )
            ]
        
        @self.app.get_prompt()
        async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
            """
            Get a specific prompt template with arguments filled in.
            """
            args = arguments or {}
            
            if name == "drug_info":
                drug_name = args.get("drug_name", "medicamento")
                mode = args.get("mode", "patient")
                
                return GetPromptResult(
                    description=f"Informações sobre {drug_name}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"Por favor, forneça informações completas sobre o medicamento {drug_name}. "
                                     f"Inclua: indicações, contraindicações, posologia, efeitos colaterais e interações. "
                                     f"Modo de resposta: {mode}."
                            )
                        )
                    ]
                )
            
            elif name == "interaction_check":
                drug1 = args.get("drug1", "medicamento 1")
                drug2 = args.get("drug2", "medicamento 2")
                
                return GetPromptResult(
                    description=f"Verificar interação entre {drug1} e {drug2}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"Verifique se existe interação medicamentosa entre {drug1} e {drug2}. "
                                     f"Se houver, explique os riscos e recomendações."
                            )
                        )
                    ]
                )
            
            elif name == "dosage_guide":
                drug_name = args.get("drug_name", "medicamento")
                patient_type = args.get("patient_type", "adulto")
                
                return GetPromptResult(
                    description=f"Guia de dosagem de {drug_name} para {patient_type}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"Qual a dosagem recomendada de {drug_name} para {patient_type}? "
                                     f"Inclua frequência, duração do tratamento e ajustes necessários."
                            )
                        )
                    ]
                )
            
            raise ValueError(f"Unknown prompt: {name}")
    
    # ==========================================
    # Resource Handlers
    # ==========================================
    
    async def _get_bulas_list(self, tipo: str) -> str:
        """Get list of available bulas."""
        # Would connect to actual data source
        return json.dumps({
            "tipo": tipo,
            "total": 100,
            "bulas": [
                {"id": "bula_001", "medicamento": "Paracetamol", "fabricante": "Genérico"},
                {"id": "bula_002", "medicamento": "Ibuprofeno", "fabricante": "Genérico"},
                {"id": "bula_003", "medicamento": "Dipirona", "fabricante": "Genérico"}
            ]
        }, ensure_ascii=False)
    
    async def _get_medications_list(self) -> str:
        """Get list of all medications."""
        # Would fetch from database
        return json.dumps({
            "total": 50,
            "medicamentos": [
                "Paracetamol", "Ibuprofeno", "Dipirona", "Omeprazol", "Amoxicilina"
            ]
        }, ensure_ascii=False)
    
    async def _get_drug_info(self, drug_name: str) -> str:
        """Get information about a specific drug."""
        context, _ = await self.drug_service.get_drug_context(drug_name, n_results=5)
        
        return json.dumps({
            "medicamento": drug_name,
            "documentos": context
        }, ensure_ascii=False)
    
    async def _get_interactions_database(self) -> str:
        """Get interactions database summary."""
        return json.dumps({
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "total_interactions": 500,
            "categories": [
                "Antibióticos", "Anti-inflamatórios", "Analgésicos",
                "Antialérgicos", "Cardiovasculares"
            ]
        }, ensure_ascii=False)
    
    # ==========================================
    # Tool Handlers
    # ==========================================
    
    async def _tool_search_bulas(
        self,
        query: str,
        tipo_bula: str,
        top_k: int
    ) -> dict:
        """Search for relevant bulas."""
        context, was_updated = await self.drug_service.get_drug_context(
            query, n_results=top_k
        )
        
        return {
            "query": query,
            "tipo_bula": tipo_bula,
            "results_count": len(context),
            "was_updated": was_updated,
            "results": context
        }
    
    async def _tool_query_medication(
        self,
        question: str,
        mode: str
    ) -> dict:
        """Query medication information with judge evaluation."""
        
        # Get context
        context, _ = await self.drug_service.get_drug_context(question, n_results=5)
        
        # Format context for Claude
        context_str = "\n\n".join([
            f"### {doc.get('source', 'unknown')}\n{doc.get('content', '')}"
            for doc in context
        ])
        
        # Generate with Claude
        response = self.anthropic_client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=2000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""Você é um assistente de bulário eletrônico brasileiro.

CONTEXTO DAS BULAS:
{context_str}

MODO: {mode}

PERGUNTA: {question}

Responda em JSON:
{{
    "response": "sua resposta",
    "confidence": "alta|média|baixa",
    "sources": ["fonte1"],
    "disclaimer": "aviso obrigatório"
}}"""
                }
            ]
        )
        
        answer = response.content[0].text
        
        # Evaluate through judges
        if self.settings.enable_judge_pipeline:
            judgment = await self.judge_pipeline.evaluate(
                user_query=question,
                generated_response=answer,
                retrieved_documents=context,
                mode=mode
            )
            
            return {
                "query": question,
                "mode": mode,
                "answer": answer,
                "judgment": {
                    "decision": judgment.final_decision.value,
                    "score": judgment.overall_score,
                    "score_breakdown": judgment.score_breakdown
                },
                "sources": [doc.get("source") for doc in context]
            }
        
        return {
            "query": question,
            "mode": mode,
            "answer": answer,
            "sources": [doc.get("source") for doc in context]
        }
    
    async def _tool_check_interactions(self, drugs: list[str]) -> dict:
        """Check for drug interactions."""
        if len(drugs) < 2:
            return {"error": "Forneça pelo menos dois medicamentos"}
        
        # Get context for all drugs
        all_context = []
        for drug in drugs:
            context, _ = await self.drug_service.get_drug_context(
                f"interações medicamentosas {drug}", n_results=2
            )
            all_context.extend(context)
        
        context_str = "\n\n".join([
            doc.get("content", "") for doc in all_context
        ])
        
        # Generate interaction analysis with Claude
        response = self.anthropic_client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=2000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analise interações entre: {', '.join(drugs)}

CONTEXTO:
{context_str}

Responda em JSON:
{{
    "drugs": {json.dumps(drugs)},
    "interactions": [
        {{
            "pair": ["med1", "med2"],
            "severity": "alta|média|baixa",
            "effect": "descrição do efeito",
            "recommendation": "recomendação"
        }}
    ],
    "overall_risk": "alto|médio|baixo|nenhum",
    "disclaimer": "aviso"
}}"""
                }
            ]
        )
        
        return json.loads(response.content[0].text)
    
    async def _tool_get_drug_summary(
        self,
        drug_name: str,
        sections: list[str] = None
    ) -> dict:
        """Get comprehensive drug summary."""
        context, _ = await self.drug_service.get_drug_context(drug_name, n_results=10)
        
        sections_str = ", ".join(sections) if sections else "todas as seções"
        context_str = "\n\n".join([doc.get("content", "") for doc in context])
        
        response = self.anthropic_client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=3000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""Faça um resumo de {drug_name} incluindo: {sections_str}

BULAS:
{context_str}

Responda em JSON estruturado com as seções solicitadas."""
                }
            ]
        )
        
        return {
            "drug_name": drug_name,
            "sections_requested": sections,
            "summary": response.content[0].text,
            "sources_count": len(context)
        }
    
    async def _tool_validate_response(
        self,
        query: str,
        response: str,
        sources: list[dict]
    ) -> dict:
        """Validate a response through the judge pipeline."""
        judgment = await self.judge_pipeline.evaluate(
            user_query=query,
            generated_response=response,
            retrieved_documents=sources,
            mode="patient"
        )
        
        return {
            "query": query,
            "response_valid": judgment.final_decision in [
                JudgmentDecision.APPROVED,
                JudgmentDecision.APPROVED_WITH_CAVEATS
            ],
            "decision": judgment.final_decision.value,
            "overall_score": judgment.overall_score,
            "score_breakdown": judgment.score_breakdown,
            "blocking_issues": judgment.blocking_issues,
            "required_disclaimers": judgment.disclaimers_to_add,
            "revision_suggestions": judgment.revision_suggestions
        }
    
    async def run(self):
        """Run the MCP server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options()
            )


# ============================================================
# Entry Point
# ============================================================

def main():
    """Main entry point for running the MCP server."""
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    server = PharmaBulaMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
