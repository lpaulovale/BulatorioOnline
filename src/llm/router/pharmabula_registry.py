"""
PharmaBula Tool Registry

Defines all tools available for the PharmaBula medication assistant.
These tools integrate with the existing services (vector store, drug service, etc.)
Uses Pydantic schemas for type-safe input/output validation.
"""

from src.llm.router.schemas import Tool, API, ToolRegistry, CostTier, Latency
from src.llm.router.pharmabula_schemas import (
    DrugSearchInput,
    DrugSearchOutput,
    DrugContextInput,
    DrugContextOutput,
    InteractionCheckInput,
    InteractionCheckOutput,
    DrugSummaryInput,
    DrugSummaryOutput,
    AnvisaFetchInput,
    AnvisaFetchOutput,
    QuickAnswerInput,
    QuickAnswerOutput,
    ClinicalProtocolInput,
    ClinicalProtocolOutput,
)


def get_pharmabula_tool_registry() -> ToolRegistry:
    """
    Get the PharmaBula-specific tool registry.
    
    Contains tools for:
    - Drug information search and retrieval
    - Drug interaction checking
    - Summary generation
    - ANVISA API access
    - Clinical protocols (PCDT) from CONITEC
    
    All tools use Pydantic schemas for input/output validation.
    """
    return ToolRegistry(
        tools=[
            Tool(
                id="drug_search",
                description="Busca medicamentos no banco de dados por nome, princípio ativo ou indicação",
                capabilities=[
                    "search_by_name",
                    "search_by_ingredient",
                    "search_by_indication",
                    "semantic_search"
                ],
                input_schema=DrugSearchInput.model_json_schema(),
                output_schema=DrugSearchOutput.model_json_schema(),
                cost_tier=CostTier.LOW,
                latency=Latency.FAST,
                examples=[
                    "Buscar paracetamol",
                    "Encontrar medicamentos para dor de cabeça",
                    "Pesquisar por ibuprofeno"
                ]
            ),
            Tool(
                id="drug_context",
                description="Obtém contexto detalhado da bula de um medicamento para responder perguntas",
                capabilities=[
                    "get_full_context",
                    "get_section",
                    "rag_retrieval"
                ],
                input_schema=DrugContextInput.model_json_schema(),
                output_schema=DrugContextOutput.model_json_schema(),
                cost_tier=CostTier.LOW,
                latency=Latency.FAST,
                examples=[
                    "Quais os efeitos colaterais do paracetamol?",
                    "Como tomar ibuprofeno?",
                    "Contraindicações da dipirona"
                ]
            ),
            Tool(
                id="interaction_check",
                description="Verifica interações medicamentosas entre dois ou mais medicamentos",
                capabilities=[
                    "check_pair",
                    "check_multiple",
                    "severity_assessment"
                ],
                input_schema=InteractionCheckInput.model_json_schema(),
                output_schema=InteractionCheckOutput.model_json_schema(),
                cost_tier=CostTier.MEDIUM,
                latency=Latency.MODERATE,
                requirements=["Mínimo de 2 medicamentos"],
                examples=[
                    "Verificar interação entre aspirina e ibuprofeno",
                    "Checar se posso tomar paracetamol com dipirona"
                ]
            ),
            Tool(
                id="drug_summary",
                description="Gera um resumo completo da bula de um medicamento",
                capabilities=[
                    "full_summary",
                    "section_summary",
                    "patient_friendly"
                ],
                input_schema=DrugSummaryInput.model_json_schema(),
                output_schema=DrugSummaryOutput.model_json_schema(),
                cost_tier=CostTier.MEDIUM,
                latency=Latency.MODERATE,
                examples=[
                    "Resumo completo do paracetamol",
                    "Resumo das contraindicações do ibuprofeno"
                ]
            ),
            Tool(
                id="anvisa_fetch",
                description="Busca dados atualizados diretamente da API da ANVISA",
                capabilities=[
                    "search_anvisa",
                    "fetch_bulletin",
                    "update_cache"
                ],
                input_schema=AnvisaFetchInput.model_json_schema(),
                output_schema=AnvisaFetchOutput.model_json_schema(),
                cost_tier=CostTier.HIGH,
                latency=Latency.SLOW,
                requirements=["Conexão com internet", "API ANVISA disponível"],
                examples=[
                    "Buscar dados atualizados do paracetamol na ANVISA",
                    "Atualizar informações de um medicamento"
                ]
            ),
            Tool(
                id="quick_answer",
                description="Responde perguntas simples e frequentes sobre medicamentos",
                capabilities=[
                    "dosage_info",
                    "common_side_effects",
                    "basic_usage"
                ],
                input_schema=QuickAnswerInput.model_json_schema(),
                output_schema=QuickAnswerOutput.model_json_schema(),
                cost_tier=CostTier.LOW,
                latency=Latency.FAST,
                examples=[
                    "Qual a dose de paracetamol para adulto?",
                    "Paracetamol pode ser tomado de estômago vazio?"
                ]
            ),
            Tool(
                id="clinical_protocol",
                description="Busca protocolos clínicos e diretrizes terapêuticas (PCDT) do SUS para doenças",
                capabilities=[
                    "search_protocol",
                    "get_treatment_guidelines",
                    "get_diagnosis_criteria"
                ],
                input_schema=ClinicalProtocolInput.model_json_schema(),
                output_schema=ClinicalProtocolOutput.model_json_schema(),
                cost_tier=CostTier.MEDIUM,
                latency=Latency.MODERATE,
                requirements=["Conexão com internet"],
                examples=[
                    "Protocolo clínico para diabetes",
                    "Diretrizes de tratamento para hipertensão",
                    "PCDT de artrite reumatoide"
                ]
            )
        ],
        apis=[
            API(
                name="anvisa_api",
                endpoint="https://consultas.anvisa.gov.br/api",
                methods=["GET"],
                capabilities=["search_drugs", "get_bulletin", "get_details"],
                rate_limits={
                    "requests_per_minute": 30,
                    "requests_per_day": 1000
                },
                auth_required=False
            ),
            API(
                name="conitec_portal",
                endpoint="https://www.gov.br/conitec/pt-br",
                methods=["GET"],
                capabilities=["search_protocols", "get_pcdt", "get_guidelines"],
                rate_limits={
                    "requests_per_minute": 20,
                    "requests_per_day": 500
                },
                auth_required=False
            )
        ]
    )


# Singleton instance
_pharmabula_registry = None


def get_cached_pharmabula_registry() -> ToolRegistry:
    """Get cached PharmaBula registry (singleton)."""
    global _pharmabula_registry
    if _pharmabula_registry is None:
        _pharmabula_registry = get_pharmabula_tool_registry()
    return _pharmabula_registry
