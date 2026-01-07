"""
Pydantic schemas for PharmaBula tool inputs and outputs.

Provides type-safe input/output models for each tool in the registry.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ============================================================================
# DRUG SEARCH TOOL
# ============================================================================

class DrugSearchInput(BaseModel):
    """Input schema for drug_search tool."""
    query: str = Field(..., description="Termo de busca (nome, princípio ativo ou indicação)")
    n_results: int = Field(default=5, ge=1, le=20, description="Número de resultados")


class DrugSearchResult(BaseModel):
    """Single drug search result."""
    drug_id: str
    drug_name: str
    company: str
    relevance_score: float = Field(ge=0, le=1)


class DrugSearchOutput(BaseModel):
    """Output schema for drug_search tool."""
    results: list[DrugSearchResult]
    count: int


# ============================================================================
# DRUG CONTEXT TOOL
# ============================================================================

class DrugContextInput(BaseModel):
    """Input schema for drug_context tool."""
    query: str = Field(..., description="Pergunta ou nome do medicamento")
    n_results: int = Field(default=5, ge=1, le=20, description="Número de chunks de contexto")
    mode: Literal["patient", "professional"] = Field(default="patient")


class ContextChunk(BaseModel):
    """Single context chunk from vector store."""
    content: str
    metadata: dict
    relevance: float = Field(ge=0, le=1)


class DrugContextOutput(BaseModel):
    """Output schema for drug_context tool."""
    context: list[ContextChunk]
    was_updated: bool = Field(description="Se dados foram atualizados on-demand")


# ============================================================================
# INTERACTION CHECK TOOL
# ============================================================================

class InteractionCheckInput(BaseModel):
    """Input schema for interaction_check tool."""
    drugs: list[str] = Field(..., min_length=2, max_length=10, description="Lista de medicamentos")


class DrugInteraction(BaseModel):
    """Single drug interaction."""
    drugs: list[str]
    severity: Literal["leve", "moderada", "grave"]
    description: str


class InteractionCheckOutput(BaseModel):
    """Output schema for interaction_check tool."""
    drugs_checked: list[str]
    interactions: list[DrugInteraction]
    has_severe: bool
    recommendations: str
    context_found: int


# ============================================================================
# DRUG SUMMARY TOOL
# ============================================================================

class DrugSummaryInput(BaseModel):
    """Input schema for drug_summary tool."""
    drug_name: str = Field(..., description="Nome do medicamento")
    sections: Optional[list[str]] = Field(default=None, description="Seções específicas")
    mode: Literal["patient", "professional"] = Field(default="patient")


class DrugSummaryOutput(BaseModel):
    """Output schema for drug_summary tool."""
    found: bool
    drug_name: str
    full_text: Optional[str] = None
    sections_requested: list[str]
    mode: str
    was_updated: bool = False


# ============================================================================
# ANVISA FETCH TOOL
# ============================================================================

class AnvisaFetchInput(BaseModel):
    """Input schema for anvisa_fetch tool."""
    drug_name: str = Field(..., description="Nome do medicamento")
    force_update: bool = Field(default=False, description="Forçar atualização do cache")


class AnvisaFetchOutput(BaseModel):
    """Output schema for anvisa_fetch tool."""
    found: bool
    drug_name: str
    drug_id: Optional[str] = None
    company: Optional[str] = None
    has_content: bool = False
    cached: bool = False
    error: Optional[str] = None


# ============================================================================
# QUICK ANSWER TOOL
# ============================================================================

class QuickAnswerInput(BaseModel):
    """Input schema for quick_answer tool."""
    question: str = Field(..., description="Pergunta do usuário")
    drug_name: Optional[str] = Field(default=None, description="Nome do medicamento específico")


class QuickAnswerOutput(BaseModel):
    """Output schema for quick_answer tool."""
    found: bool
    answer: Optional[str] = None
    context: Optional[str] = None
    confidence: Literal["high", "medium", "low"]
    source: Optional[str] = None


# ============================================================================
# CLINICAL PROTOCOL TOOL (PCDT from CONITEC)
# ============================================================================

class ClinicalProtocolInput(BaseModel):
    """Input schema for clinical_protocol tool."""
    query: str = Field(..., description="Doença ou condição de saúde")
    max_results: int = Field(default=5, ge=1, le=20, description="Número máximo de protocolos")


class ProtocolResult(BaseModel):
    """Single clinical protocol result."""
    id: str
    name: str
    disease: str
    pdf_url: Optional[str] = None
    portaria: Optional[str] = None


class ClinicalProtocolOutput(BaseModel):
    """Output schema for clinical_protocol tool."""
    found: bool
    query: str
    protocols: list[ProtocolResult]
    count: int
    source: str = "CONITEC/Ministério da Saúde"


# ============================================================================
# SCHEMA MAPPINGS
# ============================================================================

TOOL_INPUT_SCHEMAS = {
    "drug_search": DrugSearchInput,
    "drug_context": DrugContextInput,
    "interaction_check": InteractionCheckInput,
    "drug_summary": DrugSummaryInput,
    "anvisa_fetch": AnvisaFetchInput,
    "quick_answer": QuickAnswerInput,
    "clinical_protocol": ClinicalProtocolInput,
}

TOOL_OUTPUT_SCHEMAS = {
    "drug_search": DrugSearchOutput,
    "drug_context": DrugContextOutput,
    "interaction_check": InteractionCheckOutput,
    "drug_summary": DrugSummaryOutput,
    "anvisa_fetch": AnvisaFetchOutput,
    "quick_answer": QuickAnswerOutput,
    "clinical_protocol": ClinicalProtocolOutput,
}


def get_input_schema(tool_id: str) -> type[BaseModel] | None:
    """Get the input schema class for a tool."""
    return TOOL_INPUT_SCHEMAS.get(tool_id)


def get_output_schema(tool_id: str) -> type[BaseModel] | None:
    """Get the output schema class for a tool."""
    return TOOL_OUTPUT_SCHEMAS.get(tool_id)

