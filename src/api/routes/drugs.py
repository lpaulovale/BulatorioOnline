"""
Drug Information API Routes for PharmaBula

Provides endpoints for searching and retrieving drug information.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.database.metadata_cache import get_metadata_cache
from src.database.vector_store import get_vector_store
from src.llm.gemini_client import get_gemini_client

router = APIRouter(prefix="/api/drugs", tags=["Drugs"])


class DrugSearchResult(BaseModel):
    """A single drug search result."""
    
    drug_id: str
    drug_name: str
    company: Optional[str] = None
    active_ingredient: Optional[str] = None
    relevance_score: float = Field(ge=0, le=1)


class DrugSearchResponse(BaseModel):
    """Response for drug search."""
    
    query: str
    results: list[DrugSearchResult]
    total: int


class DrugDetails(BaseModel):
    """Detailed drug information."""
    
    drug_id: str
    drug_name: str
    company: Optional[str] = None
    active_ingredient: Optional[str] = None
    summary: Optional[str] = None
    is_indexed: bool


class DrugSummaryResponse(BaseModel):
    """Response for drug summary."""
    
    drug_name: str
    summary: str


@router.get("/search", response_model=DrugSearchResponse)
async def search_drugs(
    q: str = Query(
        ...,
        min_length=2,
        max_length=100,
        description="Search query (drug name or active ingredient)"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results"
    )
) -> DrugSearchResponse:
    """
    Search for drugs by name or active ingredient.
    
    Returns a list of matching drugs with relevance scores.
    """
    vector_store = get_vector_store()
    
    # Search in vector store
    results = vector_store.search(q, n_results=limit)
    
    # Deduplicate by drug_id and format results
    seen_drugs = set()
    search_results = []
    
    for result in results:
        drug_id = result["metadata"].get("drug_id", "")
        if drug_id and drug_id not in seen_drugs:
            seen_drugs.add(drug_id)
            
            # Convert distance to relevance score (1 - distance for cosine)
            distance = result.get("distance", 0)
            relevance = max(0, 1 - distance)
            
            search_results.append(DrugSearchResult(
                drug_id=drug_id,
                drug_name=result["metadata"].get("drug_name", "Desconhecido"),
                company=result["metadata"].get("company"),
                active_ingredient=result["metadata"].get("active_ingredient"),
                relevance_score=round(relevance, 3)
            ))
    
    return DrugSearchResponse(
        query=q,
        results=search_results,
        total=len(search_results)
    )


@router.get("/{drug_id}", response_model=DrugDetails)
async def get_drug(drug_id: str) -> DrugDetails:
    """
    Get detailed information about a specific drug.
    
    Returns cached metadata and indicates if the drug is fully indexed.
    """
    cache = get_metadata_cache()
    drug = cache.get_drug(drug_id)
    
    if not drug:
        raise HTTPException(
            status_code=404,
            detail=f"Medicamento com ID '{drug_id}' não encontrado"
        )
    
    return DrugDetails(
        drug_id=drug["drug_id"],
        drug_name=drug["drug_name"],
        company=drug.get("company"),
        active_ingredient=drug.get("active_ingredient"),
        summary=None,  # Lazy-loaded via /summary endpoint
        is_indexed=drug.get("is_indexed", False)
    )


@router.get("/{drug_id}/summary", response_model=DrugSummaryResponse)
async def get_drug_summary(drug_id: str) -> DrugSummaryResponse:
    """
    Get an AI-generated summary of a drug's bulletin.
    
    This uses the LLM to create a structured summary from the indexed content.
    """
    cache = get_metadata_cache()
    drug = cache.get_drug(drug_id)
    
    if not drug:
        raise HTTPException(
            status_code=404,
            detail=f"Medicamento com ID '{drug_id}' não encontrado"
        )
    
    if not drug.get("is_indexed"):
        raise HTTPException(
            status_code=404,
            detail="Informações detalhadas ainda não disponíveis para este medicamento"
        )
    
    try:
        client = get_gemini_client()
        summary = await client.get_drug_summary(drug["drug_name"])
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail="Não foi possível gerar o resumo"
            )
        
        return DrugSummaryResponse(
            drug_name=drug["drug_name"],
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar resumo: {str(e)}"
        )


@router.get("/stats/overview")
async def get_stats():
    """
    Get database statistics.
    
    Returns information about indexed drugs and system health.
    """
    cache = get_metadata_cache()
    vector_store = get_vector_store()
    
    stats = cache.get_stats()
    
    return {
        "total_drugs": stats["total_drugs"],
        "indexed_drugs": stats["indexed_drugs"],
        "pending_indexing": stats["pending_indexing"],
        "vector_documents": vector_store.count(),
        "last_scrape": stats.get("last_scrape")
    }
