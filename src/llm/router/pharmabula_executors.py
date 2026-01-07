"""
PharmaBula Tool Executors

Implements the actual execution logic for each PharmaBula tool.
These executors bridge the RouterAgent with existing services.
"""

import logging
from typing import Optional

from src.database.vector_store import get_vector_store
from src.database.metadata_cache import get_metadata_cache
from src.services.drug_service import get_drug_service
from src.llm.prompts import build_rag_prompt, build_interaction_check_prompt

logger = logging.getLogger(__name__)


async def drug_search_executor(
    query: str,
    n_results: int = 5,
    **kwargs
) -> dict:
    """
    Search for drugs in the vector store.
    
    Args:
        query: Search term
        n_results: Number of results to return
        
    Returns:
        Dict with results and count
    """
    vector_store = get_vector_store()
    
    results = vector_store.search(query, n_results=n_results)
    
    formatted_results = []
    seen_drugs = set()
    
    for item in results:
        metadata = item.get("metadata", {})
        drug_id = metadata.get("drug_id", "")
        
        # Deduplicate by drug_id
        if drug_id not in seen_drugs:
            seen_drugs.add(drug_id)
            formatted_results.append({
                "drug_id": drug_id,
                "drug_name": metadata.get("drug_name", "Desconhecido"),
                "company": metadata.get("company", ""),
                "relevance_score": 1 - item.get("distance", 0)
            })
    
    return {
        "results": formatted_results,
        "count": len(formatted_results)
    }


async def drug_context_executor(
    query: str,
    n_results: int = 5,
    mode: str = "patient",
    **kwargs
) -> dict:
    """
    Get drug context for RAG using the hybrid DrugDataService.
    
    Args:
        query: Question or drug name
        n_results: Number of context chunks
        mode: Response mode (patient/professional)
        
    Returns:
        Dict with context and update status
    """
    drug_service = get_drug_service()
    
    context, was_updated = await drug_service.get_drug_context(
        query, n_results=n_results
    )
    
    # Format context for use
    formatted_context = []
    for item in context:
        formatted_context.append({
            "content": item.get("content", ""),
            "metadata": item.get("metadata", {}),
            "relevance": 1 - item.get("distance", 0)
        })
    
    return {
        "context": formatted_context,
        "was_updated": was_updated
    }


async def interaction_check_executor(
    drugs: list[str],
    **kwargs
) -> dict:
    """
    Check for drug interactions.
    
    Args:
        drugs: List of drug names (minimum 2)
        
    Returns:
        Dict with interactions and recommendations
    """
    if len(drugs) < 2:
        return {
            "error": "Mínimo de 2 medicamentos necessário",
            "interactions": [],
            "has_severe": False
        }
    
    drug_service = get_drug_service()
    
    # Get context for each drug
    all_context = []
    for drug in drugs:
        context, _ = await drug_service.get_drug_context(
            f"interações medicamentosas {drug}",
            n_results=2
        )
        all_context.extend(context)
    
    # Build interaction prompt data
    prompt_data = build_interaction_check_prompt(drugs)
    
    return {
        "drugs_checked": drugs,
        "context_found": len(all_context),
        "prompt_data": prompt_data,
        "interactions": [],  # To be filled by LLM
        "has_severe": False,
        "recommendations": "Consulte um profissional de saúde para análise completa."
    }


async def drug_summary_executor(
    drug_name: str,
    sections: list[str] = None,
    mode: str = "patient",
    **kwargs
) -> dict:
    """
    Get summary context for a drug's bulletin.
    
    Args:
        drug_name: Name of the drug
        sections: Specific sections to include
        mode: Response mode
        
    Returns:
        Dict with summary context
    """
    drug_service = get_drug_service()
    
    # Get extensive context for summary
    context, was_updated = await drug_service.get_drug_context(
        drug_name, n_results=10
    )
    
    if not context:
        return {
            "found": False,
            "drug_name": drug_name,
            "summary": None
        }
    
    # Combine all context
    full_text = "\n\n".join([c.get("content", "") for c in context])
    
    return {
        "found": True,
        "drug_name": drug_name,
        "full_text": full_text[:8000],  # Limit for token constraints
        "sections_requested": sections or ["all"],
        "mode": mode,
        "was_updated": was_updated
    }


async def anvisa_fetch_executor(
    drug_name: str,
    force_update: bool = False,
    **kwargs
) -> dict:
    """
    Fetch fresh data from ANVISA API.
    
    Args:
        drug_name: Drug name to search
        force_update: Whether to force cache update
        
    Returns:
        Dict with fetch results
    """
    from src.scrapers.anvisa_scraper import AnvisaScraper
    
    try:
        async with AnvisaScraper() as scraper:
            results = await scraper.search_drugs(drug_name, page_size=1)
            
            if results:
                drug_data = results[0]
                drug_id = str(drug_data.get("idProduto", ""))
                
                if drug_id:
                    bulletin = await scraper.fetch_and_process_bulletin(drug_id)
                    
                    if bulletin:
                        return {
                            "found": True,
                            "drug_id": drug_id,
                            "drug_name": bulletin.name,
                            "company": bulletin.company,
                            "has_content": bool(bulletin.text_content),
                            "cached": False
                        }
            
            return {
                "found": False,
                "drug_name": drug_name,
                "error": "Medicamento não encontrado na ANVISA"
            }
            
    except Exception as e:
        logger.error(f"ANVISA fetch error: {e}")
        return {
            "found": False,
            "drug_name": drug_name,
            "error": str(e)
        }


async def quick_answer_executor(
    question: str,
    drug_name: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Provide quick answers using cached context.
    
    Args:
        question: User's question
        drug_name: Optional specific drug name
        
    Returns:
        Dict with answer data
    """
    drug_service = get_drug_service()
    
    # Use drug_name if provided, otherwise extract from question
    search_query = drug_name or question
    
    context, _ = await drug_service.get_drug_context(
        search_query, n_results=3
    )
    
    if not context:
        return {
            "answer": None,
            "confidence": "low",
            "source": None,
            "found": False
        }
    
    top_result = context[0]
    
    return {
        "answer": None,  # To be generated by LLM
        "context": top_result.get("content", "")[:1000],
        "confidence": "high" if len(context) > 1 else "medium",
        "source": top_result.get("metadata", {}).get("drug_name", "Unknown"),
        "found": True
    }
async def clinical_protocol_executor(
    query: str,
    max_results: int = 5,
    **kwargs
) -> dict:
    """
    Search for clinical protocols (PCDT) from CONITEC.
    
    Args:
        query: Disease or condition name
        max_results: Maximum number of protocols
        
    Returns:
        Dict with matching protocols
    """
    from src.scrapers.conitec_scraper import CONITECScraper
    
    try:
        async with CONITECScraper() as scraper:
            protocols = await scraper.search_protocols(query, max_results=max_results)
            
            if protocols:
                return {
                    "found": True,
                    "query": query,
                    "protocols": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "disease": p.disease,
                            "pdf_url": p.pdf_url,
                            "portaria": p.portaria_number or ""
                        }
                        for p in protocols
                    ],
                    "count": len(protocols),
                    "source": "CONITEC/Ministério da Saúde"
                }
            
            return {
                "found": False,
                "query": query,
                "protocols": [],
                "count": 0,
                "source": "CONITEC/Ministério da Saúde"
            }
            
    except Exception as e:
        logger.error(f"CONITEC fetch error: {e}")
        return {
            "found": False,
            "query": query,
            "protocols": [],
            "count": 0,
            "error": str(e),
            "source": "CONITEC/Ministério da Saúde"
        }


# Dictionary mapping tool IDs to executor functions
PHARMABULA_EXECUTORS = {
    "drug_search": drug_search_executor,
    "drug_context": drug_context_executor,
    "interaction_check": interaction_check_executor,
    "drug_summary": drug_summary_executor,
    "anvisa_fetch": anvisa_fetch_executor,
    "quick_answer": quick_answer_executor,
    "clinical_protocol": clinical_protocol_executor,
}


def get_pharmabula_executors() -> dict:
    """Get all PharmaBula tool executors."""
    return PHARMABULA_EXECUTORS.copy()

