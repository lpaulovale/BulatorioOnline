"""
Google Gemini Client for PharmaBula

Handles all interactions with the Gemini API for drug information queries.
Uses the hybrid DrugDataService for on-demand data updates.
Optionally uses RouterAgent for intelligent tool selection and execution.
"""

import logging
from typing import Optional

import google.generativeai as genai

from src.config import get_settings
from src.llm.prompts import build_rag_prompt, build_interaction_check_prompt

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Provides methods for querying drug information with RAG support.
    Uses hybrid data service for on-demand updates.
    Optionally routes requests through RouterAgent for intelligent tool selection.
    """
    
    def __init__(self, use_router: bool = False):
        """
        Initialize the Gemini client with API key from settings.
        
        Args:
            use_router: Enable RouterAgent for intelligent tool selection
        """
        settings = get_settings()
        
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured. "
                "Please set it in your .env file."
            )
        
        genai.configure(api_key=settings.gemini_api_key)
        
        self.model = genai.GenerativeModel(settings.gemini_model)
        self.use_router = use_router
        self._router = None
        self._executors = None
        
        # Import here to avoid circular imports
        from src.services.drug_service import get_drug_service
        self.drug_service = get_drug_service()
        
        logger.info(f"Gemini client initialized with model: {settings.gemini_model}")
        if use_router:
            logger.info("RouterAgent enabled for intelligent tool selection")
    
    def _get_router(self):
        """Lazy-load the RouterAgent and executors."""
        if self._router is None:
            from src.llm.router import RouterAgent
            from src.llm.router.pharmabula_registry import get_cached_pharmabula_registry
            from src.llm.router.pharmabula_executors import get_pharmabula_executors
            
            self._router = RouterAgent(
                tool_registry=get_cached_pharmabula_registry()
            )
            self._executors = get_pharmabula_executors()
        return self._router, self._executors
    
    async def query(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5,
        priority: str = "medium",
        preferences: dict = None
    ) -> str:
        """
        Answer a drug-related question using RAG with on-demand updates.
        
        This method can operate in two modes:
        1. Direct RAG (default): Uses vector store context + LLM
        2. Router mode: Uses RouterAgent for intelligent tool selection
        
        Args:
            question: User's question about medications
            mode: Response mode - "professional" or "patient"
            n_context: Number of context chunks to retrieve
            priority: Request priority (low/medium/high/urgent)
            preferences: Optional user preferences dict
            
        Returns:
            Generated response from Gemini
        """
        # Use router if enabled
        if self.use_router:
            return await self._query_with_router(question, mode, priority, preferences)
        
        # Direct RAG approach
        return await self._query_direct(question, mode, n_context)
    
    async def _query_direct(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5
    ) -> str:
        """Direct RAG query without router."""
        # Get context with on-demand update capability
        context, was_updated = await self.drug_service.get_drug_context(
            question, n_results=n_context
        )
        
        if was_updated:
            logger.info("Drug data was updated for this query")
        
        # Build prompt with context
        messages = build_rag_prompt(question, context, mode)
        
        try:
            # Configure the model to return JSON format
            response = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            return response.text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to plain text if JSON generation fails
            try:
                response = self.model.generate_content(messages)
                return response.text
            except Exception:
                return (
                    "Desculpe, ocorreu um erro ao processar sua pergunta. "
                    "Por favor, tente novamente."
                )
    
    async def _query_with_router(
        self,
        question: str,
        mode: str = "patient",
        priority: str = "medium",
        preferences: dict = None
    ) -> str:
        """Query using RouterAgent for intelligent tool selection."""
        from src.llm.router.schemas import Priority, UserPreferences, CostTier
        
        router, executors = self._get_router()
        
        # Map string priority to enum
        priority_map = {
            "low": Priority.LOW,
            "medium": Priority.MEDIUM,
            "high": Priority.HIGH,
            "urgent": Priority.URGENT
        }
        priority_enum = priority_map.get(priority, Priority.MEDIUM)
        
        # Build user preferences
        prefs = preferences or {}
        user_prefs = UserPreferences(
            prefer_speed=prefs.get("prefer_speed", False),
            cost_sensitivity=CostTier(prefs.get("cost_sensitivity", "medium")),
            custom={"mode": mode}
        )
        
        try:
            # Route the request
            decision = router.route_request(
                message=question,
                priority=priority_enum,
                preferences=user_prefs,
                additional_context={"mode": mode}
            )
            
            logger.info(
                f"Routed to: {[t.tool_id for t in decision.selected_tools]}, "
                f"confidence: {decision.confidence.value}"
            )
            
            # Execute the plan
            result = await router.execute_plan(
                decision=decision,
                tool_executors=executors,
                aggregate_results=True
            )
            
            if result.final_response:
                return result.final_response
            
            # Fallback to direct query if router fails
            logger.warning("Router execution failed, falling back to direct query")
            return await self._query_direct(question, mode)
            
        except Exception as e:
            logger.error(f"Router query error: {e}")
            # Fallback to direct query
            return await self._query_direct(question, mode)
    
    async def check_interactions(self, drugs: list[str]) -> str:
        """
        Check for drug interactions with on-demand data fetching.
        
        Args:
            drugs: List of drug names to check
            
        Returns:
            Analysis of potential interactions
        """
        if len(drugs) < 2:
            return "Por favor, forneça pelo menos dois medicamentos para verificar interações."
        
        # Get context for each drug (with on-demand updates)
        all_context = []
        for drug in drugs:
            context, _ = await self.drug_service.get_drug_context(
                f"interações medicamentosas {drug}",
                n_results=2
            )
            all_context.extend(context)
        
        # Build interaction prompt
        messages = build_interaction_check_prompt(drugs)
        
        # Add context if available
        if all_context:
            context_text = "\n\n".join([c["content"] for c in all_context])
            messages[0]["parts"][0]["text"] += f"\n\nContexto adicional:\n{context_text}"
        
        try:
            response = self.model.generate_content(messages)
            return response.text
            
        except Exception as e:
            logger.error(f"Error checking interactions: {e}")
            return (
                "Não foi possível verificar as interações. "
                "Consulte um farmacêutico ou médico."
            )
    
    async def get_drug_summary(self, drug_name: str) -> Optional[str]:
        """
        Get a summary of a drug's bulletin with on-demand fetching.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Summary of the drug's information or None if not found
        """
        # Get context with on-demand update
        context, was_updated = await self.drug_service.get_drug_context(
            drug_name, n_results=10
        )
        
        if not context:
            return None
        
        if was_updated:
            logger.info(f"Fetched fresh data for {drug_name}")
        
        # Combine all chunks for this drug
        full_text = "\n\n".join([c["content"] for c in context])
        
        prompt = f"""Faça um resumo completo das informações do medicamento {drug_name} 
baseado no seguinte texto da bula:

{full_text}

Organize em seções: Indicações, Posologia, Contraindicações, Efeitos colaterais, Interações."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None


# Singleton instance
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the global Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client

