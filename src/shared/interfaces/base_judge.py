"""
Base Judge interface that all judge implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.shared.schemas.document import Document
from src.shared.schemas.judgment import JudgeScore


class BaseJudge(ABC):
    """
    Abstract base class for judge implementations.
    
    Each judge type (Safety, Quality, Source, Format) implements this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Judge name identifier."""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Weight for aggregation (0-1)."""
        pass
    
    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response: str,
        documents: List[Document],
        mode: str = "patient"
    ) -> JudgeScore:
        """
        Evaluate a response.
        
        Args:
            query: Original user query
            response: Generated response to evaluate
            documents: Retrieved documents used for context
            mode: Response mode (patient/professional)
        
        Returns:
            JudgeScore with evaluation results
        """
        pass
    
    def get_prompt(self, **kwargs) -> str:
        """
        Get the prompt template for this judge.
        
        Override in subclass to provide custom prompt.
        """
        raise NotImplementedError("Subclass must implement get_prompt()")


class BaseJudgePipeline(ABC):
    """
    Abstract base class for judge pipeline.
    
    Orchestrates multiple judges and aggregates results.
    """
    
    @abstractmethod
    async def evaluate(
        self,
        user_query: str,
        generated_response: str,
        retrieved_documents: List[Document],
        mode: str = "patient"
    ) -> Dict[str, Any]:
        """
        Run all judges and aggregate results.
        
        Args:
            user_query: Original user query
            generated_response: Response to evaluate
            retrieved_documents: Documents used for context
            mode: Response mode
        
        Returns:
            Aggregated judgment result
        """
        pass
