"""
Base RAG interface that all framework implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.shared.schemas.message import Message
from src.shared.schemas.document import Document
from src.shared.schemas.response import RAGResponse


class BaseRAG(ABC):
    """
    Abstract base class for RAG implementations.
    
    All frameworks (MCP, LangChain, OpenAI) must implement this interface.
    """
    
    @abstractmethod
    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional filters (e.g., tipo_bula, medicamento)
        
        Returns:
            List of relevant documents
        """
        pass
    
    @abstractmethod
    async def generate_answer(
        self,
        query: str,
        documents: List[Document],
        conversation_history: Optional[List[Message]] = None
    ) -> RAGResponse:
        """
        Generate answer with judge pipeline evaluation.
        
        Args:
            query: User question
            documents: Retrieved documents for context
            conversation_history: Optional conversation history
        
        Returns:
            RAGResponse with answer and judgment
        """
        pass
    
    @abstractmethod
    async def add_message(self, message: Message) -> None:
        """
        Add a message to conversation history.
        
        Args:
            message: Message to add
        """
        pass
    
    @abstractmethod
    async def get_conversation_history(self) -> List[Message]:
        """
        Get current conversation history.
        
        Returns:
            List of messages in history
        """
        pass
    
    @abstractmethod
    async def clear_history(self) -> None:
        """Clear conversation history."""
        pass
    
    async def query(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5,
        **kwargs
    ) -> str:
        """
        Convenience method for full RAG query.
        
        Args:
            question: User question
            mode: Response mode (patient/professional)
            n_context: Number of context documents
        
        Returns:
            Answer string (JSON formatted)
        """
        # Search documents
        documents = await self.search_documents(question, top_k=n_context)
        
        # Get history
        history = await self.get_conversation_history()
        
        # Generate answer
        response = await self.generate_answer(question, documents, history)
        
        # Add to history
        from src.shared.schemas.message import MessageRole
        await self.add_message(Message(role=MessageRole.USER, content=question))
        await self.add_message(Message(role=MessageRole.ASSISTANT, content=response.answer))
        
        return response.answer
    
    async def check_interactions(self, drugs: List[str]) -> str:
        """
        Check drug interactions.
        
        Args:
            drugs: List of drug names to check
        
        Returns:
            JSON string with interaction information
        """
        import json
        
        if len(drugs) < 2:
            return json.dumps({"error": "Forneça pelo menos dois medicamentos."}, ensure_ascii=False)
        
        # Search for interaction info
        query = f"interações medicamentosas {' '.join(drugs)}"
        documents = await self.search_documents(query, top_k=5)
        
        # Generate interaction analysis
        response = await self.generate_answer(
            f"Verifique interações entre: {', '.join(drugs)}",
            documents
        )
        
        return response.answer
