"""
Shared package for Bulário RAG System.

Contains reusable components across all frameworks:
- schemas: Data models (Message, Document, Judgment, Response)
- interfaces: Abstract base classes (BaseRAG, BaseJudge)
- prompts: Prompt templates
- judges: Judge implementations
"""

from src.shared.schemas import Message, MessageRole, Document, RAGResponse
from src.shared.interfaces import BaseRAG, BaseJudge

__all__ = [
    "Message",
    "MessageRole", 
    "Document",
    "RAGResponse",
    "BaseRAG",
    "BaseJudge",
]
