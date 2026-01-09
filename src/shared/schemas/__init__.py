"""
Shared schemas for PharmaBula.
"""

from src.shared.schemas.message import Message, MessageRole, ConversationHistory
from src.shared.schemas.document import Document, BulaDocument
from src.shared.schemas.response import RAGResponse
from src.shared.schemas.judgment import JudgmentResult, JudgmentDecision
from src.shared.schemas.judges import (
    SafetyResult,
    QualityResult,
    SourceResult,
    FormatResult,
    AggregatedJudgment,
    JudgeContext,
    JUDGE_WEIGHTS,
)

__all__ = [
    "Message",
    "MessageRole",
    "ConversationHistory",
    "Document",
    "BulaDocument",
    "RAGResponse",
    "JudgmentResult",
    "JudgmentDecision",
    "SafetyResult",
    "QualityResult",
    "SourceResult",
    "FormatResult",
    "AggregatedJudgment",
    "JudgeContext",
    "JUDGE_WEIGHTS",
]
