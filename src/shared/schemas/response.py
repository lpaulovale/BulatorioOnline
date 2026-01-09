"""
RAG Response model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from src.shared.schemas.judgment import JudgmentResult


@dataclass
class RAGResponse:
    """
    Complete RAG response with judgment.
    """
    query: str
    answer: str
    judgment: Optional[JudgmentResult] = None
    sources: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: str = "média"  # alta, média, baixa
    disclaimer: str = ""
    
    # Performance metrics
    latency_ms: float = 0.0
    token_count: int = 0
    
    # Framework info
    framework: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "judgment": self.judgment.to_dict() if self.judgment else None,
            "sources": self.sources,
            "confidence": self.confidence,
            "disclaimer": self.disclaimer,
            "latency_ms": self.latency_ms,
            "token_count": self.token_count,
            "framework": self.framework
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGResponse":
        judgment = None
        if data.get("judgment"):
            judgment = JudgmentResult.from_dict(data["judgment"])
        
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            judgment=judgment,
            sources=data.get("sources", []),
            confidence=data.get("confidence", "média"),
            disclaimer=data.get("disclaimer", ""),
            latency_ms=data.get("latency_ms", 0.0),
            token_count=data.get("token_count", 0),
            framework=data.get("framework", "")
        )
    
    def to_json_response(self) -> Dict[str, Any]:
        """Format as JSON response for API."""
        return {
            "response": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "disclaimer": self.disclaimer
        }
