"""
Judgment models for RAG evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class JudgmentDecision(str, Enum):
    """Final judgment decisions."""
    APPROVED = "approved"
    APPROVED_WITH_CAVEATS = "approved_with_caveats"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class SafetyStatus(str, Enum):
    """Safety evaluation status."""
    SAFE = "SAFE"
    WARNING = "WARNING"
    UNSAFE = "UNSAFE"


class QualityStatus(str, Enum):
    """Quality evaluation status."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"


@dataclass
class JudgeScore:
    """Individual judge score."""
    judge_name: str
    score: float  # 0-100
    status: str
    approved: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "score": self.score,
            "status": self.status,
            "approved": self.approved,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "details": self.details
        }


@dataclass
class JudgmentResult:
    """
    Aggregated judgment result from all judges.
    """
    final_decision: JudgmentDecision
    overall_score: float  # 0-100
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Individual judge results
    safety: Optional[JudgeScore] = None
    quality: Optional[JudgeScore] = None
    source: Optional[JudgeScore] = None
    format: Optional[JudgeScore] = None
    
    # Actions
    blocking_issues: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    disclaimers_to_add: List[str] = field(default_factory=list)
    revision_suggestions: List[str] = field(default_factory=list)
    
    # Final response (may be modified by judges)
    final_response: Optional[str] = None
    confidence: float = 0.0
    
    def is_approved(self) -> bool:
        """Check if response was approved."""
        return self.final_decision in [
            JudgmentDecision.APPROVED,
            JudgmentDecision.APPROVED_WITH_CAVEATS
        ]
    
    def needs_revision(self) -> bool:
        """Check if response needs revision."""
        return self.final_decision == JudgmentDecision.NEEDS_REVISION
    
    def is_rejected(self) -> bool:
        """Check if response was rejected."""
        return self.final_decision == JudgmentDecision.REJECTED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_decision": self.final_decision.value,
            "overall_score": self.overall_score,
            "score_breakdown": self.score_breakdown,
            "safety": self.safety.to_dict() if self.safety else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "source": self.source.to_dict() if self.source else None,
            "format": self.format.to_dict() if self.format else None,
            "blocking_issues": self.blocking_issues,
            "required_actions": self.required_actions,
            "disclaimers_to_add": self.disclaimers_to_add,
            "revision_suggestions": self.revision_suggestions,
            "final_response": self.final_response,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgmentResult":
        return cls(
            final_decision=JudgmentDecision(data["final_decision"]),
            overall_score=data["overall_score"],
            score_breakdown=data.get("score_breakdown", {}),
            blocking_issues=data.get("blocking_issues", []),
            required_actions=data.get("required_actions", []),
            disclaimers_to_add=data.get("disclaimers_to_add", []),
            revision_suggestions=data.get("revision_suggestions", []),
            final_response=data.get("final_response"),
            confidence=data.get("confidence", 0.0)
        )


# Judge weights for aggregation
JUDGE_WEIGHTS = {
    "safety": 0.40,
    "quality": 0.30,
    "source": 0.20,
    "format": 0.10
}
