"""
Judge Schemas.

Shared Pydantic models for judge inputs and outputs.
Used by all framework judge implementations.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


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


class JudgmentDecision(str, Enum):
    """Final judgment decision."""
    APPROVED = "APPROVED"
    APPROVED_WITH_CAVEATS = "APPROVED_WITH_CAVEATS"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECTED = "REJECTED"


class CriticalIssue(BaseModel):
    """A critical issue identified by a judge."""
    issue: str
    severity: str = Field(description="CRITICAL, HIGH, or MEDIUM")
    category: str
    location: Optional[str] = None


class SafetyResult(BaseModel):
    """Result from Safety Judge."""
    safety_score: int = Field(ge=0, le=100)
    safety_status: SafetyStatus
    critical_issues: List[CriticalIssue] = []
    required_disclaimers: List[str] = []
    recommendations: Optional[str] = None
    approved: bool


class QualityResult(BaseModel):
    """Result from Quality Judge."""
    quality_score: int = Field(ge=0, le=100)
    quality_status: QualityStatus
    dimension_scores: Dict[str, int] = {}
    factual_issues: List[Dict[str, Any]] = []
    missing_information: List[str] = []
    approved: bool
    revision_needed: bool = False
    suggestions: Optional[str] = None


class SourceResult(BaseModel):
    """Result from Source Attribution Judge."""
    attribution_score: int = Field(ge=0, le=100)
    total_claims: int = 0
    attributed_claims: int = 0
    unattributed_claims: int = 0
    claim_analysis: List[Dict[str, Any]] = []
    unsupported_claims: List[Dict[str, Any]] = []
    approved: bool


class FormatResult(BaseModel):
    """Result from Format Judge."""
    format_score: int = Field(ge=0, le=100)
    format_status: str = "GOOD"
    dimension_scores: Dict[str, int] = {}
    issues: List[Dict[str, Any]] = []
    approved: bool


class AggregatedJudgment(BaseModel):
    """Final aggregated judgment from all judges."""
    final_decision: JudgmentDecision
    overall_score: int = Field(ge=0, le=100)
    score_breakdown: Dict[str, int] = {}
    blocking_issues: List[str] = []
    required_actions: List[Dict[str, Any]] = []
    disclaimers_to_add: List[str] = []
    revision_suggestions: List[str] = []
    final_response: Optional[str] = None
    confidence: float = 0.0


class JudgeContext(BaseModel):
    """Context passed to judges for evaluation."""
    user_query: str
    generated_response: str
    retrieved_documents: List[Dict[str, Any]] = []
    mode: str = "patient"


# Scoring weights
JUDGE_WEIGHTS = {
    "safety": 0.40,
    "quality": 0.30,
    "attribution": 0.20,
    "format": 0.10,
}
