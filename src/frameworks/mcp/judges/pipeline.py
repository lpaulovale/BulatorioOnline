"""
MCP Judge Pipeline.

Orchestrates all MCP judges and aggregates results.
"""

import asyncio
import logging
from typing import List, Dict, Any

from src.shared.schemas.judges import (
    AggregatedJudgment,
    JudgmentDecision,
    JUDGE_WEIGHTS
)
from src.frameworks.mcp.judges.safety import MCPSafetyJudge
from src.frameworks.mcp.judges.quality import MCPQualityJudge
from src.frameworks.mcp.judges.source import MCPSourceJudge
from src.frameworks.mcp.judges.format import MCPFormatJudge

logger = logging.getLogger(__name__)


class MCPJudgePipeline:
    """
    MCP Judge Pipeline using Anthropic Claude.
    
    Runs all judges in parallel and aggregates results.
    """
    
    def __init__(self):
        self.safety_judge = MCPSafetyJudge()
        self.quality_judge = MCPQualityJudge()
        self.source_judge = MCPSourceJudge()
        self.format_judge = MCPFormatJudge()
    
    async def evaluate(
        self,
        user_query: str,
        generated_response: str,
        retrieved_documents: List[Dict[str, Any]],
        mode: str = "patient"
    ) -> AggregatedJudgment:
        """Run all judges and aggregate results."""
        
        # Run judges in parallel
        safety_task = self.safety_judge.evaluate(
            user_query, generated_response, retrieved_documents, mode
        )
        quality_task = self.quality_judge.evaluate(
            user_query, generated_response, retrieved_documents, mode
        )
        source_task = self.source_judge.evaluate(
            user_query, generated_response, retrieved_documents, mode
        )
        format_task = self.format_judge.evaluate(
            user_query, generated_response, retrieved_documents, mode
        )
        
        safety, quality, source, format_r = await asyncio.gather(
            safety_task, quality_task, source_task, format_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(safety, Exception):
            logger.error(f"Safety judge failed: {safety}")
            safety = None
        if isinstance(quality, Exception):
            logger.error(f"Quality judge failed: {quality}")
            quality = None
        if isinstance(source, Exception):
            logger.error(f"Source judge failed: {source}")
            source = None
        if isinstance(format_r, Exception):
            logger.error(f"Format judge failed: {format_r}")
            format_r = None
        
        # Calculate scores
        scores = {}
        if safety:
            scores["safety"] = safety.safety_score
        if quality:
            scores["quality"] = quality.quality_score
        if source:
            scores["attribution"] = source.attribution_score
        if format_r:
            scores["format"] = format_r.format_score
        
        # Weighted average
        overall = 0
        total_weight = 0
        for key, score in scores.items():
            weight = JUDGE_WEIGHTS.get(key, 0.1)
            overall += score * weight
            total_weight += weight
        
        overall_score = int(overall / total_weight) if total_weight > 0 else 70
        
        # Determine decision
        if safety and not safety.approved:
            decision = JudgmentDecision.REJECTED
        elif overall_score >= 80:
            decision = JudgmentDecision.APPROVED
        elif overall_score >= 60:
            decision = JudgmentDecision.APPROVED_WITH_CAVEATS
        else:
            decision = JudgmentDecision.NEEDS_REVISION
        
        # Collect disclaimers
        disclaimers = []
        if safety and safety.required_disclaimers:
            disclaimers.extend(safety.required_disclaimers)
        
        return AggregatedJudgment(
            final_decision=decision,
            overall_score=overall_score,
            score_breakdown=scores,
            blocking_issues=[],
            required_actions=[],
            disclaimers_to_add=disclaimers,
            revision_suggestions=[],
            final_response=None,
            confidence=overall_score / 100
        )
