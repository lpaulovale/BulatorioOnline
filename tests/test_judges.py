"""
Unit Tests for Judge Pipeline

Tests for individual judges and the pipeline orchestration.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json

from src.llm.judges.schemas import (
    SafetyResult,
    QualityResult,
    SourceResult,
    FormatResult,
    AggregatedJudgment,
    JudgmentDecision,
    SafetyStatus,
    QualityStatus,
    JudgeContext,
)
from src.llm.judges.aggregator import AggregatorJudge


class TestJudgeSchemas:
    """Tests for judge data schemas."""
    
    def test_safety_result_creation(self):
        """Test SafetyResult can be created with valid data."""
        result = SafetyResult(
            safety_score=85,
            safety_status=SafetyStatus.SAFE,
            critical_issues=[],
            required_disclaimers=["Test disclaimer"],
            approved=True
        )
        assert result.safety_score == 85
        assert result.safety_status == SafetyStatus.SAFE
        assert result.approved is True
    
    def test_quality_result_creation(self):
        """Test QualityResult can be created with valid data."""
        result = QualityResult(
            quality_score=78,
            quality_status=QualityStatus.GOOD,
            dimension_scores={
                "relevance": 8,
                "completeness": 7,
                "accuracy": 8,
                "grounding": 7,
                "clarity": 8
            },
            factual_issues=[],
            missing_information=[],
            approved=True
        )
        assert result.quality_score == 78
        assert result.quality_status == QualityStatus.GOOD
    
    def test_judge_context_creation(self):
        """Test JudgeContext can be created."""
        context = JudgeContext(
            user_query="Para que serve paracetamol?",
            generated_response="O paracetamol é usado para dor e febre.",
            retrieved_documents=[],
            mode="patient"
        )
        assert context.user_query == "Para que serve paracetamol?"
        assert context.mode == "patient"


class TestAggregatorJudge:
    """Tests for the Aggregator Judge."""
    
    @pytest.fixture
    def aggregator(self):
        return AggregatorJudge()
    
    @pytest.fixture
    def safe_results(self):
        """Create a set of passing judge results."""
        safety = SafetyResult(
            safety_score=90,
            safety_status=SafetyStatus.SAFE,
            critical_issues=[],
            required_disclaimers=[],
            approved=True
        )
        quality = QualityResult(
            quality_score=85,
            quality_status=QualityStatus.EXCELLENT,
            dimension_scores={
                "relevance": 9, "completeness": 8,
                "accuracy": 9, "grounding": 8, "clarity": 8
            },
            factual_issues=[],
            missing_information=[],
            approved=True
        )
        source = SourceResult(
            attribution_score=88,
            total_claims=5,
            attributed_claims=5,
            unattributed_claims=0,
            claim_analysis=[],
            unsupported_claims=[],
            approved=True
        )
        format_result = FormatResult(
            format_score=80,
            format_status="GOOD",
            dimension_scores={
                "appropriateness": 8, "logical_structure": 8,
                "readability": 8, "consistency": 8
            },
            issues=[],
            approved=True
        )
        return safety, quality, source, format_result
    
    def test_aggregate_approved(self, aggregator, safe_results):
        """Test that good results lead to APPROVED decision."""
        safety, quality, source, format_result = safe_results
        
        judgment = aggregator.aggregate(
            safety=safety,
            quality=quality,
            source=source,
            format_result=format_result,
            generated_response="Test response"
        )
        
        assert judgment.final_decision == JudgmentDecision.APPROVED
        assert judgment.overall_score >= 80
        assert judgment.final_response == "Test response"
    
    def test_aggregate_rejected_unsafe(self, aggregator, safe_results):
        """Test that unsafe response leads to REJECTED decision."""
        safety, quality, source, format_result = safe_results
        
        # Make safety UNSAFE
        safety.safety_status = SafetyStatus.UNSAFE
        safety.safety_score = 30
        
        judgment = aggregator.aggregate(
            safety=safety,
            quality=quality,
            source=source,
            format_result=format_result,
            generated_response="Test response"
        )
        
        assert judgment.final_decision == JudgmentDecision.REJECTED
    
    def test_aggregate_with_caveats(self, aggregator, safe_results):
        """Test that warning status leads to APPROVED_WITH_CAVEATS."""
        safety, quality, source, format_result = safe_results
        
        # Make safety WARNING
        safety.safety_status = SafetyStatus.WARNING
        safety.safety_score = 75
        
        judgment = aggregator.aggregate(
            safety=safety,
            quality=quality,
            source=source,
            format_result=format_result,
            generated_response="Test response"
        )
        
        assert judgment.final_decision == JudgmentDecision.APPROVED_WITH_CAVEATS
    
    def test_weighted_scoring(self, aggregator, safe_results):
        """Test that scores are properly weighted."""
        safety, quality, source, format_result = safe_results
        
        # Set specific scores
        safety.safety_score = 100
        quality.quality_score = 80
        source.attribution_score = 60
        format_result.format_score = 50
        
        judgment = aggregator.aggregate(
            safety=safety,
            quality=quality,
            source=source,
            format_result=format_result,
            generated_response="Test"
        )
        
        # Expected: 100*0.40 + 80*0.30 + 60*0.20 + 50*0.10 = 81
        expected = 100 * 0.40 + 80 * 0.30 + 60 * 0.20 + 50 * 0.10
        assert judgment.overall_score == int(expected)


class TestJudgePipeline:
    """Tests for the JudgePipeline orchestrator."""
    
    @pytest.fixture
    def mock_judges(self):
        """Create mock judges that return passing results."""
        with patch('src.llm.judges.pipeline.SafetyJudge') as mock_safety, \
             patch('src.llm.judges.pipeline.QualityJudge') as mock_quality, \
             patch('src.llm.judges.pipeline.SourceJudge') as mock_source, \
             patch('src.llm.judges.pipeline.FormatJudge') as mock_format:
            
            # Configure mock return values
            mock_safety.return_value.evaluate = AsyncMock(return_value=SafetyResult(
                safety_score=90, safety_status=SafetyStatus.SAFE,
                critical_issues=[], required_disclaimers=[], approved=True
            ))
            mock_quality.return_value.evaluate = AsyncMock(return_value=QualityResult(
                quality_score=85, quality_status=QualityStatus.EXCELLENT,
                dimension_scores={"relevance": 9, "completeness": 8, "accuracy": 9, "grounding": 8, "clarity": 8},
                factual_issues=[], missing_information=[], approved=True
            ))
            mock_source.return_value.evaluate = AsyncMock(return_value=SourceResult(
                attribution_score=88, total_claims=3, attributed_claims=3,
                unattributed_claims=0, claim_analysis=[], unsupported_claims=[], approved=True
            ))
            mock_format.return_value.evaluate = AsyncMock(return_value=FormatResult(
                format_score=80, format_status="GOOD",
                dimension_scores={"appropriateness": 8, "logical_structure": 8, "readability": 8, "consistency": 8},
                issues=[], approved=True
            ))
            
            yield mock_safety, mock_quality, mock_source, mock_format
    
    @pytest.mark.asyncio
    async def test_pipeline_approve(self, mock_judges):
        """Test that pipeline approves good responses."""
        from src.llm.judges.pipeline import JudgePipeline
        
        pipeline = JudgePipeline(max_retries=2)
        
        judgment = await pipeline.evaluate(
            user_query="Para que serve paracetamol?",
            generated_response='{"response": "O paracetamol é para dor e febre."}',
            retrieved_documents=[],
            mode="patient"
        )
        
        assert judgment.final_decision in [
            JudgmentDecision.APPROVED,
            JudgmentDecision.APPROVED_WITH_CAVEATS
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
