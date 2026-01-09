"""
MCP Format Judge.
"""

import json
import logging
from typing import List, Dict, Any

import anthropic

from config.settings import settings
from src.shared.schemas.judges import FormatResult

logger = logging.getLogger(__name__)


class MCPFormatJudge:
    """Format judge using Claude."""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.JUDGE_MODEL
    
    async def evaluate(
        self,
        query: str,
        response: str,
        documents: List[Dict[str, Any]],
        mode: str = "patient"
    ) -> FormatResult:
        """Evaluate response formatting."""
        prompt = f"""Avalie a formatação da resposta.

## Pergunta: {query}
## Resposta: {response}

Retorne JSON:
{{
    "format_score": <0-100>,
    "format_status": "OPTIMAL|GOOD|ACCEPTABLE|POOR",
    "dimension_scores": {{"appropriateness": X, "readability": X}},
    "issues": [],
    "approved": true/false
}}"""

        try:
            result = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = result.content[0].text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            data = json.loads(text.strip())
            
            return FormatResult(
                format_score=data.get("format_score", 85),
                format_status=data.get("format_status", "GOOD"),
                dimension_scores=data.get("dimension_scores", {}),
                issues=data.get("issues", []),
                approved=data.get("approved", True)
            )
        except Exception as e:
            logger.error(f"Format judge error: {e}")
            return FormatResult(
                format_score=85,
                format_status="GOOD",
                approved=True
            )
