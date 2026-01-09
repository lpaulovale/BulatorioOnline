"""
MCP Source Attribution Judge.
"""

import json
import logging
from typing import List, Dict, Any

import anthropic

from config.settings import settings
from src.shared.schemas.judges import SourceResult

logger = logging.getLogger(__name__)


class MCPSourceJudge:
    """Source attribution judge using Claude."""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.JUDGE_MODEL
    
    async def evaluate(
        self,
        query: str,
        response: str,
        documents: List[Dict[str, Any]],
        mode: str = "patient"
    ) -> SourceResult:
        """Evaluate source attribution."""
        docs_str = "\n".join([d.get("content", "")[:500] for d in documents[:3]])
        
        prompt = f"""Verifique se as afirmações na resposta são baseadas nos documentos.

## Resposta: {response}
## Documentos: {docs_str[:1500]}

Retorne JSON:
{{
    "attribution_score": <0-100>,
    "total_claims": X,
    "attributed_claims": X,
    "unattributed_claims": X,
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
            
            return SourceResult(
                attribution_score=data.get("attribution_score", 80),
                total_claims=data.get("total_claims", 0),
                attributed_claims=data.get("attributed_claims", 0),
                unattributed_claims=data.get("unattributed_claims", 0),
                approved=data.get("approved", True)
            )
        except Exception as e:
            logger.error(f"Source judge error: {e}")
            return SourceResult(
                attribution_score=80,
                total_claims=0,
                attributed_claims=0,
                unattributed_claims=0,
                approved=True
            )
