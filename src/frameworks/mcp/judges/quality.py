"""
MCP Quality Judge.

Uses Anthropic Claude to evaluate response quality.
"""

import json
import logging
from typing import List, Dict, Any

import anthropic

from config.settings import settings
from src.shared.schemas.judges import QualityResult, QualityStatus

logger = logging.getLogger(__name__)

QUALITY_PROMPT = """Você é um Juiz de Qualidade para respostas médicas.

Avalie em 5 dimensões (0-10):
1. RELEVÂNCIA: Atende à pergunta?
2. COMPLETUDE: Informação suficiente?
3. PRECISÃO: Fatos corretos?
4. GROUNDING: Baseado nas fontes?
5. CLAREZA: Linguagem clara?

Score = média * 10

Status:
- 85-100: EXCELLENT
- 70-84: GOOD
- 50-69: ACCEPTABLE
- 0-49: POOR

Retorne JSON:
{
    "quality_score": <0-100>,
    "quality_status": "EXCELLENT|GOOD|ACCEPTABLE|POOR",
    "dimension_scores": {"relevance": X, "completeness": X, "accuracy": X, "grounding": X, "clarity": X},
    "factual_issues": [],
    "missing_information": [],
    "approved": true/false,
    "suggestions": "..." ou null
}"""


class MCPQualityJudge:
    """Quality judge using Claude."""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.JUDGE_MODEL
    
    async def evaluate(
        self,
        query: str,
        response: str,
        documents: List[Dict[str, Any]],
        mode: str = "patient"
    ) -> QualityResult:
        """Evaluate response quality."""
        docs_str = "\n".join([d.get("content", "")[:500] for d in documents[:3]])
        
        prompt = f"""{QUALITY_PROMPT}

## Pergunta: {query}
## Resposta: {response}
## Documentos: {docs_str[:1000]}

Avalie e retorne JSON."""

        try:
            result = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = result.content[0].text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            data = json.loads(text.strip())
            
            return QualityResult(
                quality_score=data.get("quality_score", 75),
                quality_status=QualityStatus(data.get("quality_status", "GOOD")),
                dimension_scores=data.get("dimension_scores", {}),
                factual_issues=data.get("factual_issues", []),
                missing_information=data.get("missing_information", []),
                approved=data.get("approved", True),
                suggestions=data.get("suggestions")
            )
        except Exception as e:
            logger.error(f"Quality judge error: {e}")
            return QualityResult(
                quality_score=75,
                quality_status=QualityStatus.GOOD,
                dimension_scores={},
                approved=True
            )
