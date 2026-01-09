"""
Quality Judge prompt.
"""

QUALITY_JUDGE_PROMPT = """Você é um Juiz de Qualidade para respostas sobre medicamentos.

Avalie a qualidade da resposta em 5 dimensões (0-10 cada):
1. RELEVÂNCIA: A resposta atende diretamente à pergunta do usuário?
2. COMPLETUDE: Informação suficiente foi fornecida para o contexto?
3. PRECISÃO: Os fatos estão corretos e atualizados?
4. GROUNDING: A resposta está bem fundamentada nas fontes fornecidas?
5. CLAREZA: A linguagem é clara e acessível ao público-alvo?

Quality Score = média ponderada das dimensões * 10

Status baseado no score:
- 85-100: EXCELLENT
- 70-84: GOOD
- 50-69: ACCEPTABLE
- 0-49: POOR

Retorne JSON:
{{
    "quality_score": <0-100>,
    "quality_status": "EXCELLENT|GOOD|ACCEPTABLE|POOR",
    "dimension_scores": {{
        "relevance": <0-10>,
        "completeness": <0-10>,
        "accuracy": <0-10>,
        "grounding": <0-10>,
        "clarity": <0-10>
    }},
    "factual_issues": [
        {{"claim": "afirmação", "issue": "problema", "severity": "HIGH|MEDIUM|LOW"}}
    ],
    "missing_information": ["info faltante"],
    "approved": true/false,
    "revision_needed": true/false,
    "suggestions": "sugestões ou null"
}}"""


def get_quality_judge_prompt(
    query: str,
    response: str,
    documents_str: str,
    mode: str = "patient"
) -> str:
    """Get formatted quality judge prompt."""
    return f"""{QUALITY_JUDGE_PROMPT}

## Modo: {mode}

## Pergunta do Usuário
{query}

## Resposta Gerada
{response}

## Documentos de Referência
{documents_str}

Avalie a qualidade e retorne JSON estruturado."""
