"""
Source Attribution Judge prompt.
"""

SOURCE_JUDGE_PROMPT = """Você é um Juiz de Atribuição de Fontes para informações médicas.

Sua tarefa é verificar se CADA afirmação factual na resposta pode ser rastreada às fontes oficiais.

TIPOS DE GROUNDING:
- EXACT: Citação exata ou muito próxima da fonte
- PARAPHRASED: Paráfrase fiel que mantém o sentido
- INFERRED: Inferência razoável baseada na fonte
- UNSUPPORTED: Não encontrado nas fontes fornecidas

Attribution Score = (claims atribuídos / total de claims) * 100

Retorne JSON:
{{
    "attribution_score": <0-100>,
    "total_claims": <número>,
    "attributed_claims": <número>,
    "unattributed_claims": <número>,
    "claim_analysis": [
        {{
            "claim": "afirmação extraída",
            "source_document": "id ou nome do documento",
            "source_excerpt": "trecho relevante",
            "grounding_quality": "EXACT|PARAPHRASED|INFERRED|UNSUPPORTED"
        }}
    ],
    "unsupported_claims": [
        {{"claim": "afirmação sem suporte", "risk_level": "HIGH|MEDIUM|LOW"}}
    ],
    "approved": true/false
}}"""


def get_source_judge_prompt(response: str, documents_str: str) -> str:
    """Get formatted source judge prompt."""
    return f"""{SOURCE_JUDGE_PROMPT}

## Resposta Gerada
{response}

## Documentos de Referência
{documents_str}

Analise a atribuição de fontes e retorne JSON estruturado."""
