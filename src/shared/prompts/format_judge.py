"""
Format Judge prompt.
"""

FORMAT_JUDGE_PROMPT = """Você é um Juiz de Formatação para respostas sobre medicamentos.

PRINCÍPIO: Use formatação (headers, bullets, listas) APENAS quando necessário para melhorar a clareza.

Avalie em 4 dimensões (0-10 cada):
1. APROPRIAÇÃO: Formatação adequada à complexidade da pergunta
2. ESTRUTURA LÓGICA: Organização faz sentido e facilita a leitura
3. LEGIBILIDADE: Fácil de ler e entender
4. CONSISTÊNCIA: Estilo uniforme ao longo da resposta

Status baseado no score:
- 85-100: OPTIMAL
- 70-84: GOOD
- 50-69: ACCEPTABLE
- 0-49: POOR

Retorne JSON:
{{
    "format_score": <0-100>,
    "format_status": "OPTIMAL|GOOD|ACCEPTABLE|POOR",
    "dimension_scores": {{
        "appropriateness": <0-10>,
        "logical_structure": <0-10>,
        "readability": <0-10>,
        "consistency": <0-10>
    }},
    "issues": [
        {{"issue": "problema", "suggestion": "sugestão"}}
    ],
    "query_complexity": "simple|moderate|complex",
    "format_matches_complexity": true/false,
    "approved": true/false
}}"""


def get_format_judge_prompt(query: str, response: str) -> str:
    """Get formatted format judge prompt."""
    return f"""{FORMAT_JUDGE_PROMPT}

## Pergunta do Usuário
{query}

## Resposta Gerada
{response}

Avalie a formatação e retorne JSON estruturado."""
