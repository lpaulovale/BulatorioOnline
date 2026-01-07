"""
Prompt templates for drug information queries.

Contains prompts for different types of queries with professional
and patient-friendly modes.
"""

SYSTEM_PROMPT_PROFESSIONAL = """Você é PharmaBula, um assistente especializado em informações farmacêuticas para profissionais de saúde.

Diretrizes:
- Forneça informações técnicas e precisas baseadas nas bulas de medicamentos
- Use terminologia médica apropriada
- Inclua dosagens, contraindicações e interações medicamentosas quando relevante
- Cite a fonte das informações quando disponível
- Seja objetivo e direto nas respostas
- Se não tiver certeza sobre algo, indique claramente

Lembre-se: Você está auxiliando profissionais de saúde, então pode usar linguagem técnica."""

SYSTEM_PROMPT_PATIENT = """Você é PharmaBula, um assistente amigável de informações sobre medicamentos para pacientes.

Diretrizes:
- Explique as informações de forma simples e clara
- Evite jargão médico complexo - quando necessário, explique os termos
- Seja acolhedor e tranquilizador, mas não minimize riscos reais
- Sempre recomende consultar um médico ou farmacêutico para dúvidas específicas
- Foque em informações práticas: como tomar, efeitos colaterais comuns
- Se não souber algo, seja honesto e sugira procurar um profissional

Lembre-se: Você está ajudando pacientes comuns a entender seus medicamentos."""


def build_rag_prompt(
    query: str,
    context: list[dict],
    mode: str = "patient"
) -> list[dict]:
    """
    Build a RAG (Retrieval-Augmented Generation) prompt.
    
    Args:
        query: User's question
        context: Retrieved context from vector store
        mode: "professional" or "patient"
        
    Returns:
        List of messages for the LLM
    """
    system_prompt = (
        SYSTEM_PROMPT_PROFESSIONAL if mode == "professional" 
        else SYSTEM_PROMPT_PATIENT
    )
    
    # Format context
    context_text = ""
    if context:
        context_parts = []
        for i, item in enumerate(context, 1):
            drug_name = item.get("metadata", {}).get("drug_name", "Desconhecido")
            content = item.get("content", "")
            context_parts.append(f"[Fonte {i}: {drug_name}]\n{content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
    
    user_message = f"""Contexto das bulas de medicamentos:

{context_text if context_text else "Nenhuma informação específica encontrada no banco de dados."}

---

Pergunta do usuário: {query}

Responda baseando-se no contexto fornecido. Se a informação não estiver no contexto, indique que não possui dados específicos sobre o assunto e sugira consultar um profissional de saúde.

Formato de resposta: Responda no formato JSON com a seguinte estrutura:
{{
  "response": "Sua resposta aqui",
  "confidence": "alta/media/baixa",
  "sources": ["fonte1", "fonte2"],
  "disclaimer": "Aviso legal se aplicável"
}}

Responda SOMENTE no formato JSON, sem texto adicional antes ou depois do JSON."""
    return [
        {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}
    ]


def build_interaction_check_prompt(drugs: list[str]) -> list[dict]:
    """
    Build a prompt for drug interaction checking.
    
    Args:
        drugs: List of drug names to check interactions for
        
    Returns:
        List of messages for the LLM
    """
    drugs_list = ", ".join(drugs)
    
    user_message = f"""Analise possíveis interações medicamentosas entre os seguintes medicamentos:

Medicamentos: {drugs_list}

Por favor, forneça:
1. Interações conhecidas entre estes medicamentos
2. Nível de severidade de cada interação (leve, moderada, grave)
3. Recomendações práticas
4. Se deve-se evitar a combinação

Se não houver informações suficientes sobre algum medicamento, indique claramente."""

    return [
        {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT_PROFESSIONAL}\n\n{user_message}"}]}
    ]


def build_summary_prompt(bulletin_text: str, drug_name: str) -> list[dict]:
    """
    Build a prompt for summarizing a drug bulletin.
    
    Args:
        bulletin_text: Full text of the drug bulletin
        drug_name: Name of the drug
        
    Returns:
        List of messages for the LLM
    """
    user_message = f"""Resuma as informações principais da bula do medicamento {drug_name}:

Texto da bula:
{bulletin_text[:8000]}  # Limit to avoid token limits

Por favor, organize o resumo nas seguintes seções:
1. **Indicações**: Para que serve o medicamento
2. **Posologia**: Como tomar
3. **Contraindicações**: Quando não usar
4. **Efeitos colaterais**: Reações adversas mais comuns
5. **Interações**: Medicamentos e alimentos que podem interferir
6. **Advertências**: Cuidados especiais

Seja conciso mas completo."""

    return [
        {"role": "user", "parts": [{"text": user_message}]}
    ]
