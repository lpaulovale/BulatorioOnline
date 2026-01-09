"""
Chat API Routes for PharmaBula

Handles chat interactions with the drug information assistant.
"""

import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.frameworks.factory import get_rag

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    """Request model for chat messages."""
    
    message: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User's question about medications"
    )
    mode: Literal["professional", "patient"] = Field(
        default="patient",
        description="Response mode: 'professional' for healthcare workers, 'patient' for general public"
    )


class ChatResponse(BaseModel):
    """Response model for chat messages."""

    response: str = Field(description="AI assistant's response")
    mode: str = Field(description="Mode used for the response")
    sources: list[str] = Field(
        default=[],
        description="List of drug sources used in the response"
    )
    metadata: dict = Field(
        default={},
        description="Additional metadata from the response"
    )


class InteractionCheckRequest(BaseModel):
    """Request model for drug interaction check."""
    
    drugs: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of drug names to check for interactions"
    )


class InteractionCheckResponse(BaseModel):
    """Response model for drug interaction check."""
    
    analysis: str = Field(description="Analysis of potential drug interactions")
    drugs_checked: list[str] = Field(description="List of drugs that were analyzed")


@router.post("/", response_model=ChatResponse)
async def send_message(request: ChatMessage) -> ChatResponse:
    """
    Send a message to the PharmaBula assistant.

    The assistant will search for relevant drug information and provide
    a contextual response based on the selected mode.

    - **message**: Your question about medications
    - **mode**: 'patient' for simple explanations, 'professional' for technical details
    """
    try:
        client = get_rag()
        raw_response = await client.query(
            question=request.message,
            mode=request.mode
        )

        # Try to parse the response as JSON
        try:
            parsed_response = json.loads(raw_response)
            # If it's a valid JSON response, extract the response text
            if isinstance(parsed_response, dict):
                response_text = parsed_response.get("response", raw_response)
                metadata = parsed_response
            else:
                response_text = raw_response
                metadata = {}
        except json.JSONDecodeError:
            # If not valid JSON, use the raw response
            response_text = raw_response
            metadata = {}

        return ChatResponse(
            response=response_text,
            mode=request.mode,
            sources=[],  # Could be enhanced to return actual sources
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar sua mensagem: {str(e)}"
        )


@router.post("/interactions", response_model=InteractionCheckResponse)
async def check_interactions(request: InteractionCheckRequest) -> InteractionCheckResponse:
    """
    Check for potential drug interactions.
    
    Provide a list of drug names to analyze for possible interactions.
    The system will search its database and provide relevant warnings.
    
    - **drugs**: List of at least 2 drug names
    """
    try:
        client = get_rag()
        analysis = await client.check_interactions(request.drugs)
        
        return InteractionCheckResponse(
            analysis=analysis,
            drugs_checked=request.drugs
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao verificar interações: {str(e)}"
        )


@router.get("/health")
async def chat_health():
    """Health check for the chat service."""
    try:
        # Just verify client can be created
        get_rag()
        return {"status": "healthy", "service": "chat"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
