"""
Document models for Bulário RAG.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Document:
    """
    Generic document for vector store retrieval.
    """
    id: str
    content: str
    source: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Format document for context."""
        header = f"### {self.source}" if self.source else "### Documento"
        return f"{header}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_langchain(cls, doc, score: float = 0.0) -> "Document":
        """Create from LangChain Document."""
        return cls(
            id=doc.metadata.get("id", ""),
            content=doc.page_content,
            source=doc.metadata.get("source", ""),
            score=score,
            metadata=doc.metadata
        )


@dataclass
class BulaDocument(Document):
    """
    Specialized document for drug bulletins (bulas).
    """
    medicamento: str = ""
    principio_ativo: str = ""
    fabricante: str = ""
    tipo_bula: str = "ambos"  # profissional, paciente, ambos
    secao: str = ""
    
    def to_string(self) -> str:
        """Format bula document for context."""
        header = f"### {self.medicamento}"
        if self.secao:
            header += f" - {self.secao}"
        
        lines = [header]
        if self.principio_ativo:
            lines.append(f"Princípio Ativo: {self.principio_ativo}")
        if self.fabricante:
            lines.append(f"Fabricante: {self.fabricante}")
        if self.tipo_bula != "ambos":
            lines.append(f"Tipo: Bula para {self.tipo_bula}")
        if self.source:
            lines.append(f"Fonte: {self.source}")
        
        lines.append("")
        lines.append(self.content)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "medicamento": self.medicamento,
            "principio_ativo": self.principio_ativo,
            "fabricante": self.fabricante,
            "tipo_bula": self.tipo_bula,
            "secao": self.secao
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulaDocument":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
            medicamento=data.get("medicamento", ""),
            principio_ativo=data.get("principio_ativo", ""),
            fabricante=data.get("fabricante", ""),
            tipo_bula=data.get("tipo_bula", "ambos"),
            secao=data.get("secao", "")
        )


def format_documents(docs: List[Document]) -> str:
    """Format list of documents for context."""
    if not docs:
        return "Nenhum documento encontrado."
    
    return "\n\n---\n\n".join([doc.to_string() for doc in docs])
