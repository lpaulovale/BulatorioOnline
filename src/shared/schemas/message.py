"""
Unified Message model for all frameworks.

Provides conversion methods for OpenAI, Anthropic, and LangChain formats.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    Unified message that works with MCP, LangChain, and OpenAI.
    
    Provides conversion methods for each framework's format.
    """
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # Metadata for context management
    priority: int = 1  # 0=critical, 1=high, 2=medium, 3=low
    entities: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    token_count: int = 0
    
    # Framework-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai(self) -> Dict[str, str]:
        """Convert to OpenAI chat message format."""
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    def to_anthropic(self) -> Dict[str, str]:
        """Convert to Anthropic message format."""
        return {
            "role": self.role.value,
            "content": self.content
        }
    
    def to_langchain(self):
        """Convert to LangChain message type."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        if self.role == MessageRole.USER:
            return HumanMessage(content=self.content)
        elif self.role == MessageRole.ASSISTANT:
            return AIMessage(content=self.content)
        else:
            return SystemMessage(content=self.content)
    
    @classmethod
    def from_openai(cls, msg: Dict[str, str]) -> "Message":
        """Create from OpenAI format."""
        return cls(
            role=MessageRole(msg["role"]),
            content=msg["content"]
        )
    
    @classmethod
    def from_langchain(cls, msg) -> "Message":
        """Create from LangChain message."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        if isinstance(msg, HumanMessage):
            role = MessageRole.USER
        elif isinstance(msg, AIMessage):
            role = MessageRole.ASSISTANT
        else:
            role = MessageRole.SYSTEM
        
        return cls(role=role, content=msg.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "entities": self.entities,
            "token_count": self.token_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            priority=data.get("priority", 1),
            entities=data.get("entities", []),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationHistory:
    """Manages conversation history with framework agnostic interface."""
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 10
    
    def add(self, role: MessageRole, content: str, **kwargs) -> None:
        """Add a message to history."""
        self.messages.append(Message(role=role, content=content, **kwargs))
        self._trim()
    
    def add_user(self, content: str, **kwargs) -> None:
        """Add user message."""
        self.add(MessageRole.USER, content, **kwargs)
    
    def add_assistant(self, content: str, **kwargs) -> None:
        """Add assistant message."""
        self.add(MessageRole.ASSISTANT, content, **kwargs)
    
    def _trim(self) -> None:
        """Keep only max_messages most recent."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_openai_messages(self) -> List[Dict[str, str]]:
        """Get messages in OpenAI format."""
        return [m.to_openai() for m in self.messages]
    
    def get_anthropic_messages(self) -> List[Dict[str, str]]:
        """Get messages in Anthropic format."""
        return [m.to_anthropic() for m in self.messages]
    
    def get_langchain_messages(self) -> List:
        """Get messages in LangChain format."""
        return [m.to_langchain() for m in self.messages]
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
