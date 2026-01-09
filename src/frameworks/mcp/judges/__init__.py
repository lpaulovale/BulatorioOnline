"""
MCP Judges Package.

Self-contained judge implementation using Anthropic Claude.
"""

from src.frameworks.mcp.judges.pipeline import MCPJudgePipeline
from src.frameworks.mcp.judges.safety import MCPSafetyJudge
from src.frameworks.mcp.judges.quality import MCPQualityJudge
from src.frameworks.mcp.judges.source import MCPSourceJudge
from src.frameworks.mcp.judges.format import MCPFormatJudge

__all__ = [
    "MCPJudgePipeline",
    "MCPSafetyJudge",
    "MCPQualityJudge",
    "MCPSourceJudge",
    "MCPFormatJudge",
]
