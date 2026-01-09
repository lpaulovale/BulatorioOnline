"""
Shared prompts package.

Centralized prompt templates used by all frameworks.
"""

from src.shared.prompts.generator import (
    SYSTEM_PROMPT,
    GENERATOR_PROMPT,
    get_system_prompt,
    get_generator_prompt
)
from src.shared.prompts.safety_judge import SAFETY_JUDGE_PROMPT, get_safety_judge_prompt
from src.shared.prompts.quality_judge import QUALITY_JUDGE_PROMPT, get_quality_judge_prompt
from src.shared.prompts.source_judge import SOURCE_JUDGE_PROMPT, get_source_judge_prompt
from src.shared.prompts.format_judge import FORMAT_JUDGE_PROMPT, get_format_judge_prompt

__all__ = [
    "SYSTEM_PROMPT",
    "GENERATOR_PROMPT", 
    "get_system_prompt",
    "get_generator_prompt",
    "SAFETY_JUDGE_PROMPT",
    "get_safety_judge_prompt",
    "QUALITY_JUDGE_PROMPT",
    "get_quality_judge_prompt",
    "SOURCE_JUDGE_PROMPT",
    "get_source_judge_prompt",
    "FORMAT_JUDGE_PROMPT",
    "get_format_judge_prompt",
]
