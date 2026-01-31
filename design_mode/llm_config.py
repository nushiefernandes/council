from __future__ import annotations

"""LLM provider configuration for design mode.

Returns CrewAI-compatible LLM configurations matching the main council.
"""

from functools import lru_cache
from typing import Any, Literal

from crewai import LLM


ProviderName = Literal["claude", "gpt", "deepseek"]


@lru_cache(maxsize=8)
def get_llm_for_provider(provider: ProviderName) -> Any:
    """Return a cached LLM configuration for the given provider.

    Uses the same models as the main council:
    - Claude: claude-opus-4-5-20251101
    - GPT: gpt-5.2
    - DeepSeek: deepseek-coder-v2:16b via Ollama
    """
    if provider == "claude":
        return "anthropic/claude-opus-4-5-20251101"
    elif provider == "gpt":
        return "openai/gpt-5.2"
    elif provider == "deepseek":
        return LLM(model="ollama/deepseek-coder-v2:16b", base_url="http://localhost:11434")
    else:
        # Default to Claude for unknown providers
        return "anthropic/claude-opus-4-5-20251101"
