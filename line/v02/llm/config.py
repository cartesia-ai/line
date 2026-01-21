"""LLM configuration. See README.md for examples."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LlmConfig:
    """
    Configuration for LLM agents. Passed to LiteLLM.

    See https://docs.litellm.ai/docs/completion/input for full documentation.
    """

    # Agent behavior
    system_prompt: str = ""
    introduction: Optional[str] = None  # Sent on CallStarted

    # Sampling
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    # Penalties
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    # Resilience
    num_retries: int = 2
    fallbacks: Optional[List[str]] = None
    timeout: Optional[float] = None

    # Web search (for supported models like gpt-4o-search-preview, grok-3, gemini-2.0-flash)
    # See https://docs.litellm.ai/docs/completion/web_search
    web_search_options: Optional[Dict[str, Any]] = None  # e.g. {"search_context_size": "medium"}

    # Provider-specific pass-through
    extra: Dict[str, Any] = field(default_factory=dict)
