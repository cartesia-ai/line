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

    # Provider-specific pass-through
    extra: Dict[str, Any] = field(default_factory=dict)
