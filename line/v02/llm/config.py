"""
LLM configuration for the Line SDK.

This module provides the LlmConfig class for configuring LLM agents.
Configuration is passed to LiteLLM which handles provider-specific translation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LlmConfig:
    """
    Configuration for LLM agents using LiteLLM.

    LiteLLM translates these parameters to provider-specific formats automatically.
    See https://docs.litellm.ai/docs/completion/input for full parameter documentation.

    Attributes:
        system_prompt: The system prompt/instructions for the LLM.
        introduction: Optional initial message to send when conversation starts.

        # Sampling parameters
        temperature: Sampling temperature (0.0 to 2.0). Higher = more creative.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling parameter (0.0 to 1.0).
        stop: Sequences that will stop generation.
        seed: Random seed for reproducible outputs (if supported by provider).

        # Penalty parameters
        presence_penalty: Penalize tokens based on presence in text so far (-2.0 to 2.0).
        frequency_penalty: Penalize tokens based on frequency in text so far (-2.0 to 2.0).

        # LiteLLM parameters
        num_retries: Number of retries for failed requests.
        fallbacks: List of fallback models to try if primary fails.
        timeout: Request timeout in seconds.

        # Pass-through
        extra: Additional parameters passed directly to LiteLLM.

    Example:
        ```python
        config = LlmConfig(
            system_prompt="You are a helpful assistant.",
            introduction="Hello! How can I help you today?",
            temperature=0.7,
            max_tokens=1000,
            num_retries=3,
            fallbacks=["anthropic/claude-3-5-sonnet-20241022"],
        )

        agent = LlmAgent(model="gpt-4o", config=config)
        ```
    """

    # Agent behavior
    system_prompt: str = ""
    introduction: Optional[str] = None

    # Sampling parameters (passed to LiteLLM)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    # Penalty parameters
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    # LiteLLM-specific
    num_retries: int = 2
    fallbacks: Optional[List[str]] = None
    timeout: Optional[float] = None

    # Pass-through for provider-specific options
    extra: Dict[str, Any] = field(default_factory=dict)
