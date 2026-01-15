"""
LLM configuration classes for the Line SDK.

This module provides configuration classes for LLM agents, including
model settings, generation parameters, and provider-specific options.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LlmConfig:
    """
    Configuration for LLM agents.

    This configuration is provider-agnostic and will be converted to
    provider-specific formats by the LLM providers.

    Attributes:
        system_prompt: The system prompt/instructions for the LLM.
        introduction: Optional initial message to send when conversation starts.
        temperature: Sampling temperature (0.0 to 2.0). Higher = more creative.
        max_tokens: Maximum number of tokens to generate.
        thinking_budget: For models with thinking/reasoning, the token budget.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter (mainly for Gemini).
        stop_sequences: Sequences that will stop generation.
        presence_penalty: Penalize new tokens based on presence in text so far.
        frequency_penalty: Penalize new tokens based on frequency in text so far.
        extra: Provider-specific extra parameters.

    Example:
        ```python
        config = LlmConfig(
            system_prompt="You are a helpful assistant...",
            introduction="Hello! How can I help you today?",
            temperature=0.7,
            max_tokens=1000,
        )

        agent = LlmAgent(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            config=config,
        )
        ```
    """

    system_prompt: str = ""
    introduction: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """Convert to OpenAI API kwargs."""
        kwargs: Dict[str, Any] = {}

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty

        kwargs.update(self.extra)
        return kwargs

    def to_anthropic_kwargs(self) -> Dict[str, Any]:
        """Convert to Anthropic API kwargs."""
        kwargs: Dict[str, Any] = {}

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences
        if self.thinking_budget is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}

        kwargs.update(self.extra)
        return kwargs

    def to_gemini_kwargs(self) -> Dict[str, Any]:
        """Convert to Gemini API kwargs."""
        kwargs: Dict[str, Any] = {}

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty

        kwargs.update(self.extra)
        return kwargs


# Common model aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    # OpenAI
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o1-preview": "o1-preview",
    # Anthropic
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "opus-4.5": "claude-opus-4-5-20251101",
    # Google
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-1.5-pro",
}


def resolve_model_alias(model: str) -> str:
    """
    Resolve a model alias to its full model ID.

    Args:
        model: Model name or alias.

    Returns:
        The full model ID.
    """
    return MODEL_ALIASES.get(model, model)


def detect_provider(model: str) -> str:
    """
    Detect the provider from a model name.

    Args:
        model: Model name or alias.

    Returns:
        Provider name: "openai", "anthropic", or "google".

    Raises:
        ValueError: If provider cannot be determined.
    """
    model_lower = model.lower()

    # OpenAI models
    if any(
        prefix in model_lower for prefix in ["gpt-", "o1", "davinci", "curie", "babbage", "ada"]
    ):
        return "openai"

    # Anthropic models
    if "claude" in model_lower or "opus" in model_lower or "sonnet" in model_lower:
        return "anthropic"

    # Google models
    if "gemini" in model_lower or "palm" in model_lower:
        return "google"

    raise ValueError(
        f"Cannot determine provider for model '{model}'. "
        "Please use the provider parameter explicitly."
    )
