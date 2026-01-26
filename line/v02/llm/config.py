"""LLM configuration. See README.md for examples."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from line.call_request import CallRequest

# Fallback values used when CallRequest doesn't specify them
FALLBACK_SYSTEM_PROMPT = (
    "You are a friendly and helpful assistant powered by Cartesia. Have a natural conversation with the user."
)
FALLBACK_INTRODUCTION = "Hello! I'm your Cartesia assistant. How can I help you today?"


@dataclass
class LlmConfig:
    """
    Configuration for LLM agents. Passed to LiteLLM.

    See https://docs.litellm.ai/docs/completion/input for full documentation.
    """

    # Agent behavior
    system_prompt: str = ""
    introduction: Optional[str] = None  # Sent on CallStarted; None or "" = skip

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

    @classmethod
    def from_call_request(
        cls,
        call_request: CallRequest,
        fallback_system_prompt: Optional[str] = None,
        fallback_introduction: Optional[str] = None,
        **kwargs: Any,
    ) -> "LlmConfig":
        """
        Create LlmConfig from a CallRequest with sensible defaults.

        Priority (highest to lowest):
        1. CallRequest value (if not None)
        2. User-provided fallback (fallback_system_prompt / fallback_introduction)
        3. SDK default (FALLBACK_SYSTEM_PROMPT / FALLBACK_INTRODUCTION)

        Args:
            call_request: The CallRequest containing agent configuration
            fallback_system_prompt: Custom fallback if CallRequest doesn't specify one
            fallback_introduction: Custom fallback if CallRequest doesn't specify one
            **kwargs: Additional LlmConfig options (temperature, max_tokens, etc.)

        Note:
            - system_prompt: Empty strings are treated as None (will use fallbacks).
              A valid system prompt is always required for proper agent behavior.
            - introduction: Empty strings ARE preserved (agent waits for user to speak first).

        Example:
            # Use SDK defaults
            config = LlmConfig.from_call_request(call_request)

            # Use custom fallbacks (overridden by CallRequest if set)
            config = LlmConfig.from_call_request(
                call_request,
                fallback_system_prompt="You are a sales assistant.",
                fallback_introduction="Hi! How can I help with your purchase?",
                temperature=0.7,
            )
        """
        # Priority: call_request > user fallback > SDK default
        # Note: Empty strings for system_prompt are treated as None (fall back to fallbacks)
        if call_request.agent.system_prompt:  # Truthiness check treats "" as None
            system_prompt = call_request.agent.system_prompt
        elif fallback_system_prompt:  # Also use truthiness for consistency
            system_prompt = fallback_system_prompt
        else:
            system_prompt = FALLBACK_SYSTEM_PROMPT

        if call_request.agent.introduction is not None:
            introduction = call_request.agent.introduction
        elif fallback_introduction is not None:
            introduction = fallback_introduction
        else:
            introduction = FALLBACK_INTRODUCTION

        return cls(
            system_prompt=system_prompt,
            introduction=introduction,
            **kwargs,
        )
