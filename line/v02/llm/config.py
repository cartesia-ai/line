"""LLM configuration. See README.md for examples."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from line.call_request import CallRequest

# Default values used when CallRequest doesn't specify them
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly and helpful assistant powered by Cartesia. Have a natural conversation with the user."
)
DEFAULT_INTRODUCTION = "Hello! I'm your Cartesia assistant. How can I help you today?"


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
        default_system_prompt: Optional[str] = None,
        default_introduction: Optional[str] = None,
        **kwargs: Any,
    ) -> "LlmConfig":
        """
        Create LlmConfig from a CallRequest with sensible defaults.

        Priority (highest to lowest):
        1. CallRequest value (if not None)
        2. User-provided default (default_system_prompt / default_introduction)
        3. SDK default (DEFAULT_SYSTEM_PROMPT / DEFAULT_INTRODUCTION)

        Args:
            call_request: The CallRequest containing agent configuration
            default_system_prompt: Custom default if CallRequest doesn't specify one
            default_introduction: Custom default if CallRequest doesn't specify one
            **kwargs: Additional LlmConfig options (temperature, max_tokens, etc.)

        Note:
            If CallRequest sends empty string "" for introduction, it is preserved
            (agent waits for user to speak first).

        Example:
            # Use SDK defaults
            config = LlmConfig.from_call_request(call_request)

            # Use custom defaults (overridden by CallRequest if set)
            config = LlmConfig.from_call_request(
                call_request,
                default_system_prompt="You are a sales assistant.",
                default_introduction="Hi! How can I help with your purchase?",
                temperature=0.7,
            )
        """
        # Priority: call_request > user default > SDK default
        if call_request.agent.system_prompt is not None:
            system_prompt = call_request.agent.system_prompt
        elif default_system_prompt is not None:
            system_prompt = default_system_prompt
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        if call_request.agent.introduction is not None:
            introduction = call_request.agent.introduction
        elif default_introduction is not None:
            introduction = default_introduction
        else:
            introduction = DEFAULT_INTRODUCTION

        return cls(
            system_prompt=system_prompt,
            introduction=introduction,
            **kwargs,
        )
