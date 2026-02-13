"""LLM configuration. See README.md for examples."""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from line.voice_agent_app import CallRequest

# Fallback values used when CallRequest doesn't specify them
FALLBACK_SYSTEM_PROMPT = (
    "You are a friendly and helpful assistant. Have a natural conversation with the user."
)
FALLBACK_INTRODUCTION = "Hello! I'm your AI assistant. How can I help you today?"


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
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None

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


def _merge_configs(base: LlmConfig, override: LlmConfig) -> LlmConfig:
    """Create a new LlmConfig by merging override values onto a base config.

    Non-default values in the override take precedence over values in the base.
    Fields in the override that are left at their default value are ignored,
    preserving the corresponding value from the base.
    """
    merged_kwargs = {}
    for f in dataclasses.fields(LlmConfig):
        base_val = getattr(base, f.name)
        override_val = getattr(override, f.name)
        # Determine the field's default value
        if f.default is not dataclasses.MISSING:
            default_val = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default_val = f.default_factory()
        else:
            # No default defined; always prefer the override
            merged_kwargs[f.name] = override_val
            continue
        # Use override value when it differs from the default
        if override_val != default_val:
            merged_kwargs[f.name] = override_val
        else:
            merged_kwargs[f.name] = base_val
    return LlmConfig(**merged_kwargs)
