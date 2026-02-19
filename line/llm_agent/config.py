"""LLM configuration. See README.md for examples."""

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from line.voice_agent_app import CallRequest

# Sentinel to distinguish "field not passed" from "field explicitly set to None/default".
_UNSET: Any = object()


@dataclass
class LlmConfig:
    """
    Configuration for LLM agents.  Passed to LiteLLM.

    All fields default to ``_UNSET``, meaning "not specified".  Use
    :func:`_merge_configs` to layer an override config onto a base config and
    resolve every sentinel to its real default value.

    See https://docs.litellm.ai/docs/completion/input for full documentation.
    """

    # Agent behavior
    system_prompt: str = _UNSET
    introduction: Optional[str] = _UNSET  # Sent on CallStarted; None or "" = skip

    # Sampling
    temperature: Optional[float] = _UNSET
    max_tokens: Optional[int] = _UNSET
    top_p: Optional[float] = _UNSET
    stop: Optional[List[str]] = _UNSET
    seed: Optional[int] = _UNSET
    reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = _UNSET

    # Penalties
    presence_penalty: Optional[float] = _UNSET
    frequency_penalty: Optional[float] = _UNSET

    # Resilience
    num_retries: int = _UNSET
    fallbacks: Optional[List[str]] = _UNSET
    timeout: Optional[float] = _UNSET

    # Provider-specific pass-through
    extra: Dict[str, Any] = _UNSET

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


# Fallback values used when CallRequest doesn't specify them
FALLBACK_SYSTEM_PROMPT = (
    "You are a friendly and helpful assistant. Have a natural conversation with the user."
)
FALLBACK_INTRODUCTION = "Hello! I'm your AI assistant. How can I help you today?"

# Real defaults for each LlmConfig field.  Callables (e.g. ``dict``) are
# invoked to produce a fresh value each time so mutable defaults are safe.
_FIELD_DEFAULTS: Dict[str, Any] = {
    "system_prompt": "",
    "introduction": None,
    "temperature": None,
    "max_tokens": None,
    "top_p": None,
    "stop": None,
    "seed": None,
    "reasoning_effort": None,
    "presence_penalty": None,
    "frequency_penalty": None,
    "num_retries": 2,
    "fallbacks": None,
    "timeout": None,
    "extra": dict,  # callable → invoked each time
}


def _field_default(name: str) -> Any:
    """Return the real default for *name*, invoking factories as needed."""
    val = _FIELD_DEFAULTS[name]
    return val() if callable(val) else val


def _merge_configs(base: LlmConfig, override: LlmConfig) -> LlmConfig:
    """Create a fully-resolved LlmConfig by merging *override* onto *base*.

    For each field the last non-``_UNSET`` value wins, checked in order:

    1. The field's real default (from ``_FIELD_DEFAULTS``)
    2. *base* value (if explicitly set)
    3. *override* value (if explicitly set)

    """
    merged_kwargs = {}
    for f in dataclasses.fields(LlmConfig):
        override_val = getattr(override, f.name)
        base_val = getattr(base, f.name)
        default_val = _field_default(f.name)
        if override_val is not _UNSET:
            merged_kwargs[f.name] = override_val
        elif base_val is not _UNSET:
            merged_kwargs[f.name] = base_val
        else:
            merged_kwargs[f.name] = default_val
    return LlmConfig(**merged_kwargs)


def _normalize_config(config: LlmConfig) -> LlmConfig:
    """Normalize an LlmConfig by replacing any remaining ``_UNSET`` with real defaults."""
    normalized_kwargs = {}
    for f in dataclasses.fields(LlmConfig):
        val = getattr(config, f.name)
        if val is _UNSET:
            normalized_kwargs[f.name] = _field_default(f.name)
        else:
            normalized_kwargs[f.name] = val
    return LlmConfig(**normalized_kwargs)
