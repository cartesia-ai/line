"""LLM configuration. See README.md for examples."""

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Union

from line.voice_agent_app import CallRequest

# Sentinel to distinguish "field not passed" from "field explicitly set to None/default".
_UNSET: Any = object()


# Default detection pattern for SpeechLeakGuardConfig. Five alternatives, all safe
# to search mid-text (natural TTS speech contains none of them):
#   - ``{\s*"``            — a JSON object opening onto a quoted key
#   - ``ident\s*(\s*[{"]`` — a snake_case identifier called with a JSON/str
#     arg, e.g. ``record_call_summary({``. Requiring ``{`` or ``"`` right after
#     the paren keeps spoken oddities like ``name(at)domain`` from matching.
#   - ``to=\w+\.``          — the harmony tool-call recipient marker
#     (``to=functions.record_call_summary``); pure protocol syntax that
#     survives header corruption and appears well before the JSON payload.
#   - ``functions.<ident>``  — the harmony tool-namespace prefix glued to an
#     identifier (no space after the dot, so prose like "it functions. Also"
#     cannot match).
#   - ``<|``                 — harmony special-token delimiters leaking
#     literally (``<|constrain|>`` etc.); unambiguous non-speech.
# The header-artifact alternatives matter for the observed degraded form
# ``record_call_summary <glitch tokens> json {"summary":...`` where neither a
# paren nor a leading ``{`` exists — they pull detection to (near) offset 0 so
# nothing escapes the holdback window.
DEFAULT_SPEECH_LEAK_PATTERN = (
    r'\{\s*"'
    r"|[a-z_][a-z0-9_]*\s*\(\s*[{\"]"
    r"|to=\w+\."
    r"|\bfunctions\.[a-z_]"
    r"|<\|"
)

DEFAULT_SPEECH_LEAK_RETRY_NOTE = (
    "Your previous reply was discarded: it wrote a tool call (JSON arguments "
    "or a function-call expression) into the spoken message text. Tool calls "
    "must be made ONLY through the function-calling channel; the spoken "
    "message must contain only natural language for the caller (or nothing). "
    "Produce your reply again now: make any tool calls properly, and never "
    "include JSON, braces, field names, or tool syntax in the spoken text."
)

DEFAULT_SPEECH_LEAK_BRIDGE_TEXT = "Sorry, I'm having some technical issues. Let me try that again."

DEFAULT_SPEECH_LEAK_FALLBACK_TEXT = (
    "I'm sorry, I'm having some technical difficulties right now. Would you like me to try again?"
)


@dataclass
class SpeechLeakGuardConfig:
    """Guard spoken text against tool-call payloads leaking into TTS.

    Some reasoning models (observed: ``gpt-5.4-mini``) occasionally write a
    tool call's JSON arguments — or the whole call expression,
    ``record_call_summary({...})`` — into a user-visible ``message`` item
    instead of (or in addition to) the proper ``function_call`` item. Without
    a guard the caller hears raw JSON read out loud.

    When enabled, the ``http_responses`` provider buffers each spoken message
    item with ``lookahead_chars`` of holdback and scans the accumulated text
    for ``detection_pattern`` (``re.search``, so a leak is caught anywhere in
    the item, not just at its head). On a hit the current LLM invocation is
    aborted before any of its tool calls execute or its output is committed
    to history, and the whole invocation is retried up to ``max_retries``
    times with ``retry_note`` injected (via ``retry_note_channel``) and an
    optional ``retry_reasoning_effort`` override to decorrelate the retry
    from the failed attempt. If the caller already heard part of the failed
    attempt, ``bridge_text`` is spoken before the retry; if every attempt
    leaks, ``fallback_text`` is spoken and the turn ends with no tool calls.

    ``bridge_text`` and ``fallback_text`` are caller-audible, so they may be
    given as zero-arg callables instead of strings for agents whose spoken
    language isn't fixed at config time: the callable is invoked at speak
    time (each time it is needed) and returns the line to speak — return
    ``""`` to speak nothing. If the callable raises, the English default is
    spoken and the error logged; leak recovery must not crash. ``retry_note``
    stays a plain string — it is model-facing, and models follow English
    instructions regardless of the conversation language.

    Off by default; enable per agent/model. Only the ``http_responses``
    backend reads it.
    """

    enabled: bool = False
    # Which message-item phases are guarded. Leaks have only been observed on
    # ``commentary`` items; add ``"final_answer"`` to guard those too.
    phases: FrozenSet[str] = frozenset({"commentary"})
    # Chars of holdback between what has streamed in and what is released to
    # TTS. Must comfortably exceed the distance from a leak's start to its
    # first pattern-matchable signature. The degraded-header form (tool name +
    # unbounded glitch-token run + ``json {"``) put that signature at offset
    # ~67 in production, so 64 let 3 chars slip out; 128 covers every shape
    # observed so far. Cost is negligible: release lags generation (which far
    # outpaces speech), and clean items shorter than the window are flushed
    # whole the moment the item completes.
    lookahead_chars: int = 128
    detection_pattern: str = DEFAULT_SPEECH_LEAK_PATTERN
    max_retries: int = 1
    retry_note: str = DEFAULT_SPEECH_LEAK_RETRY_NOTE
    # "developer": append a developer-role input item (privileged, most
    # recent instruction). "instructions": append to the system instructions
    # for the retry request only.
    retry_note_channel: Literal["developer", "instructions"] = "developer"
    # Reasoning effort override for retry requests (e.g. "medium" when the
    # base config runs "minimal"). None keeps the original effort.
    retry_reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high"]] = None
    bridge_text: Union[str, Callable[[], str]] = DEFAULT_SPEECH_LEAK_BRIDGE_TEXT
    fallback_text: Union[str, Callable[[], str]] = DEFAULT_SPEECH_LEAK_FALLBACK_TEXT


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
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high"]] = _UNSET

    # Penalties
    presence_penalty: Optional[float] = _UNSET
    frequency_penalty: Optional[float] = _UNSET

    # Resilience
    num_retries: int = _UNSET
    fallbacks: Optional[List[str]] = _UNSET
    timeout: Optional[float] = _UNSET

    # Provider-specific pass-through
    extra: Dict[str, Any] = _UNSET

    # Tool schema settings
    strict_tool_schemas: bool = _UNSET

    # Zero Data Retention mode. Default False — the SDK sends ``store: true``
    # to OpenAI's Responses API and chains turns via ``previous_response_id``
    # for efficient continuation. Set True for OpenAI organizations with ZDR
    # enabled (server-side conversation storage rejected by the API) — the
    # SDK then sends ``store: false`` and the full conversation as ``input``
    # on every turn. Only used by the WebSocket and ``http_responses``
    # backends; the ``http`` backend ignores it.
    zdr_enabled: bool = _UNSET

    # Spoken-text tool-call-leak guard (http_responses backend only).
    # None = disabled. See :class:`SpeechLeakGuardConfig`.
    speech_leak_guard: Optional[SpeechLeakGuardConfig] = _UNSET

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
    "strict_tool_schemas": True,
    "zdr_enabled": False,
    "speech_leak_guard": None,
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
