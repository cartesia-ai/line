"""
LLM Provider using LiteLLM.

Provides a unified interface to 100+ LLM providers via LiteLLM.
See https://docs.litellm.ai/docs/providers for supported providers.

Model naming:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-haiku-4-5-20251001"
- Google: "gemini/gemini-2.5-flash-preview-09-2025"
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterable, List, Optional, Protocol, Tuple, runtime_checkable

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.tools.utils import _merge_tools, _normalize_tools


@dataclass
class ToolCall:
    """A tool/function call from the LLM."""

    id: str
    name: str
    arguments: str = ""
    is_complete: bool = False
    thought_signature: Optional[str] = None  # For Gemini 3+ models


@dataclass
class StreamChunk:
    """An output chunk from an LLM stream."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_final: bool = False


@dataclass
class Message:
    """An input message in the conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


def _extract_instructions_and_messages(
    messages: List["Message"],
    config: LlmConfig,
) -> Tuple[Optional[str], List["Message"]]:
    """Split system messages out and fold them into a single instructions string.

    WebSocket backends use session/request-level instructions rather than
    in-band system messages. Match the HTTP backend by prepending
    ``config.system_prompt`` ahead of any explicit system messages.
    """
    cfg = _normalize_config(config)
    system_parts: List[str] = []
    if cfg.system_prompt:
        system_parts.append(cfg.system_prompt)

    non_system: List[Message] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content or "")
        else:
            non_system.append(msg)

    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, non_system


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol defining the interface all provider backends must implement.

    ``config`` is always required — the ``LlmProvider`` facade normalizes
    it before dispatching so backends never need to handle ``None``.
    """

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        *,
        config: LlmConfig,
        **kwargs: Any,
    ) -> Any: ...

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> None: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------


class LlmProvider:
    """Unified LLM provider facade.

    Selects the appropriate backend (HTTP/LiteLLM, Realtime WS, or Responses WS)
    based on the model name, and delegates all calls to it.

    Handles config normalization and reasoning-effort detection so that
    backends receive normalized configs and pre-computed flags.  This class
    is a public SDK surface — callers may pass raw ``LlmConfig`` and tool
    specs which are normalized internally.

    Args:
        model: Model name (e.g. ``"gpt-4o"``, ``"gpt-4o-realtime-preview"``).
        api_key: Provider API key. Required.
        config: LLM configuration (normalized internally).
        tools: Tool specs (``List[FunctionTool]``).  Normalized internally,
            stored as defaults and merged with any per-call tool overrides by
            tool name.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[Any]] = None,
    ):
        if not api_key:
            raise ValueError("Missing API key in LlmProvider initialization")
        if not _is_supported_model(model):
            raise ValueError(
                f"Model {model} is not supported. See https://models.litellm.ai/ for supported models."
            )

        self._model = model
        self._tools = list(tools or [])
        normalized_config = _normalize_config(config or LlmConfig())
        self._config = normalized_config

        use_realtime = _is_realtime_model(model)
        use_websocket = _is_websocket_model(model)
        supports_reasoning_effort, default_reasoning_effort = _detect_reasoning_effort(model)

        if use_realtime:
            from line.llm_agent.realtime_provider import _RealtimeProvider

            self._backend: ProviderProtocol = _RealtimeProvider(
                model=model,
                api_key=api_key,
            )
        elif use_websocket:
            from line.llm_agent.websocket_provider import _WebSocketProvider

            self._backend = _WebSocketProvider(
                model=model,
                api_key=api_key,
                default_reasoning_effort=default_reasoning_effort,
            )
        else:
            from line.llm_agent.http_provider import _HttpProvider

            self._backend = _HttpProvider(
                model=model,
                api_key=api_key,
                supports_reasoning_effort=supports_reasoning_effort,
                default_reasoning_effort=default_reasoning_effort,
            )

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        config: Optional[LlmConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamChunk]:
        effective_config = _normalize_config(config) if config else self._config
        effective_tools, web_search_options = _normalize_tools(
            _merge_tools(self._tools, tools), model=self._model
        )

        if web_search_options is not None:
            kwargs = {**kwargs, "web_search_options": web_search_options}

        return self._backend.chat(messages, effective_tools, config=effective_config, **kwargs)

    def _set_tools(self, tools: Optional[List[Any]]) -> None:
        """Replace the provider's default tool specs."""
        self._tools = list(tools or [])

    async def warmup(
        self,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        effective_config = _normalize_config(config) if config else self._config
        effective_tools, web_search_options = _normalize_tools(
            _merge_tools(self._tools, tools), model=self._model
        )
        await self._backend.warmup(
            config=effective_config,
            tools=effective_tools,
            web_search_options=web_search_options,
        )

    async def aclose(self) -> None:
        await self._backend.aclose()


def _detect_reasoning_effort(model: str) -> Tuple[bool, Optional[str]]:
    """Detect whether *model* supports ``reasoning_effort`` and find the best default.

    Returns ``(supports_reasoning_effort, default_reasoning_effort)``.

    "none" is ideal (disables reasoning entirely) but not all providers
    support it.  We probe litellm's own parameter mapping to find out: if
    mapping "none" through the provider's config raises, fall back to "low"
    (the lowest universally-supported level).
    """
    from litellm import get_llm_provider, get_supported_openai_params
    from litellm.utils import get_optional_params

    supported = get_supported_openai_params(model=model) or []
    supports = "reasoning_effort" in supported

    default = "low"
    if supports:
        try:
            _, provider, _, _ = get_llm_provider(model=model)
            get_optional_params(model=model, custom_llm_provider=provider, reasoning_effort="none")
            default = "none"
        except Exception:
            # HACK: Anthropic's LiteLLM mapping annoyingly doesn't support `"none"` (the string) as a
            # value for reasoning_effort, so None (omitting the param) is the correct way
            # to skip the thinking block entirely; "low" would still enable a 1024-token
            # thinking budget.
            if "anthropic" in model.lower():
                default = None

    return supports, default


def _supported_openai_params(model: str) -> Optional[List[str]]:
    """Return supported OpenAI-style params for accepted public model names.

    Direct OpenAI WebSocket/realtime models are accepted even when LiteLLM does
    not yet know about them, because they do not route through the HTTP/LiteLLM
    backend.
    """
    if _is_websocket_model(model):
        return ["reasoning_effort"]
    if _is_realtime_model(model):
        return []

    from litellm import get_supported_openai_params

    return get_supported_openai_params(model=model)


def _is_supported_model(model: str) -> bool:
    """Return True if ``model`` is accepted by the public provider surface."""
    return _supported_openai_params(model) is not None


def _is_direct_openai_model(model: str) -> bool:
    """Return True for bare OpenAI model names or explicit ``openai/`` ones."""
    lower = model.lower()
    return "/" not in lower or lower.startswith("openai/")


def _is_realtime_model(model: str) -> bool:
    """Check if a model name indicates a direct OpenAI Realtime model."""
    return _is_direct_openai_model(model) and "realtime" in model.lower().split("/", 1)[-1]


def _is_websocket_model(model: str) -> bool:
    """Check if a model should use the WebSocket (Responses API) backend.

    Use WebSocket mode for direct OpenAI gpt-5 variants.
    """
    lower = model.lower().split("/", 1)[-1]
    return _is_direct_openai_model(model) and (lower.startswith("gpt-5") or lower.startswith("gpt5"))


# Backward-compat alias — emits a deprecation warning on instantiation.
class LLMProvider(LlmProvider):
    """Deprecated: use :class:`LlmProvider` instead."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "LLMProvider is deprecated, use LlmProvider instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
