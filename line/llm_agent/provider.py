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
        mcfg = _get_model_config(model)
        if mcfg is None:
            raise ValueError(
                f"Model {model} is not supported. See https://models.litellm.ai/ for supported models."
            )

        self._model = model
        self._tools = list(tools or [])
        normalized_config = _normalize_config(config or LlmConfig())
        self._config = normalized_config
        self._http_fallback_backend: Optional[ProviderProtocol] = None


        if mcfg.backend == "realtime":
            from line.llm_agent.realtime_provider import _RealtimeProvider

            self._backend: ProviderProtocol = _RealtimeProvider(
                model=model,
                api_key=api_key,
            )
        elif mcfg.backend == "websocket":
            from line.llm_agent.http_provider import _HttpProvider
            from line.llm_agent.websocket_provider import _WebSocketProvider

            self._backend = _WebSocketProvider(
                model=model,
                api_key=api_key,
                default_reasoning_effort=mcfg.default_reasoning_effort,
            )
            self._http_fallback_backend = _HttpProvider(
                model=model,
                api_key=api_key,
                supports_reasoning_effort=mcfg.supports_reasoning_effort,
                default_reasoning_effort=mcfg.default_reasoning_effort,
            )
        else:
            from line.llm_agent.http_provider import _HttpProvider

            self._backend = _HttpProvider(
                model=model,
                api_key=api_key,
                supports_reasoning_effort=mcfg.supports_reasoning_effort,
                default_reasoning_effort=mcfg.default_reasoning_effort,
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
        backend = self._select_backend(effective_config)

        if web_search_options is not None:
            kwargs = {**kwargs, "web_search_options": web_search_options}

        return backend.chat(messages, effective_tools, config=effective_config, **kwargs)

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
        backend = self._select_backend(effective_config)
        await backend.warmup(
            config=effective_config,
            tools=effective_tools,
            web_search_options=web_search_options,
        )

    async def aclose(self) -> None:
        await self._backend.aclose()
        if self._http_fallback_backend is not None:
            await self._http_fallback_backend.aclose()

    def _select_backend(self, config: LlmConfig) -> ProviderProtocol:
        """Use HTTP for websocket-incompatible config fields.

        WebSocket mode is lower latency, but it cannot preserve every field on
        ``LlmConfig``. Fall back to the HTTP backend when callers rely on
        request options that only the LiteLLM/HTTP path currently supports.
        """
        if self._http_fallback_backend is None:
            return self._backend

        if (
            config.stop is not None
            or config.seed is not None
            or config.presence_penalty is not None
            or config.frequency_penalty is not None
            or config.fallbacks
            or config.timeout is not None
            or config.extra
        ):
            return self._http_fallback_backend

        return self._backend


@dataclass
class _ModelConfig:
    """Pre-computed model configuration returned by :func:`_get_model_config`.

    Centralizes backend selection, reasoning-effort detection, and supported
    parameter discovery so that every call-site uses the same precedence.
    """

    backend: str  # "realtime" | "websocket" | "http"
    supports_reasoning_effort: bool
    default_reasoning_effort: Optional[str]


def _get_model_config(model: str) -> Optional[_ModelConfig]:
    """Return the unified configuration for *model*, or ``None`` if unsupported.

    This is the **single source of truth** for backend routing, reasoning-effort
    support, and supported OpenAI parameter lists.  Precedence:

    1. Realtime models  → ``_RealtimeProvider``, no reasoning effort.
    2. WebSocket models → ``_WebSocketProvider``, reasoning effort supported.
    3. Everything else  → ``_HttpProvider`` via LiteLLM.
    """
    if _is_realtime_model(model):
        return _ModelConfig(
            backend="realtime",
            supports_reasoning_effort=False,
            default_reasoning_effort=None,
        )

    if _is_websocket_model(model):
        return _ModelConfig(
            backend="websocket",
            supports_reasoning_effort=True,
            default_reasoning_effort="low",
        )

    # HTTP / LiteLLM path — probe litellm for capabilities.
    from litellm import get_supported_openai_params

    supported = get_supported_openai_params(model=model)
    if supported is None:
        return None

    supports = "reasoning_effort" in supported
    default: Optional[str] = "low"
    if supports:
        from litellm import get_llm_provider
        from litellm.utils import get_optional_params

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

    return _ModelConfig(
        backend="http",
        supports_reasoning_effort=supports,
        default_reasoning_effort=default,
    )


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
