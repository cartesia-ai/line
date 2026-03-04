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
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.tools.utils import FunctionTool, _normalize_tools


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


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol defining the interface all provider backends must implement."""

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any: ...

    async def warmup(self, config: Optional[Any] = None, tools: Optional[List[Any]] = None) -> None: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------


class LlmProvider:
    """Unified LLM provider facade.

    Selects the appropriate backend (HTTP/LiteLLM, Realtime WS, or Responses WS)
    based on the model name, and delegates all calls to it.

    Centralizes config normalization and reasoning-effort detection so that
    backends receive pre-normalized configs and pre-computed flags.

    Args:
        model: Model name (e.g. ``"gpt-4o"``, ``"gpt-4o-realtime-preview"``).
        api_key: Provider API key.
        config: LLM configuration.
        tools: Pre-normalized tools (``List[FunctionTool]``).  Stored as
            defaults and used when ``chat()`` is called without per-call tools.
        backend: Explicit backend selection: ``"http"``, ``"realtime"``, or
            ``"websocket"``.  When ``None`` (the default) the backend is
            auto-detected from the model name.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[Any]] = None,
        backend: Optional[str] = None,
    ):
        self._model = model
        normalized_config = _normalize_config(config or LlmConfig())
        self._config = normalized_config
        self._tools = _resolve_tools(tools, model=model)

        use_realtime = backend == "realtime" or (backend is None and _is_realtime_model(model))
        use_websocket = backend == "websocket" or (backend is None and _is_websocket_model(model))
        supports_reasoning_effort, default_reasoning_effort = _detect_reasoning_effort(model)

        if use_realtime:
            from line.llm_agent.realtime_provider import _RealtimeProvider

            self._backend: ProviderProtocol = _RealtimeProvider(
                model=model, api_key=api_key,
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


    def chat(self, messages, tools=None, config=None, **kwargs):
        cfg = _normalize_config(config) if config else self._config
        effective_tools = _resolve_tools(tools, model=self._model) if tools else self._tools
        return self._backend.chat(messages, effective_tools, config=cfg, **kwargs)

    async def warmup(self, config=None):
        cfg = _normalize_config(config) if config else self._config
        await self._backend.warmup(config=cfg, tools=self._tools)

    async def aclose(self):
        await self._backend.aclose()


def _detect_reasoning_effort(model: str) -> Tuple[bool, str]:
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


def _is_realtime_model(model: str) -> bool:
    """Check if a model name indicates an OpenAI Realtime model."""
    return "realtime" in model.lower()


def _is_websocket_model(model: str) -> bool:
    """Check if a model should use the WebSocket (Responses API) backend.

    Currently matches gpt-5.2 variants.  Other models that support WebSocket
    mode can be opted in via the ``backend`` parameter on ``LlmProvider``.
    """
    lower = model.lower()
    return lower.startswith("gpt-5.2") or lower.startswith("gpt5.2")


def _resolve_tools(tools: Optional[List[Any]], model: str) -> List[FunctionTool]:
    """Resolve tools to FunctionTools, avoiding no-op re-normalization."""
    if not tools:
        return []
    if all(isinstance(tool, FunctionTool) for tool in tools):
        return list(tools)
    return _normalize_tools(tools, model=model)[0]


def _message_identity(msg: Message) -> tuple:
    """Compute an identity fingerprint for a single Message.

    Used by both WebSocket providers for divergence detection / diff-sync.

    For assistant messages with tool calls, identity includes all tool calls
    so divergence checks detect changes to any call in the turn.
    """
    if msg.tool_calls:
        if len(msg.tool_calls) == 1:
            tc = msg.tool_calls[0]
            return ("assistant_tool_call", tc.name, tc.arguments, tc.id)
        tool_calls_key = tuple((tc.name, tc.arguments, tc.id) for tc in msg.tool_calls)
        return ("assistant_tool_calls", tool_calls_key)
    return (msg.role, msg.content or "", msg.tool_call_id or "", msg.name or "")


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
