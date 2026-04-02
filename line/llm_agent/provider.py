"""
LLM Provider using LiteLLM.

Provides a unified interface to 100+ LLM providers via LiteLLM.
See https://docs.litellm.ai/docs/providers for supported providers.

Model naming:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-haiku-4-5-20251001"
- Google: "gemini/gemini-2.5-flash-preview-09-2025"
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, AsyncIterator, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from loguru import logger

from line.llm_agent.config import LlmConfig, _merge_configs, _normalize_config
from line.llm_agent.tools.utils import FunctionTool, _merge_tools, _normalize_tools


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
    system_parts: List[str] = []
    if config.system_prompt:
        system_parts.append(config.system_prompt)

    non_system: List[Message] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content or "")
        else:
            non_system.append(msg)

    instructions = "\n\n".join(system_parts) if system_parts else None
    return instructions, non_system


def _normalize_messages(messages: List["Message"]) -> Optional[List["Message"]]:
    """Normalize and validate messages before sending to the LLM backend.

    Applies the following transformations in order:

    1. **Filter empty messages** – user/assistant messages with no meaningful
       content (empty or whitespace-only) *and* no tool calls are dropped.
    2. **Remove unpaired tool calls** – every assistant-side ``tool_calls``
       entry must have a corresponding ``role="tool"`` response.  Unpaired
       tool calls are logged and removed.  Any orphaned tool responses
       (responses with no matching tool call) are also dropped.
    3. **Reorder tool responses** – tool response messages are moved to
       appear immediately after the assistant message whose tool call
       triggered them, regardless of their original position.
    4. **Validate terminal message** – the conversation must not end with an
       assistant message (providers require user/tool to continue), and the
       final user message must be non-empty.

    Returns the cleaned message list, or ``None`` when the list is empty or
    fatally invalid (caller should skip the LLM call).
    """

    # Index tool calls and responses for pairing
    tool_responses: Dict[str, List[Message]] = defaultdict(list)
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id:
            tool_responses[msg.tool_call_id].append(msg)

    result: List[Message] = []
    for msg in messages:
        if msg.role == "tool":
            continue  # placed after their tool call below

        tool_calls = msg.tool_calls or []
        paired = [tc for tc in tool_calls if tc.id in tool_responses]
        unpaired = [tc for tc in tool_calls if tc.id not in tool_responses]
        has_content = msg.content is not None and msg.content.strip()

        if msg.role in ("user", "assistant") and not has_content and not paired:
            logger.warning(f"Dropping empty message with no tool calls: role={msg.role}, name={msg.name}")
            continue

        result.append(
            Message(
                role=msg.role,
                content=msg.content,
                tool_calls=paired,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            )
        )
        for tc in unpaired:
            logger.warning(f"Dropping unpaired tool call: {tc.name} (id={tc.id})")
        for tc in paired:
            result.extend(tool_responses[tc.id])

    if not result:
        logger.warning("Skipping LLM call: no messages to send")
        return None

    # 4. Validate terminal message
    last = result[-1]
    if last.role == "assistant":
        logger.warning("Skipping LLM call: conversation cannot end with assistant message")
        return None

    if last.role == "user" and (not last.content or not last.content.strip()):
        logger.warning("Skipping LLM call: last user message must be non-empty")
        return None

    return result


async def _empty_stream() -> AsyncIterable[StreamChunk]:
    """Return an empty async iterable of StreamChunks."""
    return
    yield  # pragma: no cover – makes this an async generator


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class ChatStream(Protocol):
    """Return type of ``chat()`` — supports both ``async with`` and ``async for``."""

    def __aiter__(self) -> AsyncIterator[StreamChunk]: ...
    async def __aenter__(self) -> "ChatStream": ...
    async def __aexit__(self, *exc_info: Any) -> None: ...


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol defining the interface all provider backends must implement.

    ``config`` is always required — the ``LlmProvider`` facade normalizes
    it before dispatching so backends never need to handle ``None``.
    """

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        *,
        config: LlmConfig,
        **kwargs: Any,
    ) -> ChatStream: ...

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[FunctionTool]] = None,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
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
        tools: Tool specs (``List[ToolSpec]``).  Normalized internally,
            stored as defaults and merged with any per-call tool overrides by
            tool name.
        backend: Force a specific backend (``"http"``, ``"realtime"``, or
            ``"websocket"``).  When ``None`` (default), the backend is
            auto-detected from the model name.  Raises ``ValueError`` if
            the requested backend is incompatible with the model.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[Any]] = None,
        backend: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("Missing API key in LlmProvider initialization")
        mcfg = _get_model_config(model, backend=backend)

        self._model = model
        self._tools = list(tools or [])
        normalized_config = _normalize_config(config or LlmConfig())
        self._config = normalized_config
        self._http_fallback_backend: Optional[ProviderProtocol] = None

        if mcfg.backend == "realtime":
            from line.llm_agent.realtime_provider import _RealtimeProvider
            logger.info(f"Realtime provider selected for model: {model}")

            self._backend: ProviderProtocol = _RealtimeProvider(
                model=model,
                api_key=api_key,
            )
        elif mcfg.backend == "websocket":
            from line.llm_agent.http_provider import _HttpProvider
            from line.llm_agent.websocket_provider import _WebSocketProvider
            logger.info(f"WebSocket provider selected for model: {model}")

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
            logger.info(f"HTTP provider selected for model: {model}")
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
    ) -> "ChatStream":
        normalized = _normalize_messages(messages)
        if normalized is None:
            return _empty_stream()

        effective_config = _merge_configs(self._config, config) if config else self._config
        effective_tools, web_search_options = _normalize_tools(
            _merge_tools(self._tools, tools), model=self._model
        )
        backend = self._select_backend(effective_config)

        if web_search_options is not None:
            kwargs = {**kwargs, "web_search_options": web_search_options}

        return backend.chat(normalized, effective_tools, config=effective_config, **kwargs)

    def _set_tools(self, tools: Optional[List[Any]]) -> None:
        """Replace the provider's default tool specs."""
        self._tools = list(tools or [])

    async def warmup(
        self,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[Any]] = None,
    ) -> None:
        effective_config = _merge_configs(self._config, config) if config else self._config
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


_VALID_BACKENDS = frozenset({"http", "realtime", "websocket"})


def _get_model_config(model: str, *, backend: Optional[str] = None) -> _ModelConfig:
    """Return the unified configuration for *model*.

    This is the **single source of truth** for backend routing, reasoning-effort
    support, and supported OpenAI parameter lists.

    Args:
        model: Model name.
        backend: Force a specific backend.  Raises ``ValueError`` if
            incompatible with the model or if the model is unsupported.
    """
    if backend is not None and backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend {backend!r}. Must be one of: {', '.join(sorted(_VALID_BACKENDS))}")

    # Realtime models — dedicated realtime backend, no override allowed.
    if _is_realtime_model(model):
        if backend is not None and backend != "realtime":
            raise ValueError(f"Realtime model {model!r} is incompatible with backend {backend!r}")
        return _ModelConfig(
            backend="realtime",
            supports_reasoning_effort=False,
            default_reasoning_effort=None,
        )

    # WebSocket models — auto-select websocket, but allow http override.
    if _is_websocket_model(model):
        effective = backend or "websocket"
        if effective == "realtime":
            raise ValueError(
                f"Backend 'realtime' requires a realtime model (e.g. gpt-4o-realtime-preview), got {model!r}"
            )
        return _ModelConfig(
            backend=effective,
            supports_reasoning_effort=True,
            default_reasoning_effort="low",
        )

    # WebSocket/realtime backends require specific OpenAI models.
    if backend == "websocket":
        raise ValueError(
            f"Backend 'websocket' requires a websocket-compatible model (e.g. gpt-5.2), got {model!r}"
        )
    if backend == "realtime":
        raise ValueError(
            f"Backend 'realtime' requires a realtime model (e.g. gpt-4o-realtime-preview), got {model!r}"
        )

    # Everything else — HTTP via LiteLLM.
    from litellm import get_supported_openai_params

    supported = get_supported_openai_params(model=model)
    if supported is None:
        raise ValueError(
            f"Model {model} is not supported. See https://models.litellm.ai/ for supported models."
        )

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


def _get_model_info(model: str) -> dict:
    """Get model info from LiteLLM's model_cost registry.

    Strips the ``openai/`` prefix for lookup since LiteLLM keys OpenAI models
    without it (e.g., ``gpt-5.4`` not ``openai/gpt-5.4``).
    """
    import litellm

    lower = model.lower()
    if lower.startswith("openai/"):
        lookup = lower[len("openai/") :]
    else:
        lookup = lower
    return litellm.model_cost.get(lookup, {})


def _is_openai_model(model: str) -> bool:
    """Return True for OpenAI models via LiteLLM's model_cost registry."""
    info = _get_model_info(model)
    provider = info.get("litellm_provider")
    if provider:
        # "openai" for direct API, "chatgpt" for ChatGPT API - both use OpenAI endpoints
        return provider in ("openai", "chatgpt")
    # Fallback to pattern matching for models not yet in LiteLLM registry
    lower = model.lower()
    if lower.startswith("openai/"):
        lower = lower[len("openai/") :]
    return lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt/", "codex"))


def _is_realtime_model(model: str) -> bool:
    """Check if a model name indicates a direct OpenAI Realtime model."""
    return _is_openai_model(model) and "realtime" in model.lower().split("/", 1)[-1]


def _is_websocket_model(model: str) -> bool:
    """Check if a model should use the WebSocket (Responses API) backend.

    Returns True for OpenAI models that support the /v1/responses endpoint,
    as detected via LiteLLM's model_cost registry.
    """
    info = _get_model_info(model)
    if info.get("litellm_provider") not in ("openai", "chatgpt"):
        return False
    endpoints = info.get("supported_endpoints", [])
    return "/v1/responses" in endpoints


# Backward-compat alias — emits a deprecation warning on instantiation.
def LLMProvider(*args, **kwargs):
    """Deprecated: use :class:`LlmProvider` instead."""
    import warnings

    warnings.warn(
        "LLMProvider is deprecated, use LlmProvider instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return LlmProvider(*args, **kwargs)
