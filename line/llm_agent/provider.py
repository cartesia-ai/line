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
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from loguru import logger

from line.llm_agent.config import LlmConfig, _merge_configs, _normalize_config
from line.llm_agent.tools.utils import FunctionTool, _merge_tools, _normalize_tools


class ParsedModelId(NamedTuple):
    """Parsed model identifier in ``(provider, model)`` form."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


def parse_model_id(model: str) -> ParsedModelId:
    """Parse a model string into a canonical ``(provider, model)`` pair.

    ``"gpt-4o"``                      → ``("openai", "gpt-4o")``
    ``"openai/gpt-4o"``               → ``("openai", "gpt-4o")``
    ``"chatgpt/gpt-5.4-pro"``         → ``("openai", "gpt-5.4-pro")``
    ``"anthropic/claude-sonnet-4-5"`` → ``("anthropic", "claude-sonnet-4-5")``

    Bare strings (no ``/``) default to the ``openai`` provider. The
    ``chatgpt/`` prefix is a historical alias for the OpenAI endpoint and is
    normalized to ``openai`` so downstream code has a single canonical form.
    """
    if "/" in model:
        provider, name = model.split("/", 1)
        provider = provider.lower()
        if provider == "chatgpt":
            provider = "openai"
        return ParsedModelId(provider, name)
    return ParsedModelId("openai", model)


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

    Selects the appropriate backend (HTTP/LiteLLM by default; Realtime WS for
    realtime models; Responses WS or HTTPS Responses as opt-in alternatives
    for ``gpt-5.2`` / ``gpt-5.4-*`` models) based on the model name and the
    optional ``backend`` override, and delegates all calls to it.

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
        backend: Force a specific backend (``"http"``, ``"http_responses"``,
            ``"realtime"``, or ``"websocket"``).  When ``None`` (default),
            the backend is auto-detected from the model name (HTTP/LiteLLM
            for all non-realtime models).  Set ``backend="http_responses"``
            for Responses-API-capable OpenAI models (e.g. ``gpt-5.2``,
            ``gpt-5.4-mini``) to speak the Responses API natively over
            HTTPS via ``litellm.aresponses``, bypassing LiteLLM's
            chat-completions → responses bridge — useful for avoiding
            duplicated TTS output on gpt-5.4+ where the bridge drops the
            ``phase`` field on ``message`` output items.  Raises
            ``ValueError`` if the requested backend is incompatible with
            the model.
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
        model_id = parse_model_id(model)

        self._model_id = model_id
        self._tools = list(tools or [])
        normalized_config = _normalize_config(config or LlmConfig())
        self._config = normalized_config
        self._backend_override = backend
        mcfg = _get_model_config(model_id, backend=backend, config=normalized_config)

        if mcfg.backend == "realtime":
            from line.llm_agent.realtime_provider import _RealtimeProvider

            logger.info(f"Realtime provider selected for model: {model_id}")

            self._backend: ProviderProtocol = _RealtimeProvider(
                model_id=model_id,
                api_key=api_key,
            )
        elif mcfg.backend == "websocket":
            from line.llm_agent.websocket_provider import _WebSocketProvider

            logger.info(f"WebSocket provider selected for model: {model_id}")

            self._backend = _WebSocketProvider(
                model_id=model_id,
                api_key=api_key,
                default_reasoning_effort=mcfg.default_reasoning_effort,
            )
        elif mcfg.backend == "http_responses":
            from line.llm_agent.http_responses_provider import _HttpResponsesProvider

            logger.info(f"HTTPS Responses provider selected for model: {model_id}")

            self._backend = _HttpResponsesProvider(
                model_id=model_id,
                api_key=api_key,
                default_reasoning_effort=mcfg.default_reasoning_effort,
            )
        else:
            from line.llm_agent.http_provider import _HttpProvider

            logger.info(f"HTTP provider selected for model: {model_id}")
            self._backend = _HttpProvider(
                model_id=model_id,
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
            _merge_tools(self._tools, tools), model_id=self._model_id
        )
        _get_model_config(self._model_id, backend=self._backend_override, config=effective_config)

        if web_search_options is not None:
            kwargs = {**kwargs, "web_search_options": web_search_options}

        return self._backend.chat(normalized, effective_tools, config=effective_config, **kwargs)

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
            _merge_tools(self._tools, tools), model_id=self._model_id
        )
        _get_model_config(self._model_id, backend=self._backend_override, config=effective_config)
        await self._backend.warmup(
            config=effective_config,
            tools=effective_tools,
            web_search_options=web_search_options,
        )

    async def aclose(self) -> None:
        await self._backend.aclose()


@dataclass
class _ModelConfig:
    """Pre-computed model configuration returned by :func:`_get_model_config`.

    Centralizes backend selection, reasoning-effort detection, and supported
    parameter discovery so that every call-site uses the same precedence.
    """

    backend: str  # "realtime" | "websocket" | "http_responses" | "http"
    supports_reasoning_effort: bool
    default_reasoning_effort: Optional[str]


_VALID_BACKENDS = frozenset({"http", "http_responses", "realtime", "websocket"})


def _get_model_config(
    model_id: ParsedModelId,
    *,
    backend: Optional[str] = None,
    config: Optional[LlmConfig] = None,
) -> _ModelConfig:
    """Return the unified configuration for *model_id*.

    This is the **single source of truth** for backend routing, reasoning-effort
    support, supported OpenAI parameter lists, and config applicability.

    Args:
        model_id: Parsed model identifier.
        backend: Force a specific backend.  Raises ``ValueError`` if
            incompatible with the model or if the model is unsupported.
        config: When provided, validate that every field is compatible with the
            selected backend. Raises ``ValueError`` otherwise.
    """
    # Responses-API-capable OpenAI models (gpt-5.2 / gpt-5.4-*) default to
    # the HTTP/LiteLLM backend (chat-completions endpoint).  The Responses
    # API can be reached directly via the opt-in ``websocket`` (WS) or
    # ``http_responses`` (HTTPS via ``litellm.aresponses``) backends.
    from litellm import get_supported_openai_params

    litellm_model = str(model_id)
    litellm_support = get_supported_openai_params(model=litellm_model)

    if backend is not None and backend not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend {backend!r}. Must be one of: {', '.join(sorted(_VALID_BACKENDS))}")

    # Realtime models — dedicated realtime backend, no override allowed.
    elif backend == "realtime" and not _is_realtime_model(model_id):
        raise ValueError(
            "Backend 'realtime' requires a realtime model"
            + f"(e.g. gpt-4o-realtime-preview), got {str(model_id)!r}"
        )
    elif backend is not None and backend != "realtime" and _is_realtime_model(model_id):
        raise ValueError(f"Realtime model {str(model_id)!r} is incompatible with backend {backend!r}")
    elif _is_realtime_model(model_id):
        mcfg = _ModelConfig(
            backend="realtime",
            supports_reasoning_effort=False,
            default_reasoning_effort=None,
        )

    elif backend == "websocket" and not _is_websocket_model(model_id):
        raise ValueError(
            f"Backend 'websocket' requires a websocket-compatible model (e.g. gpt-5.2), got {str(model_id)!r}"
        )
    elif backend == "http_responses" and not _is_websocket_model(model_id):
        raise ValueError(
            f"Backend 'http_responses' requires a Responses-API-capable model "
            f"(e.g. gpt-5.2, gpt-5.4-mini), got {str(model_id)!r}"
        )
    elif _is_websocket_model(model_id) and backend in ("websocket", "http_responses"):
        # Opt-in Responses-API-native paths.  WS speaks the Responses
        # API over ``wss://api.openai.com/v1/responses``; http_responses
        # speaks it over HTTPS via ``litellm.aresponses``.  Either
        # avoids LiteLLM's chat-completions → responses bridge
        # translator, which drops the ``phase`` field on ``message``
        # output items and produces duplicated TTS output for gpt-5.4+.
        responses_supported = get_supported_openai_params(model=litellm_model) or []
        responses_supports_reasoning = "reasoning_effort" in responses_supported
        mcfg = _ModelConfig(
            backend=backend,
            supports_reasoning_effort=responses_supports_reasoning,
            default_reasoning_effort="low" if responses_supports_reasoning else None,
        )

    elif litellm_support:
        supports = "reasoning_effort" in litellm_support
        default: Optional[str] = None
        if supports:
            # Anthropic: omit the param entirely to skip the thinking block. Across
            # litellm versions the "none" string is handled inconsistently — older
            # versions raise, newer versions accept it. None (no param) is the only
            # deterministic skip; "low" would enable a 1024-token thinking budget.
            # Other providers (OpenAI reasoning models, etc.): "none" is the
            # well-defined "no reasoning" value.
            default = None if model_id.provider == "anthropic" else "none"

        mcfg = _ModelConfig(
            backend="http",
            supports_reasoning_effort=supports,
            default_reasoning_effort=default,
        )

    else:
        raise ValueError(
            f"Model {str(model_id)} is not supported. See https://models.litellm.ai/ for supported models."
        )

    _check_config_applicability(mcfg.backend, config)
    return mcfg


# LlmConfig fields the WebSocket (Responses API) backend cannot honor. Setting
# temperature or top_p on the Responses endpoint causes it to close the
# connection silently, and the rest have no equivalent in the Responses request
# schema. Callers that need these must use the HTTP backend.
_WEBSOCKET_INCOMPATIBLE_FIELDS = (
    "temperature",
    "top_p",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "fallbacks",
    "timeout",
    "extra",
)


def _check_config_applicability(backend: str, config: Optional[LlmConfig]) -> None:
    """Raise ``ValueError`` if *config* sets fields *backend* cannot honor."""
    if config is None:
        return
    elif backend != "websocket":
        return
    violations: List[str] = []
    for name in _WEBSOCKET_INCOMPATIBLE_FIELDS:
        value = getattr(config, name)
        if name in ("fallbacks", "extra"):
            if value:
                violations.append(name)
        elif value is not None:
            violations.append(name)
    if violations:
        raise ValueError(
            f"The 'websocket' backend does not support LlmConfig field(s): {', '.join(violations)}. "
            "Use backend='http' (the default) to send these to LiteLLM."
        )


def _is_openai_model(model_id: ParsedModelId) -> bool:
    """Return True for OpenAI models."""
    return model_id.provider == "openai"


def _is_realtime_model(model_id: ParsedModelId) -> bool:
    """Check if a model name indicates a direct OpenAI Realtime model."""
    return _is_openai_model(model_id) and "realtime" in model_id.model.lower()


_WEBSOCKET_MODELS = frozenset(
    {"gpt-5.2", "gpt-5.2-pro", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro"}
)


def _is_websocket_model(model_id: ParsedModelId) -> bool:
    """Check if a model should use the WebSocket (Responses API) backend."""
    return _is_openai_model(model_id) and model_id.model in _WEBSOCKET_MODELS


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
