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
    """A input message in the conversation."""

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

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any: ...

    async def warmup(self, config: Optional[Any] = None) -> None: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# Unified facade
# ---------------------------------------------------------------------------


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
        backend: Explicit backend selection: ``"http"``, ``"realtime"``, or
            ``"websocket"``.  When ``None`` (the default) the backend is
            auto-detected from the model name.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        backend: Optional[str] = None,
    ):
        normalized_config = _normalize_config(config or LlmConfig())

        use_realtime = backend == "realtime" or (backend is None and _is_realtime_model(model))
        use_websocket = backend == "websocket" or (backend is None and _is_websocket_model(model))

        if use_realtime:
            from line.llm_agent.realtime_provider import RealtimeProvider

            self._backend: ProviderProtocol = RealtimeProvider(
                model=model, api_key=api_key, config=normalized_config
            )
        elif use_websocket:
            from line.llm_agent.websocket_provider import WebSocketProvider

            supports, default = self._detect_reasoning_effort(model)
            self._backend = WebSocketProvider(
                model=model,
                api_key=api_key,
                config=normalized_config,
                default_reasoning_effort=default,
            )
        else:
            from line.llm_agent.http_provider import HttpProvider

            supports, default = self._detect_reasoning_effort(model)
            self._backend = HttpProvider(
                model=model,
                api_key=api_key,
                config=normalized_config,
                supports_reasoning_effort=supports,
                default_reasoning_effort=default,
            )

    @staticmethod
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
                pass

        return supports, default

    async def chat(self, messages, tools=None, config=None, **kwargs):
        cfg = _normalize_config(config) if config else None
        return await self._backend.chat(messages, tools, config=cfg, **kwargs)

    async def warmup(self, config=None):
        cfg = _normalize_config(config) if config else None
        await self._backend.warmup(cfg)

    async def aclose(self):
        await self._backend.aclose()


# Backward-compat alias — existing scripts and examples import LLMProvider directly.
from line.llm_agent.http_provider import HttpProvider as LLMProvider  # noqa: E402, F401
