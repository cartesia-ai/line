"""
HTTP-based LLM Provider using LiteLLM.

Provides a unified interface to 100+ LLM providers via LiteLLM.
See https://docs.litellm.ai/docs/providers for supported providers.

Model naming:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-haiku-4-5-20251001"
- Google: "gemini/gemini-2.5-flash-preview-09-2025"
"""

import inspect
import json
import time
from typing import Any, AsyncIterator, Dict, List, NamedTuple, Optional, Protocol, cast

from litellm import acompletion
from loguru import logger

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, ParsedModelId, StreamChunk, ToolCall
from line.llm_agent.schema_converter import tools_to_litellm
from line.llm_agent.tools.utils import FunctionTool


class _ClosableAsyncIterable(Protocol):
    def __aiter__(self) -> AsyncIterator[Any]: ...

    async def aclose(self) -> None: ...


class _HttpProvider:
    """
    LLM provider using LiteLLM for unified multi-provider access.

    Handles streaming responses and tool calls for all LiteLLM-supported models.

    Config normalization and reasoning-effort resolution are handled by the
    ``LlmProvider`` facade — this class receives fully-resolved configs and
    tools on every call.
    """

    def __init__(
        self,
        model_id: ParsedModelId,
        api_key: Optional[str] = None,
    ):
        self._model_id = model_id
        self._api_key = api_key

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        *,
        config: LlmConfig,
        **kwargs,
    ) -> "_ChatStream":
        """Start a streaming chat completion.

        Returns a ``_ChatStream`` async context manager.  The actual HTTP
        request is issued in ``__aenter__``.

        Args:
            messages: Conversation messages.
            tools: Optional function tools available for this call.
            config: Pre-normalized config (required, provided by LlmProvider facade).
        """
        llm_messages = self._build_messages(messages, config)

        llm_kwargs: Dict[str, Any] = {
            "model": str(self._model_id),
            "messages": llm_messages,
            "stream": True,
            "num_retries": config.num_retries,
        }

        if self._api_key:
            llm_kwargs["api_key"] = self._api_key
        if config.fallbacks:
            llm_kwargs["fallbacks"] = config.fallbacks
        if config.timeout:
            llm_kwargs["timeout"] = config.timeout

        # Add config parameters
        if config.temperature is not None:
            llm_kwargs["temperature"] = config.temperature
        if config.max_tokens is not None:
            llm_kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            llm_kwargs["top_p"] = config.top_p
        if config.stop:
            llm_kwargs["stop"] = config.stop
        if config.seed is not None:
            llm_kwargs["seed"] = config.seed
        if config.presence_penalty is not None:
            llm_kwargs["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty is not None:
            llm_kwargs["frequency_penalty"] = config.frequency_penalty
        if config.reasoning_effort is not None:
            llm_kwargs["reasoning_effort"] = config.reasoning_effort

        if config.extra:
            llm_kwargs.update(config.extra)

        if tools:
            llm_kwargs["tools"] = tools_to_litellm(tools, strict=config.strict_tool_schemas)

        llm_kwargs.update(kwargs)

        return _ChatStream(llm_kwargs, log_llm_calls=config.log_llm_calls)

    def _build_messages(self, messages: List[Message], config: LlmConfig) -> List[Dict[str, Any]]:
        """Convert Message objects to LiteLLM format."""
        result = []

        if config.system_prompt:
            result.append({"role": "system", "content": config.system_prompt})

        for msg in messages:
            llm_msg: Dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                llm_msg["content"] = msg.content

            if msg.tool_calls:
                # ToolCallRequest
                llm_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                        # Include thought_signature for Gemini 3+ models
                        # LiteLLM expects this in provider_specific_fields
                        **(
                            {"provider_specific_fields": {"thought_signature": tc.thought_signature}}
                            if tc.thought_signature
                            else {}
                        ),
                    }
                    for tc in msg.tool_calls
                ]

            if msg.role == "tool":
                # ToolCallResponse
                llm_msg["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    llm_msg["name"] = msg.name

            result.append(llm_msg)
        return result

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[FunctionTool]] = None,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op for stateless HTTP provider."""
        pass

    async def aclose(self) -> None:
        """Close the provider (no-op for LiteLLM)."""
        pass


class _ChatStream:
    """Async-iterable stream for HTTP chat responses.

    Supports two usage patterns::

        # Pattern 1: async with
        async with provider.chat(...) as stream:
            async for chunk in stream:
                ...

        # Pattern 2: bare async for
        async for chunk in provider.chat(...):
            ...

    Both patterns are equivalent — the HTTP request is issued lazily on
    first iteration.  ``async with`` is supported for API consistency with
    the WebSocket backends.
    """

    def __init__(self, llm_kwargs: Dict[str, Any], *, log_llm_calls: bool = False):
        self._kwargs = llm_kwargs
        self._log_llm_calls = log_llm_calls

    async def __aenter__(self) -> "_ChatStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        started_at = time.perf_counter()
        response_started_at: Optional[float] = None
        first_chunk_at: Optional[float] = None
        first_text_at: Optional[float] = None
        first_tool_call_at: Optional[float] = None
        chunk_count = 0
        text_chunk_count = 0
        tool_call_chunk_count = 0

        if self._log_llm_calls:
            _log_llm_call("LiteLLM input call", self._kwargs)

        response: Optional[_ClosableAsyncIterable] = None
        output_text_parts: List[str] = []
        finish_reason = None
        output_error: Optional[BaseException] = None
        tool_calls: Dict[int, ToolCall] = {}
        try:
            response = cast(_ClosableAsyncIterable, await acompletion(**self._kwargs))
            response_started_at = time.perf_counter()
            arg_states: Dict[int, _ArgState] = {}

            async for chunk in response:
                chunk_count += 1
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()

                text = None
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        text_chunk_count += 1
                        if first_text_at is None:
                            first_text_at = time.perf_counter()
                        output_text_parts.append(text)

                    # Handle incremental tool calls
                    tc_delta = getattr(delta, "tool_calls", None)
                    if tc_delta:
                        tool_call_chunk_count += 1
                        if first_tool_call_at is None:
                            first_tool_call_at = time.perf_counter()
                        for tc in tc_delta:
                            idx = tc.index
                            if idx not in tool_calls:
                                tool_calls[idx] = ToolCall(
                                    id=tc.id or "",
                                    name=tc.function.name if tc.function else "",
                                )
                            else:
                                if tc.id:
                                    tool_calls[idx].id = tc.id
                                if tc.function and tc.function.name:
                                    tool_calls[idx].name = tc.function.name

                            if tc.function and tc.function.arguments:
                                arg_states[idx] = _feed_tool_args(arg_states.get(idx), tc.function.arguments)
                                tool_calls[idx].arguments = arg_states[idx].args

                            # Capture thought_signature for Gemini 3+ models
                            # LiteLLM stores it in provider_specific_fields
                            provider_fields = getattr(tc, "provider_specific_fields", None)
                            if provider_fields:
                                thought_sig = provider_fields.get("thought_signature")
                                if thought_sig:
                                    tool_calls[idx].thought_signature = thought_sig

                # Check finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    if finish_reason in ("tool_calls", "stop"):
                        for tc in tool_calls.values():
                            tc.is_complete = True

                yield StreamChunk(
                    text=text,
                    tool_calls=list(tool_calls.values()) if tool_calls else [],
                    is_final=finish_reason is not None,
                )
        except BaseException as exc:
            output_error = exc
            raise
        finally:
            if self._log_llm_calls:
                _log_llm_call(
                    "LiteLLM output call",
                    _build_logged_completion_output(
                        text="".join(output_text_parts),
                        tool_calls=list(tool_calls.values()),
                        finish_reason=finish_reason,
                        debug=_build_logged_completion_debug(
                            started_at=started_at,
                            completed_at=time.perf_counter(),
                            response_started_at=response_started_at,
                            first_chunk_at=first_chunk_at,
                            first_text_at=first_text_at,
                            first_tool_call_at=first_tool_call_at,
                            chunk_count=chunk_count,
                            text_chunk_count=text_chunk_count,
                            tool_call_chunk_count=tool_call_chunk_count,
                            tool_call_count=len(tool_calls),
                        ),
                        error=output_error,
                    ),
                )
            if response is not None:
                aclose = getattr(response, "aclose", None)
                if callable(aclose):
                    result = aclose()
                    if inspect.isawaitable(result):
                        await result


class _ArgState(NamedTuple):
    """Immutable state for incremental JSON argument accumulation."""

    args: str
    depth: int
    in_string: bool
    escape_next: bool


def _feed_tool_args(state: Optional[_ArgState], fragment: str) -> _ArgState:
    """Accumulate a streamed tool-call argument fragment.

    Providers stream tool call arguments differently:
    - OpenAI/Anthropic send incremental fragments that must be concatenated.
    - Gemini sends complete args repeated each chunk that should replace.

    We distinguish these by tracking unquoted brace depth. When depth reaches 0
    the JSON object is complete; any subsequent fragment is a Gemini-style resend
    and replaces rather than concatenates.
    """
    if state is None or (state.depth == 0 and state.args):
        # First fragment, or previous args were complete (Gemini resend)
        args = fragment
        depth, in_str, esc = 0, False, False
    else:
        args = state.args + fragment
        depth, in_str, esc = state.depth, state.in_string, state.escape_next

    for ch in fragment:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

    return _ArgState(args, depth, in_str, esc)


_REDACTED = "<redacted>"
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "proxy_authorization",
        "x-api-key",
        "openai-api-key",
    }
)


def _redact_llm_payload(value: Any) -> Any:
    """Return a JSON-friendly copy of a LiteLLM payload with credentials hidden."""
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            normalized = key_str.lower().replace("_", "-")
            if key_str.lower().endswith("api_key") or normalized in _SENSITIVE_KEYS:
                redacted[key] = _REDACTED
            else:
                redacted[key] = _redact_llm_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_llm_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_llm_payload(item) for item in value]
    return value


def _log_llm_call(label: str, payload: Any) -> None:
    """Log a sanitized LLM payload as stable, readable JSON."""
    logger.info(
        "{}:\n{}",
        label,
        json.dumps(_redact_llm_payload(payload), indent=2, sort_keys=True, default=str),
    )


def _build_logged_completion_output(
    *,
    text: str,
    tool_calls: List[ToolCall],
    finish_reason: Optional[str],
    debug: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    """Build a chat-completion-shaped summary from a streamed LiteLLM response."""
    result: Dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                            "is_complete": tc.is_complete,
                            **(
                                {"provider_specific_fields": {"thought_signature": tc.thought_signature}}
                                if tc.thought_signature
                                else {}
                            ),
                        }
                        for tc in tool_calls
                    ],
                },
                "finish_reason": finish_reason,
            }
        ]
    }
    if debug is not None:
        result["line_debug"] = debug
    if error is not None:
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return result


def _ms_since(start: float, end: Optional[float]) -> Optional[float]:
    if end is None:
        return None
    return round((end - start) * 1000, 3)


def _build_logged_completion_debug(
    *,
    started_at: float,
    completed_at: float,
    response_started_at: Optional[float],
    first_chunk_at: Optional[float],
    first_text_at: Optional[float],
    first_tool_call_at: Optional[float],
    chunk_count: int,
    text_chunk_count: int,
    tool_call_chunk_count: int,
    tool_call_count: int,
) -> Dict[str, Any]:
    """Build SDK-specific timing fields for a streamed LiteLLM call."""
    return {
        "invoked_tool": tool_call_count > 0,
        "tool_call_count": tool_call_count,
        "chunk_count": chunk_count,
        "text_chunk_count": text_chunk_count,
        "tool_call_chunk_count": tool_call_chunk_count,
        "duration_ms": _ms_since(started_at, completed_at),
        "time_to_stream_start_ms": _ms_since(started_at, response_started_at),
        "time_to_first_chunk_ms": _ms_since(started_at, first_chunk_at),
        "time_to_first_text_ms": _ms_since(started_at, first_text_at),
        "time_to_first_tool_call_ms": _ms_since(started_at, first_tool_call_at),
    }
