"""
Shared utilities for the Responses-API-based providers.

Contains:

- ``_ws_connect`` / ``_ws_is_closed`` — WebSocket connection helpers
- ``_WsEventStream`` — unified event→chunk translator for Realtime and Responses APIs
- ``_AsyncIterableContext`` — async context manager + async iterable wrapper
- ``_build_responses_body`` / ``_plan_responses_chat`` — transport-agnostic
  Responses-API request building and history planning
- Divergence/identity utilities used by both WebSocket and HTTPS Responses providers
"""

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from loguru import logger
import websockets
from websockets.legacy.client import WebSocketClientProtocol
from websockets.protocol import State as WsState

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import (
    Message,
    ParsedModelId,
    StreamChunk,
    ToolCall,
    _extract_instructions_and_messages,
)
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.tools.utils import FunctionTool

ExpandedItem = Tuple[Dict[str, Any], tuple]
ConversationEntry = Tuple[tuple, Optional[str]]  # (identity, opaque_id | None)

# Type: (old_history, response_or_ack) -> new_history
HistoryUpdate = Callable[[List[ConversationEntry], Dict[str, Any]], List[ConversationEntry]]


def _context_identity(
    instructions: Optional[str],
    tool_defs: Optional[List[Dict[str, Any]]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> tuple:
    """Identity fingerprint for shared WebSocket context/session configuration."""
    tools_key = tuple(json.dumps(t, sort_keys=True) for t in (tool_defs or []))
    return ("__context__", instructions or "", tools_key, temperature, max_tokens)


def _message_identity(msg: Message) -> tuple:
    """Compute an identity fingerprint for a single Message.

    Used by both WebSocket providers for divergence detection / diff-sync.

    For assistant messages with tool calls, all calls are included so that
    parallel tool-call turns are fully distinguished.
    """
    if msg.tool_calls:
        tc_keys = tuple((tc.name, tc.arguments, tc.id) for tc in msg.tool_calls)
        return ("assistant_tool_call", tc_keys)
    return (msg.role, msg.content or "", msg.tool_call_id or "", msg.name or "")


def _expand_message(
    msg: Message,
    *,
    assistant_text_type: str,
) -> List[ExpandedItem]:
    """Expand one Message into provider item payloads plus identities.

    The WebSocket Responses and Realtime providers share the same logical
    message expansion; only the assistant text part type differs
    (``output_text`` vs ``text``).
    """
    if msg.role == "user":
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": msg.content or ""}],
        }
        return [(item, _message_identity(msg))]

    if msg.role == "assistant":
        pairs: List[ExpandedItem] = []
        if msg.content:
            item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": assistant_text_type, "text": msg.content}],
            }
            pairs.append((item, ("assistant", msg.content, "", "")))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                item = {
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                identity = ("assistant_tool_call", ((tc.name, tc.arguments, tc.id),))
                pairs.append((item, identity))
        if pairs:
            return pairs

        item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": assistant_text_type, "text": msg.content or ""}],
        }
        return [(item, _message_identity(msg))]

    if msg.role == "tool":
        item = {
            "type": "function_call_output",
            "call_id": msg.tool_call_id or "",
            "output": msg.content or "",
        }
        return [(item, _message_identity(msg))]

    raise ValueError(f"Unsupported message role for WebSocket item conversion: {msg.role}")


def _expand_messages(
    messages: List[Message],
    *,
    assistant_text_type: str,
) -> List[ExpandedItem]:
    """Expand a message list into provider item payloads plus identities."""
    pairs: List[ExpandedItem] = []
    for msg in messages:
        pairs.extend(_expand_message(msg, assistant_text_type=assistant_text_type))
    return pairs


def _build_responses_body(
    *,
    model_id: ParsedModelId,
    default_reasoning_effort: Optional[str],
    instructions: Optional[str],
    tool_defs: Optional[List[Dict[str, Any]]],
    cfg: LlmConfig,
    input: Optional[List[Dict[str, Any]]] = None,
    previous_response_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the transport-agnostic Responses-API request body.

    Used by both the WebSocket provider (which wraps this in a
    ``{"type": "response.create", ...}`` envelope) and the HTTPS provider
    (which spreads it as ``litellm.aresponses(**body)`` kwargs).
    """
    body: Dict[str, Any] = {
        "model": model_id.model,
        "store": True,
    }
    if input is not None:
        body["input"] = input
    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id
    if instructions is not None:
        body["instructions"] = instructions
    if tool_defs is not None:
        body["tools"] = tool_defs
    reasoning_effort = cfg.reasoning_effort or default_reasoning_effort
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.max_tokens is not None:
        body["max_output_tokens"] = cfg.max_tokens
    if cfg.top_p is not None:
        body["top_p"] = cfg.top_p
    return body


def _plan_responses_chat(
    *,
    history: List[ConversationEntry],
    model_id: ParsedModelId,
    default_reasoning_effort: Optional[str],
    messages: List[Message],
    tools: Optional[List[FunctionTool]],
    config: LlmConfig,
    web_search_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], HistoryUpdate]:
    """Compute the Responses-API request body and the history-update callback.

    Returns ``(body, update_fn)`` where ``body`` is a transport-agnostic
    request body (no ``"type"`` envelope — that's WS-specific) and
    ``update_fn(history, response) -> new_history`` rewrites the
    provider's history once the server emits a completed response.

    Shared by ``_WebSocketProvider`` (which wraps ``body`` with
    ``"type": "response.create"``) and ``_HttpResponsesProvider`` (which
    spreads ``body`` as ``litellm.aresponses(**body)`` kwargs).
    """
    instructions, non_system = _extract_instructions_and_messages(messages, config)
    tool_defs = build_openai_tool_defs(
        tools,
        web_search_options=web_search_options,
        strict=config.strict_tool_schemas,
        responses_api=True,
    )

    context_id = _context_identity(
        instructions,
        tool_defs,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    desired_pairs = _expand_messages(non_system, assistant_text_type="output_text")
    desired_ids = [context_id] + [identity for _, identity in desired_pairs]

    if history and history[0][0] == context_id:
        current_identities = [identity for identity, _ in history[1:]]
        shared_item_count, _ = _compute_divergence(current_identities, desired_pairs)
        prefix_len = 1 + shared_item_count
    else:
        prefix_len = 0

    # Scan backwards for the latest response_id checkpoint.
    continuation_idx = 0
    continuation_resp_id: Optional[str] = None
    for i in range(min(prefix_len, len(history)) - 1, -1, -1):
        if history[i][1] is not None:
            continuation_idx = i + 1
            continuation_resp_id = history[i][1]
            break

    if prefix_len < len(history):
        logger.debug(
            "Responses conversation diverged at index %d/%d, rolling back to checkpoint at %d",
            prefix_len,
            len(history),
            continuation_idx,
        )

    input_pairs = desired_pairs[max(0, continuation_idx - 1) :]
    body = _build_responses_body(
        model_id=model_id,
        default_reasoning_effort=default_reasoning_effort,
        instructions=instructions,
        tool_defs=tool_defs,
        cfg=config,
        input=[item for item, _ in input_pairs],
        previous_response_id=continuation_resp_id,
    )

    def update(old_history: List[ConversationEntry], response: Dict[str, Any]) -> List[ConversationEntry]:
        response_id = response.get("id")
        model_ids = _extract_model_output_identities(response)
        new_history = list(old_history[:continuation_idx])
        for ident in desired_ids[continuation_idx:]:
            new_history.append((ident, None))
        for ident in model_ids:
            new_history.append((ident, response_id))
        return new_history

    return body, update


def _extract_model_output_identities(response: Dict[str, Any]) -> List[tuple]:
    """Derive ordered message identities from a Responses API output.

    Shared by the WebSocket and HTTPS Responses providers — both use the
    same identity-based history bookkeeping.
    """
    output_items = response.get("output", [])
    identities: List[tuple] = []
    for item in output_items:
        if item.get("type") == "function_call":
            identities.append(
                (
                    "assistant_tool_call",
                    ((item.get("name", ""), item.get("arguments", ""), item.get("call_id", "")),),
                )
            )
        elif item.get("type") == "message":
            text_parts: List[str] = []
            for part in item.get("content", []):
                if part.get("type") in {"output_text", "text"}:
                    text_parts.append(part.get("text", ""))
            if text_parts:
                identities.append(("assistant", "".join(text_parts), "", ""))

    return identities


def _compute_divergence(
    current_identities: List[tuple],
    desired_pairs: List[ExpandedItem],
) -> Tuple[int, List[ExpandedItem]]:
    """Compute divergence between current identities and desired expanded items.

    Returns ``(prefix_len, after_divergence)`` where ``prefix_len`` is the
    length of the shared item-identity prefix between the current state and the
    desired expanded item stream.
    """
    desired_identities = [identity for _, identity in desired_pairs]
    prefix_len = 0
    for i in range(min(len(current_identities), len(desired_identities))):
        if current_identities[i] == desired_identities[i]:
            prefix_len = i + 1
        else:
            break
    return prefix_len, desired_pairs[prefix_len:]


# Timeout for draining events after cancelling a response (seconds).
WS_DRAIN_TIMEOUT = 5

# Terminal event types across both Realtime and Responses APIs.
_TERMINAL_EVENTS = frozenset(
    {
        "response.done",  # Realtime API
        "response.completed",  # Responses API
        "response.failed",  # Responses API
        "response.incomplete",  # Responses API
    }
)

# Events we can safely ignore across both APIs.
_IGNORED_EVENTS = frozenset(
    {
        "response.created",
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.done",
        "response.output_text.annotation.added",
        "response.text.done",
        "rate_limits.updated",
    }
)


# ---------------------------------------------------------------------------
# WebSocket connection helpers
# ---------------------------------------------------------------------------


def _ws_is_closed(ws: Optional[WebSocketClientProtocol]) -> bool:
    """Check if a WebSocket is closed or uninitialized."""
    if ws is None:
        return True
    return ws.state in (WsState.CLOSED, WsState.CLOSING)


async def _ws_connect(
    url: str,
    api_key: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> WebSocketClientProtocol:
    """Open a new WebSocket connection and return it."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        **(extra_headers or {}),
    }
    return await websockets.connect(
        url,
        extra_headers=headers,
        max_size=2**24,  # 16 MB
        ping_interval=20,
        ping_timeout=20,
    )


# ---------------------------------------------------------------------------
# _WsEventStream — event → chunk translator
# ---------------------------------------------------------------------------


class _WsEventStream:
    """Reads OpenAI streaming events and yields ``StreamChunk`` objects.

    Handles both the Realtime API and Responses API WebSocket mode — the
    streaming event names are identical; only the terminal events differ,
    and we handle the superset.

    Has no lifecycle responsibilities — no lock, no cancel/drain, no WS
    management.  The ``on_response_done`` callback is invoked when a
    terminal event arrives so the provider can update its state.
    """

    def __init__(self, ws: WebSocketClientProtocol, on_response_done: Callable[[Dict[str, Any]], None]):
        self._ws = ws
        self._on_response_done = on_response_done
        self.done = False

    async def drain(self) -> None:
        """Consume WS events until the response terminates."""
        while True:
            event = json.loads(await self._ws.recv())
            evt_type = event.get("type", "")
            if evt_type in _TERMINAL_EVENTS:
                self._on_response_done(event.get("response") or {})
                return
            if evt_type == "error":
                return

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        ws = self._ws
        tool_calls: Dict[str, ToolCall] = {}
        received_content = False  # Track whether any content events arrived

        try:
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                event_type = event.get("type", "")

                # --- Text deltas ---
                if event_type in ("response.output_text.delta", "response.text.delta"):
                    delta = event.get("delta", "")
                    if delta:
                        received_content = True
                        yield StreamChunk(text=delta)

                # --- Function call argument deltas ---
                elif event_type == "response.function_call_arguments.delta":
                    call_id = event.get("call_id", "")
                    delta = event.get("delta", "")
                    if call_id not in tool_calls:
                        tool_calls[call_id] = ToolCall(
                            id=call_id,
                            name=event.get("name", ""),
                            arguments=delta,
                        )
                    else:
                        tool_calls[call_id].arguments += delta

                # --- Function call arguments done ---
                elif event_type == "response.function_call_arguments.done":
                    call_id = event.get("call_id", "")
                    if call_id in tool_calls:
                        tool_calls[call_id].arguments = event.get("arguments", tool_calls[call_id].arguments)
                        tool_calls[call_id].name = event.get("name", tool_calls[call_id].name)

                # --- Output item added (get function name early) ---
                elif event_type == "response.output_item.added":
                    received_content = True
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        name = item.get("name", "")
                        if call_id and call_id not in tool_calls:
                            tool_calls[call_id] = ToolCall(id=call_id, name=name, arguments="")

                # --- Output item done (function call complete) ---
                elif event_type == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        if call_id in tool_calls:
                            tool_calls[call_id].name = item.get("name", tool_calls[call_id].name)
                            tool_calls[call_id].arguments = item.get(
                                "arguments", tool_calls[call_id].arguments
                            )
                            tool_calls[call_id].is_complete = True
                        else:
                            tool_calls[call_id] = ToolCall(
                                id=call_id,
                                name=item.get("name", ""),
                                arguments=item.get("arguments", ""),
                                is_complete=True,
                            )

                # --- Terminal events (response complete) ---
                elif event_type in _TERMINAL_EVENTS:
                    self.done = True
                    response = event.get("response") or {}
                    status = response.get("status") or ""
                    self._on_response_done(response)

                    # Check for failure — covers both APIs:
                    # Realtime: response.done with status="failed"
                    # Responses: response.failed event
                    error = response.get("error") or {}
                    if event_type == "response.failed" or status == "failed":
                        raise RuntimeError(
                            f"OpenAI API error ({error.get('code', '')}): "
                            f"{error.get('message', 'unknown error')}"
                        )

                    # Only force-mark tool calls on completed responses.
                    # Incomplete/cancelled responses may have truncated
                    # arguments that would fail json.loads downstream.
                    if status == "completed":
                        for tc in tool_calls.values():
                            tc.is_complete = True
                    else:
                        details = response.get("incomplete_details") or {}
                        logger.warning(
                            "Non-completed response from API: status=%s, reason=%s, response_id=%s",
                            status,
                            details.get("reason", ""),
                            response.get("id", ""),
                        )

                    yield StreamChunk(
                        tool_calls=list(tool_calls.values()) if tool_calls else [],
                        is_final=True,
                    )
                    return

                # --- Error (protocol-level) ---
                elif event_type == "error":
                    self.done = True
                    error = event.get("error") or {}
                    raise RuntimeError(
                        f"OpenAI API error ({error.get('code', '')}): {error.get('message', 'unknown error')}"
                    )

                # --- Safely ignorable events ---
                elif event_type in _IGNORED_EVENTS:
                    pass

                # --- Unrecognized events ---
                else:
                    logger.warning(
                        f"Unrecognized WebSocket event type: {event_type} (keys: {list(event.keys())})"
                    )

        except websockets.exceptions.ConnectionClosed as e:
            self.done = True
            close_code = e.rcvd.code if e.rcvd else None
            close_reason = e.rcvd.reason if e.rcvd else None
            if not received_content:
                logger.error(
                    f"WebSocket closed before delivering any content: code={close_code} reason={close_reason}"
                )
                raise RuntimeError(
                    f"WebSocket closed (code={close_code}) before delivering any response content"
                ) from e
            logger.warning(f"WebSocket closed during streaming: code={close_code} reason={close_reason}")
            if e.rcvd and e.rcvd.code == 1000:
                # Server closed cleanly after partial content — yield what we have.
                yield StreamChunk(
                    tool_calls=list(tool_calls.values()) if tool_calls else [],
                    is_final=True,
                )
                return
            raise RuntimeError(f"WebSocket connection closed during streaming: {e}") from e


# ---------------------------------------------------------------------------
# Cancel / drain
# ---------------------------------------------------------------------------


async def _cancel_and_drain(stream: _WsEventStream, ws: WebSocketClientProtocol) -> None:
    """Cancel the in-progress response and drain remaining WS events.

    Catches ``BaseException`` (not just ``Exception``) so that
    ``asyncio.CancelledError`` during the send/drain cycle still results
    in the WebSocket being closed.  Without this, a task cancellation
    could leave unconsumed events on the wire, corrupting the next
    ``chat()`` call on the same connection.
    """
    if ws.state in (WsState.CLOSED, WsState.CLOSING):
        return

    try:
        await ws.send(json.dumps({"type": "response.cancel"}))
        await asyncio.wait_for(stream.drain(), timeout=WS_DRAIN_TIMEOUT)
    except BaseException as e:
        # Drain failed or was interrupted — close the WS so the next
        # chat() reconnects with a clean wire instead of reading stale
        # events from this response.
        logger.debug("Stream drain failed (%s), closing WS for clean reconnect", type(e).__name__)
        try:
            await ws.close()
        except BaseException:
            pass
        # Re-raise CancelledError / KeyboardInterrupt so the caller's
        # cancellation semantics are preserved.
        if not isinstance(e, Exception):
            raise


# ---------------------------------------------------------------------------
# _AsyncIterableContext — async context manager + async iterable
# ---------------------------------------------------------------------------


class _AsyncIterableContext:
    """Wrap an async-iterable factory so it also supports ``async with``.

    On ``__aexit__``, the underlying async iterator is closed (via
    ``aclose``), which unwinds any ``async with`` / cleanup blocks
    inside the generator.

    Usage::

        def make_iter():
            async def _gen():
                ...
                yield item
            return _gen()

        ctx = _AsyncIterableContext(make_iter)

        # Pattern 1 — context-managed
        async with ctx as stream:
            async for item in stream:
                ...

        # Pattern 2 — bare iteration
        async for item in ctx:
            ...
    """

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._it: Any = None

    async def __aenter__(self) -> "_AsyncIterableContext":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._it is not None:
            aclose = getattr(self._it, "aclose", None)
            if aclose is not None:
                await aclose()
            self._it = None

    def __aiter__(self) -> AsyncIterator:
        self._it = self._factory().__aiter__()
        return self._it
