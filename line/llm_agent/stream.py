"""
Shared stream utilities for WebSocket-based providers.

Contains:

- ``_ws_connect`` / ``_ws_is_closed`` — WebSocket connection helpers
- ``_WsEventStream`` — unified event→chunk translator for Realtime and Responses APIs
- ``_WsChatStream`` — async context manager + async iterable for safe stream lifecycle
- ``_setup_ws_chat`` / ``_ws_warmup`` — shared provider patterns (free functions)
- Divergence/identity utilities used by both providers
"""

import asyncio
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger
import websockets
from websockets.protocol import State as WsState

from line.llm_agent.provider import Message, StreamChunk, ToolCall

ExpandedItem = Tuple[Dict[str, Any], tuple]
ConversationEntry = Tuple[tuple, Optional[str]]  # (identity, opaque_id | None)


def _normalize_openai_model_name(model: str) -> str:
    """Strip LiteLLM-style ``openai/`` prefixes for direct OpenAI API calls."""
    if model.lower().startswith("openai/"):
        return model.split("/", 1)[1]
    return model


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
        "rate_limits.updated",
    }
)


# ---------------------------------------------------------------------------
# WebSocket connection helpers
# ---------------------------------------------------------------------------


def _ws_is_closed(ws: Optional[Any]) -> bool:
    """Check if a WebSocket is closed or uninitialized."""
    if ws is None:
        return True
    return ws.state in (WsState.CLOSED, WsState.CLOSING)


async def _ws_connect(
    url: str,
    api_key: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Open a new WebSocket connection and return it."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        **(extra_headers or {}),
    }
    return await websockets.connect(
        url,
        additional_headers=headers,
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

    def __init__(self, ws: Any, on_response_done: Callable[[Dict[str, Any]], None]):
        self._ws = ws
        self._on_response_done = on_response_done
        self.done = False

    async def drain(self) -> None:
        """Consume WS events until the response terminates."""
        while True:
            event = json.loads(await self._ws.recv())
            if event.get("type", "") in _TERMINAL_EVENTS:
                self._on_response_done(event.get("response", {}))
                return

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        ws = self._ws
        tool_calls: Dict[str, ToolCall] = {}

        try:
            while True:
                raw = await ws.recv()
                event = json.loads(raw)
                event_type = event.get("type", "")

                # --- Text deltas ---
                if event_type == "response.output_text.delta":
                    delta = event.get("delta", "")
                    if delta:
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
                    response = event.get("response", {})
                    self._on_response_done(response)

                    # Check for failure — covers both APIs:
                    # Realtime: response.done with status="failed"
                    # Responses: response.failed event
                    error = response.get("error", {})
                    status = response.get("status", "")
                    if event_type == "response.failed" or status == "failed":
                        raise RuntimeError(
                            f"OpenAI API error ({error.get('code', '')}): "
                            f"{error.get('message', 'unknown error')}"
                        )

                    for tc in tool_calls.values():
                        tc.is_complete = True

                    yield StreamChunk(
                        tool_calls=list(tool_calls.values()) if tool_calls else [],
                        is_final=True,
                    )
                    return

                # --- Error (protocol-level) ---
                elif event_type == "error":
                    self.done = True
                    error_msg = event.get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"OpenAI WebSocket error: {error_msg}")

                # --- Safely ignorable events ---
                elif event_type in _IGNORED_EVENTS:
                    pass

        except websockets.exceptions.ConnectionClosed as e:
            self.done = True
            raise RuntimeError(f"WebSocket connection closed during streaming: {e}") from e


# ---------------------------------------------------------------------------
# Cancel / drain
# ---------------------------------------------------------------------------


async def _cancel_and_drain(stream: _WsEventStream, ws: Any) -> None:
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
        except Exception:
            pass
        # Re-raise CancelledError / KeyboardInterrupt so the caller's
        # cancellation semantics are preserved.
        if not isinstance(e, Exception):
            raise


# ---------------------------------------------------------------------------
# _WsChatStream — async context manager + async iterable
# ---------------------------------------------------------------------------


class _WsChatStream:
    """Async context manager and async iterable for a streaming response.

    Supports two usage patterns::

        # Pattern 1: async with (recommended — deterministic cleanup)
        async with provider.chat(...) as stream:
            async for chunk in stream:
                ...

        # Pattern 2: bare async for (cleanup via generator finalization)
        async for chunk in provider.chat(...):
            ...
    """

    def __init__(self, setup: Awaitable[Tuple[_WsEventStream, Callable[[], Awaitable[None]]]]):
        self._setup = setup
        self._stream: Optional[_WsEventStream] = None
        self._cleanup_fn: Optional[Callable[[], Awaitable[None]]] = None

    # --- Async context manager ---

    async def __aenter__(self) -> "_WsChatStream":
        self._stream, self._cleanup_fn = await self._setup
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._cleanup_fn is not None:
            await self._cleanup_fn()
            self._cleanup_fn = None

    # --- Async iterable ---

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        if self._stream is not None:
            # Inside ``async with`` — iterate the already-initialized stream.
            return self._stream.__aiter__()
        # Standalone ``async for`` — manage lifecycle internally.
        return self._standalone_iter()

    async def _standalone_iter(self) -> AsyncIterator[StreamChunk]:
        async with self:
            async for chunk in self._stream:
                yield chunk
