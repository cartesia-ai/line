"""
Shared stream utilities for WebSocket-based providers.

Contains ``_WsEventStream`` — the unified event→chunk translator used by both
``RealtimeProvider`` and ``WebSocketProvider`` — along with
``_iterate_ws_query`` (shared per-request lifecycle management) and the
``WS_DRAIN_TIMEOUT`` constant.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Dict

from loguru import logger
import websockets
from websockets.protocol import State as WsState

from line.llm_agent.provider import Message, StreamChunk, ToolCall


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
            raise RuntimeError(f"WebSocket connection closed during streaming: {e}") from e


async def _iterate_ws_query(setup: Awaitable[tuple[Any, Any, asyncio.Lock]]) -> AsyncIterator[StreamChunk]:
    """Run one WebSocket query/response stream with shared cleanup.

    ``setup`` must yield ``(_WsEventStream, ws, lock)``. The lock is always
    released, and unfinished responses are cancelled and drained so the next
    request on the socket starts from a clean state.
    """
    stream, ws, lock = await setup
    try:
        async for chunk in stream:
            yield chunk
    finally:
        try:
            if not stream.done:
                await _cancel_and_drain(stream, ws)
        finally:
            lock.release()


async def _cancel_and_drain(stream: Any, ws: Any) -> None:
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
