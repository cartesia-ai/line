"""
OpenAI WebSocket Mode Provider for the Responses API.

Uses a persistent WebSocket connection to wss://api.openai.com/v1/responses
for lower-latency agentic workflows.  Instead of separate HTTP requests per
turn, we keep the socket open and use ``previous_response_id`` for efficient
multi-turn continuations.

Key differences from the Realtime API (``realtime_provider.py``):
  * Works with standard text models (gpt-5-nano, gpt-5-mini, gpt-5.2, …)
    rather than dedicated realtime models.
  * No conversation-item CRUD — the server manages state via
    ``previous_response_id``.
  * Divergence detection (e.g. TTS interruption truncating an assistant
    message) is handled by comparing identity fingerprints.  When the
    SDK's history diverges from the server's view we fall back to a fresh
    request with the full context.

Protocol reference: https://developers.openai.com/api/docs/guides/websocket-mode
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
import websockets
from websockets.protocol import State as WsState

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, StreamChunk, ToolCall
from line.llm_agent.schema_converter import build_parameters_schema
from line.llm_agent.tools.utils import FunctionTool

WS_URL = "wss://api.openai.com/v1/responses"

# Timeout for draining events after cancelling a response (seconds).
WS_DRAIN_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Identity helpers  (for divergence detection)
# ---------------------------------------------------------------------------


def _message_identity(msg: Message) -> tuple:
    """Compute an identity fingerprint for a single Message.

    For assistant messages with tool calls, identity is derived from the
    *first* tool call (mirrors how the server tracks multi-tool-call turns
    as a single logical unit).
    """
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return ("assistant_tool_call", tc.name, tc.arguments, tc.id)
    return (msg.role, msg.content or "", msg.tool_call_id or "", msg.name or "")


def _compute_identities(messages: List[Message]) -> List[tuple]:
    """Compute an ordered identity list for a message sequence."""
    return [_message_identity(m) for m in messages]


def _common_prefix_len(a: List[tuple], b: List[tuple]) -> int:
    """Return the length of the longest common prefix of two identity lists."""
    n = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            n = i + 1
        else:
            break
    return n


def _extract_model_output_identity(response: Dict[str, Any]) -> Optional[tuple]:
    """Derive a single message-level identity from a Responses API output.

    Mirrors ``_message_identity``: if the model produced tool calls we key
    on the first one; otherwise we key on the full text.
    """
    output_items = response.get("output", [])
    function_calls = [i for i in output_items if i.get("type") == "function_call"]

    if function_calls:
        fc = function_calls[0]
        return (
            "assistant_tool_call",
            fc.get("name", ""),
            fc.get("arguments", ""),
            fc.get("call_id", ""),
        )

    # Concatenate text across all message output items.
    text_parts: List[str] = []
    for item in output_items:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

    if text_parts:
        return ("assistant", "".join(text_parts), "", "")

    return None


# ---------------------------------------------------------------------------
# WebSocketProvider
# ---------------------------------------------------------------------------


class WebSocketProvider:
    """
    OpenAI WebSocket mode provider (Responses API).

    Exposes the same ``chat() -> stream`` interface as ``LLMProvider`` so it
    can be used as a drop-in replacement in ``LlmAgent``.

    Divergence handling
    -------------------
    In a voice-agent pipeline the SDK may truncate an assistant message when
    the user interrupts (barge-in).  The next ``chat()`` call will carry the
    *truncated* text in its history, while the server — via
    ``previous_response_id`` — still holds the *full* text.

    We detect this by maintaining ``_server_convo``, an identity list that
    reflects the server's view of the conversation (our sent messages **plus**
    the model's outputs).  On each ``chat()`` call we compare the desired
    identities with ``_server_convo``.  If the common prefix is shorter than
    ``_server_convo`` (i.e. a previous message was modified or removed) we
    roll back to the latest ``_response_checkpoints`` entry that still falls
    within the shared prefix, preserving server-side context up to that
    point instead of resending the entire conversation from scratch.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        default_reasoning_effort: str = "none",
    ):
        self._model = model
        self._api_key = api_key or ""
        self._config = config or LlmConfig()
        self._default_reasoning_effort = default_reasoning_effort
        self._ws: Optional[Any] = None  # websockets ClientConnection

        # --- continuation / divergence state ---
        self._last_response_id: Optional[str] = None
        # Identity list reflecting what the server "knows" (inputs + outputs).
        self._server_convo: List[tuple] = []
        # Set at chat()-time, finalised in the stream's response.completed.
        self._pending_convo: List[tuple] = []
        # Checkpoint history: [(server_convo_len, response_id), ...].
        # On divergence we roll back to the latest checkpoint still within the
        # shared prefix instead of discarding all server state.
        self._response_checkpoints: List[tuple] = []

        # Lock to serialise WS operations (connect, send request, stream iteration)
        self._lock = asyncio.Lock()

    # --- Connection management ---

    async def _connect(self) -> None:
        """Open a new WebSocket connection to the Responses API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        self._ws = await websockets.connect(
            WS_URL,
            additional_headers=headers,
            max_size=2**24,  # 16 MB
        )
        # Unlike the Realtime API there is no session.created handshake.
        # Stored responses persist server-side across connections, so we
        # intentionally do NOT reset continuation state on reconnect.
        logger.debug("WebSocket mode connected to Responses API")

    def _is_closed(self) -> bool:
        if self._ws is None:
            return True
        return self._ws.state in (WsState.CLOSED, WsState.CLOSING)

    async def _ensure_connected(self) -> None:
        """Lazy connect / reconnect if the socket is closed."""
        if self._is_closed():
            await self._connect()

    async def _reconnect(self) -> None:
        """Close and reopen the WebSocket (preserves response state)."""
        if self._ws is not None and not self._is_closed():
            await self._ws.close()
        self._ws = None
        await self._connect()

    async def _close_ws(self) -> None:
        if self._ws is not None and not self._is_closed():
            await self._ws.close()
        self._ws = None
        self._reset_response_state()

    def _reset_response_state(self) -> None:
        """Reset all continuation state — next chat() sends full context."""
        self._last_response_id = None
        self._server_convo = []
        self._pending_convo = []
        self._response_checkpoints = []

    def _finalize_response(self, response: Dict[str, Any]) -> None:
        """Finalize conversation state after a completed response.

        Updates _server_convo, _last_response_id, and _response_checkpoints
        based on the model's output.
        """
        response_id = response.get("id")
        model_id = _extract_model_output_identity(response)
        convo = list(self._pending_convo)
        if model_id is not None:
            convo.append(model_id)
        self._server_convo = convo

        if response_id:
            self._last_response_id = response_id
            self._response_checkpoints.append((len(convo), response_id))

    # --- Public interface ---

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        config: Optional[LlmConfig] = None,
        **kwargs,
    ) -> "_WebSocketChatStream":
        """Start a streaming chat completion over WebSocket mode.

        Returns a ``_WebSocketChatStream`` with the same async-context-manager
        / async-iterator protocol as ``_ChatStream``.

        The lock is held across the entire stream lifetime to prevent
        interleaved reads. It is released in _WebSocketChatStream.__aexit__.
        """
        cfg = config or self._config

        await self._lock.acquire()
        try:
            await self._ensure_connected()

            # -- Separate system messages (→ instructions) from conversation
            system_parts: List[str] = []
            non_system: List[Message] = []
            for msg in messages:
                if msg.role == "system":
                    system_parts.append(msg.content or "")
                else:
                    non_system.append(msg)

            instructions = "\n\n".join(system_parts) if system_parts else None

            # -- Tool definitions
            tool_defs: Optional[List[Dict[str, Any]]] = None
            if tools:
                tool_defs = [
                    {
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": build_parameters_schema(t.parameters),
                    }
                    for t in tools
                ]

            # -- Find the best continuation point.
            # Compare server's view with desired messages; find the latest
            # checkpoint that falls within the shared prefix.
            desired_ids = _compute_identities(non_system)
            prefix_len = _common_prefix_len(self._server_convo, desired_ids)

            shared_checkpoint = None
            for convo_len, resp_id in reversed(self._response_checkpoints):
                if convo_len <= prefix_len:
                    shared_checkpoint = (convo_len, resp_id)
                    break

            # -- Build request
            if shared_checkpoint is not None:
                ckpt_len, ckpt_resp_id = shared_checkpoint
                # Prune stale checkpoints past the continuation point.
                self._response_checkpoints = [
                    (cl, ri) for cl, ri in self._response_checkpoints if cl <= ckpt_len
                ]
                new_input = _messages_to_input_items(non_system[ckpt_len:])
                request: Dict[str, Any] = {
                    "type": "response.create",
                    "model": self._model,
                    "previous_response_id": ckpt_resp_id,
                    "input": new_input,
                }
                if prefix_len < len(self._server_convo):
                    logger.debug(
                        "WebSocket conversation diverged at index %d/%d, rolling back to checkpoint at %d",
                        prefix_len,
                        len(self._server_convo),
                        ckpt_len,
                    )
            else:
                # No valid checkpoint — send full context.
                input_items = _messages_to_input_items(non_system)
                request = {
                    "type": "response.create",
                    "model": self._model,
                    "input": input_items,
                }
                self._response_checkpoints = []

            # -- Always include instructions, tools, and sampling params
            #    (they may change between turns).
            if instructions is not None:
                request["instructions"] = instructions
            if tool_defs is not None:
                request["tools"] = tool_defs

            # Reasoning effort — default to "none" for low TTFT.
            reasoning_effort = cfg.reasoning_effort or self._default_reasoning_effort
            request["reasoning"] = {"effort": reasoning_effort}

            if cfg.temperature is not None:
                request["temperature"] = cfg.temperature
            if cfg.max_tokens is not None:
                request["max_output_tokens"] = cfg.max_tokens
            if cfg.top_p is not None:
                request["top_p"] = cfg.top_p

            # Store pending state — finalised by the stream on
            # response.completed so that _server_convo includes the model's
            # output identity.
            self._pending_convo = list(desired_ids)

            if self._ws is None:
                raise RuntimeError("WebSocket not connected")
            await self._ws.send(json.dumps(request))

        except BaseException:
            self._lock.release()
            raise

        # Lock ownership transfers to the stream — released in __aexit__
        return _WebSocketChatStream(self._ws, self, self._lock)

    async def warmup(self, config=None):
        """Pre-establish WebSocket connection.

        The Responses API sends instructions per-request (no session concept),
        so warmup only establishes the connection.
        """
        async with self._lock:
            await self._ensure_connected()

    async def aclose(self) -> None:
        """Close the WebSocket connection."""
        await self._close_ws()


# ---------------------------------------------------------------------------
# _WebSocketChatStream
# ---------------------------------------------------------------------------


class _WebSocketChatStream:
    """Async context manager / iterator matching the ``_ChatStream`` protocol.

    Reads raw Responses-API streaming events from the WebSocket and yields
    ``StreamChunk`` objects that ``LlmAgent`` already knows how to consume.

    On ``response.completed`` the stream finalises the provider's
    ``_server_convo`` so divergence detection works on the next turn.

    Holds the provider's lock for the duration of iteration, releasing it in
    ``__aexit__``.  If the stream is abandoned (not fully consumed),
    ``__aexit__`` cancels the in-progress response and drains remaining events.
    """

    def __init__(self, ws: Any, provider: WebSocketProvider, lock: asyncio.Lock):
        self._ws = ws
        self._provider = provider
        self._lock = lock
        self._done = False

    async def __aenter__(self) -> "_WebSocketChatStream":
        return self

    async def __aexit__(self, *args) -> None:
        try:
            if not self._done:
                await self._cancel_and_drain()
        finally:
            self._lock.release()

    async def _cancel_and_drain(self) -> None:
        """Cancel the in-progress response and drain remaining WS events."""
        ws = self._ws
        if ws is None or ws.state in (WsState.CLOSED, WsState.CLOSING):
            return

        try:
            await ws.send(json.dumps({"type": "response.cancel"}))
        except (websockets.exceptions.ConnectionClosed, Exception):
            return  # Connection already gone, nothing to drain

        try:

            async def _drain():
                while True:
                    raw = await ws.recv()
                    event = json.loads(raw)
                    event_type = event.get("type", "")
                    # Finalize state if the response completed before cancel took effect
                    if event_type == "response.completed":
                        response = event.get("response", {})
                        self._provider._finalize_response(response)
                        return
                    if event_type in ("response.failed", "response.done"):
                        return

            await asyncio.wait_for(_drain(), timeout=WS_DRAIN_TIMEOUT)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed, Exception) as e:
            # Drain failed — close WS so next chat() reconnects with clean state
            logger.debug("Stream drain failed (%s), closing WS for clean reconnect", type(e).__name__)
            try:
                await ws.close()
            except Exception:
                pass

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        ws = self._ws
        if ws is None or ws.state in (WsState.CLOSED, WsState.CLOSING):
            raise RuntimeError("WebSocket not connected")

        tool_calls: Dict[str, ToolCall] = {}  # keyed by call_id

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
                        tool_calls[call_id].arguments = event.get(
                            "arguments",
                            tool_calls[call_id].arguments,
                        )
                        tool_calls[call_id].name = event.get(
                            "name",
                            tool_calls[call_id].name,
                        )

                # --- Output item added (get function name early) ---
                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        name = item.get("name", "")
                        if call_id and call_id not in tool_calls:
                            tool_calls[call_id] = ToolCall(
                                id=call_id,
                                name=name,
                                arguments="",
                            )

                # --- Output item done (function call complete) ---
                elif event_type == "response.output_item.done":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        if call_id in tool_calls:
                            tool_calls[call_id].name = item.get(
                                "name",
                                tool_calls[call_id].name,
                            )
                            tool_calls[call_id].arguments = item.get(
                                "arguments",
                                tool_calls[call_id].arguments,
                            )
                            tool_calls[call_id].is_complete = True
                        else:
                            tool_calls[call_id] = ToolCall(
                                id=call_id,
                                name=item.get("name", ""),
                                arguments=item.get("arguments", ""),
                                is_complete=True,
                            )

                # --- Response completed ---
                elif event_type == "response.completed":
                    self._done = True
                    response = event.get("response", {})
                    self._provider._finalize_response(response)

                    for tc in tool_calls.values():
                        tc.is_complete = True

                    yield StreamChunk(
                        tool_calls=list(tool_calls.values()) if tool_calls else [],
                        is_final=True,
                    )
                    return

                # --- Response failed ---
                elif event_type == "response.failed":
                    self._done = True
                    error = event.get("response", {}).get("error", {})
                    error_msg = error.get("message", "unknown error")
                    error_code = error.get("code", "")

                    if error_code == "previous_response_not_found":
                        # Server-side cache miss — reset so next chat() sends
                        # full context from scratch.
                        self._provider._reset_response_state()

                    raise RuntimeError(f"WebSocket mode error ({error_code}): {error_msg}")

                # --- Error ---
                elif event_type == "error":
                    error_info = event.get("error", {})
                    error_msg = error_info.get("message", "unknown error")
                    raise RuntimeError(f"WebSocket mode error: {error_msg}")

                # --- Safely ignorable events ---
                elif event_type in (
                    "response.created",
                    "response.in_progress",
                    "response.content_part.added",
                    "response.content_part.done",
                    "response.output_text.done",
                    "response.output_text.annotation.added",
                    "rate_limits.updated",
                ):
                    pass

        except websockets.exceptions.ConnectionClosed as e:
            raise RuntimeError(f"WebSocket connection closed during streaming: {e}") from e


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _messages_to_input_items(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert a list of Messages to Responses API input items.

    A single Message may expand to multiple items (e.g. an assistant message
    with several parallel tool calls).
    """
    items: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.content or ""}],
                }
            )

        elif msg.role == "assistant":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            elif msg.content:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.content}],
                    }
                )

        elif msg.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content or "",
                }
            )

        else:
            raise ValueError(f"Unsupported message role for WebSocket input: {msg.role}")

    return items
