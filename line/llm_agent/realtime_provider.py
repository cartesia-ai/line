"""
OpenAI Realtime WebSocket Provider (text-only) with diff-sync.

Provides a persistent WebSocket connection to the OpenAI Realtime API,
exposing the same chat() -> stream interface as LLMProvider. Uses a diff-sync
mechanism to keep the WS conversation state aligned with the SDK's message list,
which may diverge due to interruptions/TTS truncation.

Protocol reference: https://platform.openai.com/docs/api-reference/realtime
"""

import asyncio
from dataclasses import dataclass, field
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import uuid

from loguru import logger
import websockets
from websockets.protocol import State as WsState

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, StreamChunk, ToolCall
from line.llm_agent.schema_converter import build_parameters_schema
from line.llm_agent.tools.utils import FunctionTool

# If diff requires deleting more items than this, reconnect instead.
RECONNECT_THRESHOLD = 20

WS_URL = "wss://api.openai.com/v1/realtime"

# Timeout for waiting on server acknowledgement events (seconds).
WS_ACK_TIMEOUT = 30

# Timeout for draining events after cancelling a response (seconds).
WS_DRAIN_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Conversation state tracking
# ---------------------------------------------------------------------------


@dataclass
class _ConversationItem:
    """Client-side view of a single item in the WS conversation."""

    item_id: str
    # Identity key for diffing: (role, content_text, tool_call_id_or_name)
    identity: tuple


@dataclass
class _ConversationState:
    """Tracks the client-side view of the WS conversation items."""

    items: List[_ConversationItem] = field(default_factory=list)
    system_instructions: Optional[str] = None
    tool_defs: Optional[List[Dict[str, Any]]] = None


# Backward-compat alias for external references.
ConversationState = _ConversationState


@dataclass
class _DeleteOp:
    item_id: str


@dataclass
class _CreateOp:
    item: Dict[str, Any]
    message: Message  # Original message for state tracking


@dataclass
class _SessionUpdateOp:
    instructions: Optional[str]
    tools: Optional[List[Dict[str, Any]]]
    temperature: Optional[float]
    max_tokens: Optional[int]


@dataclass
class _Reconnect:
    """Sentinel: diff is too large, reconnect and rebuild."""

    pass


DiffOp = Union[_DeleteOp, _CreateOp, _SessionUpdateOp, _Reconnect]

# ---------------------------------------------------------------------------
# RealtimeProvider
# ---------------------------------------------------------------------------


class RealtimeProvider:
    """
    OpenAI Realtime WebSocket provider with diff-sync.

    Exposes the same chat() -> stream interface as LLMProvider so it can be
    used as a drop-in replacement in LlmAgent.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
    ):
        self._model = model
        self._api_key = api_key or ""
        self._config = config or LlmConfig()
        self._ws: Optional[Any] = None  # websockets ClientConnection
        self._state = _ConversationState()
        # Lock to serialise WS operations (connect, diff, response, stream iteration)
        self._lock = asyncio.Lock()

    # --- Connection management ---

    async def _connect(self) -> None:
        """Open a new WS connection and wait for session.created."""
        url = f"{WS_URL}?model={self._model}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=2**24,  # 16 MB
        )
        # Wait for session.created
        msg = json.loads(await self._ws.recv())
        if msg.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {msg.get('type')}")
        self._state = _ConversationState()
        logger.debug("Realtime WS connected, session created")

    def _is_closed(self) -> bool:
        """Check if the WS connection is closed (compatible with websockets v13+)."""
        if self._ws is None:
            return True
        return self._ws.state is WsState.CLOSED or self._ws.state is WsState.CLOSING

    async def _ensure_connected(self) -> None:
        """Lazy connect / reconnect if the socket is closed."""
        if self._is_closed():
            await self._connect()

    async def _reconnect(self) -> None:
        """Close existing connection and open a fresh one."""
        await self._close_ws()
        await self._connect()

    async def _close_ws(self) -> None:
        if self._ws is not None and not self._is_closed():
            await self._ws.close()
        self._ws = None
        self._state = _ConversationState()

    # --- Receive helper ---

    async def _recv_until(self, expected_type: str, error_context: str) -> dict:
        """Receive WS events until the expected type arrives.

        Raises on timeout, connection closure, or server error events.
        Unexpected event types are logged and skipped.
        """
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        deadline = asyncio.get_event_loop().time() + WS_ACK_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting for {expected_type}")
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except websockets.exceptions.ConnectionClosed as e:
                raise RuntimeError(
                    f"WebSocket closed while waiting for {expected_type}: {e}"
                ) from e
            msg = json.loads(raw)
            if msg.get("type") == expected_type:
                return msg
            if msg.get("type") == "error":
                error_detail = msg.get("error", {}).get("message", "unknown error")
                raise RuntimeError(f"{error_context} error: {error_detail}")
            logger.debug(
                "Skipping event %s while waiting for %s",
                msg.get("type"),
                expected_type,
            )

    # --- Session update ---

    async def _send_session_update(self, op: _SessionUpdateOp) -> None:
        """Send session.update to configure modalities, instructions, tools, etc."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        session: Dict[str, Any] = {
            "modalities": ["text"],
        }
        if op.instructions is not None:
            session["instructions"] = op.instructions
            self._state.system_instructions = op.instructions
        if op.tools is not None:
            session["tools"] = op.tools
            self._state.tool_defs = op.tools
        if op.temperature is not None:
            session["temperature"] = op.temperature
        if op.max_tokens is not None:
            session["max_response_output_tokens"] = op.max_tokens

        await self._ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": session,
                }
            )
        )

        # Wait for session.updated acknowledgement
        await self._recv_until("session.updated", "session.update")

    # --- Item operations ---

    async def _create_item(self, item_dict: Dict[str, Any], message: Message) -> str:
        """Send conversation.item.create, wait for created ack, update state."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        # Let the server assign the ID, or we assign one for tracking
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        item_dict["id"] = item_id

        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": item_dict,
                }
            )
        )

        # Wait for conversation.item.created
        ack = await self._recv_until("conversation.item.created", "conversation.item.create")
        actual_id = ack.get("item", {}).get("id", item_id)
        self._state.items.append(
            _ConversationItem(
                item_id=actual_id,
                identity=_message_identity(message),
            )
        )
        return actual_id

    async def _delete_item(self, item_id: str) -> None:
        """Send conversation.item.delete, wait for deleted ack, update state."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.delete",
                    "item_id": item_id,
                }
            )
        )

        # Wait for conversation.item.deleted
        await self._recv_until("conversation.item.deleted", "conversation.item.delete")
        self._state.items = [item for item in self._state.items if item.item_id != item_id]

    # --- Execute diff ---

    async def _execute_diff(
        self,
        ops: List[DiffOp],
        desired_messages: List[Message],
        tools: Optional[List[FunctionTool]],
        config: LlmConfig,
        _depth: int = 0,
    ) -> None:
        """Execute a list of diff operations against the WS session."""
        for op in ops:
            if isinstance(op, _Reconnect):
                if _depth > 0:
                    raise RuntimeError("Reconnect loop detected in diff execution")
                await self._reconnect()
                # After reconnect, rebuild everything from scratch
                rebuild_ops = _diff_messages(self._state, desired_messages, tools, config)
                await self._execute_diff(
                    rebuild_ops, desired_messages, tools, config, _depth=_depth + 1
                )
                return
            elif isinstance(op, _SessionUpdateOp):
                await self._send_session_update(op)
            elif isinstance(op, _DeleteOp):
                await self._delete_item(op.item_id)
            elif isinstance(op, _CreateOp):
                await self._create_item(op.item, op.message)

    # --- Public interface ---

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        config: Optional[LlmConfig] = None,
        **kwargs,
    ) -> "_RealtimeChatStream":
        """Start a streaming chat completion over the Realtime WS.

        Returns a _RealtimeChatStream with the same protocol as _ChatStream.
        All diff/sync logic runs here; the stream is just a conversion layer.

        The lock is held across the entire stream lifetime to prevent
        interleaved reads. It is released in _RealtimeChatStream.__aexit__.
        """
        cfg = config or self._config

        await self._lock.acquire()
        try:
            await self._ensure_connected()
            ops = _diff_messages(self._state, messages, tools, cfg)
            await self._execute_diff(ops, messages, tools, cfg)
            if self._ws is None:
                raise RuntimeError("WebSocket not connected after diff execution")
            await self._ws.send(json.dumps({"type": "response.create"}))
        except BaseException:
            self._lock.release()
            raise

        # Lock ownership transfers to the stream — released in __aexit__
        return _RealtimeChatStream(self._ws, self._state, self._lock)

    async def warmup(self, config=None):
        """Pre-establish WS connection and send system prompt.

        Overlaps the ~1100ms handshake with introduction audio so the first
        chat() call sees the connection already open and instructions set.
        """
        async with self._lock:
            await self._ensure_connected()
            if config is not None:
                if config.system_prompt:
                    await self._send_session_update(
                        _SessionUpdateOp(
                            instructions=config.system_prompt,
                            tools=None,
                            temperature=config.temperature,
                            max_tokens=config.max_tokens,
                        )
                    )

    async def aclose(self) -> None:
        """Close the WebSocket connection."""
        await self._close_ws()


# ---------------------------------------------------------------------------
# _RealtimeChatStream
# ---------------------------------------------------------------------------


class _RealtimeChatStream:
    """Async context manager matching the _ChatStream interface for Realtime WS.

    A thin conversion layer that reads raw WS events and yields StreamChunks.
    All connection management and diff-sync is handled by RealtimeProvider
    before this stream starts iterating.

    Holds the provider's lock for the duration of iteration, releasing it in
    ``__aexit__``.  If the stream is abandoned (not fully consumed),
    ``__aexit__`` cancels the in-progress response and drains remaining events.
    """

    def __init__(self, ws: Any, state: _ConversationState, lock: asyncio.Lock):
        self._ws = ws
        self._state = state
        self._lock = lock
        self._done = False

    async def __aenter__(self) -> "_RealtimeChatStream":
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
                    # Track state for items completed before cancellation
                    if event_type == "response.output_item.done":
                        _track_output_item(self._state, event.get("item", {}))
                    if event_type == "response.done":
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
                if event_type == "response.text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        yield StreamChunk(text=delta)

                # --- Function call argument deltas ---
                elif event_type == "response.function_call_arguments.delta":
                    call_id = event.get("call_id", "")
                    delta = event.get("delta", "")
                    if call_id not in tool_calls:
                        # We get the name from response.output_item.added or the done event
                        tool_calls[call_id] = ToolCall(
                            id=call_id,
                            name=event.get("name", ""),
                            arguments=delta,
                        )
                    else:
                        tool_calls[call_id].arguments += delta

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

                    # Update conversation state
                    _track_output_item(self._state, item)

                # --- Function call arguments done (name available) ---
                elif event_type == "response.function_call_arguments.done":
                    call_id = event.get("call_id", "")
                    if call_id in tool_calls:
                        tool_calls[call_id].arguments = event.get(
                            "arguments", tool_calls[call_id].arguments
                        )
                        tool_calls[call_id].name = event.get("name", tool_calls[call_id].name)

                # --- Output item added (get function name early) ---
                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id", "")
                        name = item.get("name", "")
                        if call_id and call_id not in tool_calls:
                            tool_calls[call_id] = ToolCall(id=call_id, name=name, arguments="")

                # --- Response done ---
                elif event_type == "response.done":
                    self._done = True
                    # Mark all tool calls as complete
                    for tc in tool_calls.values():
                        tc.is_complete = True

                    yield StreamChunk(
                        tool_calls=list(tool_calls.values()) if tool_calls else [],
                        is_final=True,
                    )
                    return

                # --- Error ---
                elif event_type == "error":
                    error_msg = event.get("error", {}).get("message", "unknown error")
                    raise RuntimeError(f"Realtime API error: {error_msg}")

                # --- Rate limit / other events we can safely ignore ---
                elif event_type in (
                    "response.created",
                    "response.content_part.added",
                    "response.content_part.done",
                    "response.text.done",
                    "rate_limits.updated",
                ):
                    pass

        except websockets.exceptions.ConnectionClosed as e:
            raise RuntimeError(f"WebSocket connection closed during streaming: {e}") from e


# ---------------------------------------------------------------------------
# State tracking helper
# ---------------------------------------------------------------------------


def _track_output_item(state: _ConversationState, item: Dict[str, Any]) -> None:
    """Update conversation state for a completed output item."""
    if item.get("type") == "function_call":
        state.items.append(
            _ConversationItem(
                item_id=item.get("id", item.get("call_id", "")),
                identity=(
                    "assistant_tool_call",
                    item.get("name", ""),
                    item.get("arguments", ""),
                    item.get("call_id", ""),
                ),
            )
        )
    elif item.get("type") == "message" and item.get("role") == "assistant":
        content_parts = item.get("content", [])
        text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
        state.items.append(
            _ConversationItem(
                item_id=item.get("id", ""),
                identity=("assistant", text, "", ""),
            )
        )


# ---------------------------------------------------------------------------
# Diff algorithm
# ---------------------------------------------------------------------------


def _message_identity(msg: Message) -> tuple:
    """Compute an identity key for a message (for diffing)."""
    if msg.tool_calls:
        # Assistant message with tool calls — identity is the first tool call
        tc = msg.tool_calls[0]
        return ("assistant_tool_call", tc.name, tc.arguments, tc.id)
    return (msg.role, msg.content or "", msg.tool_call_id or "", msg.name or "")


def _message_to_item(msg: Message) -> Dict[str, Any]:
    """Convert a Message to a Realtime API conversation item dict.

    Note: for assistant messages with multiple tool calls, only the first
    tool call is converted.  The Realtime API represents each tool call as a
    separate conversation item, but the diff algorithm tracks identity at the
    message level.  Handling multi-tool-call expansion here would require
    reworking the diff model.
    """
    if msg.role == "user":
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": msg.content or ""}],
        }

    if msg.role == "assistant":
        if msg.tool_calls:
            # Tool call — each tool call becomes a function_call item
            tc = msg.tool_calls[0]
            return {
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": msg.content or ""}],
        }

    if msg.role == "tool":
        return {
            "type": "function_call_output",
            "call_id": msg.tool_call_id or "",
            "output": msg.content or "",
        }

    # system messages should not reach here — filtered by _diff_messages
    raise ValueError(f"Unsupported message role for realtime item: {msg.role}")


def _diff_messages(
    current_state: _ConversationState,
    desired_messages: List[Message],
    tools: Optional[List[FunctionTool]],
    config: LlmConfig,
) -> List[DiffOp]:
    """Compute minimal diff ops to sync WS conversation with desired messages.

    System messages are handled via session.update (instructions), not items.
    """
    ops: List[DiffOp] = []

    # --- Separate system messages from conversation items ---
    system_parts: List[str] = []
    non_system: List[Message] = []
    for msg in desired_messages:
        if msg.role == "system":
            system_parts.append(msg.content or "")
        else:
            non_system.append(msg)

    desired_instructions = "\n\n".join(system_parts) if system_parts else None

    # --- Session update (instructions, tools, sampling params) ---
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

    needs_session_update = (
        desired_instructions != current_state.system_instructions or tool_defs != current_state.tool_defs
    )
    if needs_session_update:
        ops.append(
            _SessionUpdateOp(
                instructions=desired_instructions,
                tools=tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )

    # --- Diff conversation items ---
    current_items = current_state.items
    desired_identities = [_message_identity(m) for m in non_system]
    current_identities = [item.identity for item in current_items]

    # Find longest common prefix
    prefix_len = 0
    for i in range(min(len(current_identities), len(desired_identities))):
        if current_identities[i] == desired_identities[i]:
            prefix_len = i + 1
        else:
            break

    # Items to delete (suffix after common prefix)
    to_delete = current_items[prefix_len:]
    # Items to create (suffix after common prefix)
    to_create = non_system[prefix_len:]

    if len(to_delete) > RECONNECT_THRESHOLD:
        return [_Reconnect()]

    # Delete in reverse order (last items first)
    for item in reversed(to_delete):
        ops.append(_DeleteOp(item_id=item.item_id))

    # Create new items
    for msg in to_create:
        ops.append(_CreateOp(item=_message_to_item(msg), message=msg))

    return ops
