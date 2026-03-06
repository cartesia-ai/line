"""
OpenAI Realtime WebSocket Provider (text-only) with diff-sync.

Provides a persistent WebSocket connection to the OpenAI Realtime API,
exposing the same chat() -> stream interface as LLMProvider. Uses a diff-sync
mechanism to keep the WS conversation state aligned with the SDK's message list,
which may diverge due to interruptions/TTS truncation.

Protocol reference: https://platform.openai.com/docs/api-reference/realtime
"""

import asyncio
from dataclasses import dataclass
import json
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, Union
import uuid

from loguru import logger
import websockets
from websockets.protocol import State as WsState

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, StreamChunk, _extract_instructions_and_messages
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.stream import (
    ConversationEntry,
    _compute_divergence,
    _context_identity,
    _expand_messages,
    _iterate_ws_query,
    _WsEventStream,
)
from line.llm_agent.tools.utils import FunctionTool

# If diff requires deleting more items than this, reconnect instead.
RECONNECT_THRESHOLD = 20

WS_URL = "wss://api.openai.com/v1/realtime"

# Timeout for waiting on server acknowledgement events (seconds).
WS_ACK_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Conversation state tracking
# ---------------------------------------------------------------------------


@dataclass
class _DeleteOp:
    item_id: str


@dataclass
class _CreateOp:
    item: Dict[str, Any]
    identity: tuple  # Identity fingerprint for state tracking


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


class _RealtimeProvider:
    """
    OpenAI Realtime WebSocket provider with diff-sync.

    Exposes the same chat() -> stream interface as LlmProvider so it can be
    used as a drop-in replacement in LlmAgent.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key or ""
        self._ws: Optional[Any] = None  # websockets ClientConnection
        self._history: List[ConversationEntry] = []
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
            ping_interval=20,
            ping_timeout=20,
        )
        # Wait for session.created
        msg = json.loads(await self._ws.recv())
        if msg.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {msg.get('type')}")
        self._history = []
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
        if not self._is_closed():
            await self._ws.close()
        self._history = []

    # --- Receive helper ---

    async def _recv_until(self, expected_type: str, error_context: str) -> dict:
        """Receive WS events until the expected type arrives.

        Raises on timeout, connection closure, or server error events.
        Unexpected event types are logged and skipped.
        """
        deadline = asyncio.get_running_loop().time() + WS_ACK_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting for {expected_type}")
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except websockets.exceptions.ConnectionClosed as e:
                raise RuntimeError(f"WebSocket closed while waiting for {expected_type}: {e}") from e
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
        session: Dict[str, Any] = {
            "modalities": ["text"],
            "instructions": op.instructions,
            "tools": op.tools,
            "temperature": op.temperature,
            "max_response_output_tokens": op.max_tokens,
        }

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

        context_id = _context_identity(
            op.instructions,
            op.tools,
            temperature=op.temperature,
            max_tokens=op.max_tokens,
        )
        _, item_history = _split_context_history(self._history)
        self._history = [(context_id, None)] + item_history

    # --- Item operations ---

    async def _create_item(self, item_dict: Dict[str, Any], identity: tuple) -> str:
        """Send conversation.item.create, wait for created ack, update state."""
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
        self._history.append((identity, actual_id))
        return actual_id

    async def _delete_item(self, item_id: str) -> None:
        """Send conversation.item.delete, wait for deleted ack, update state."""
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
        self._history = [item for item in self._history if item[1] != item_id]

    # --- Execute diff ---

    async def _execute_diff(
        self,
        ops: List[DiffOp],
        desired_messages: List[Message],
        tools: Optional[List[FunctionTool]],
        config: LlmConfig,
        web_search_options: Optional[Dict[str, Any]] = None,
        _depth: int = 0,
    ) -> None:
        """Execute a list of diff operations against the WS session."""
        for op in ops:
            if isinstance(op, _Reconnect):
                if _depth > 0:
                    raise RuntimeError("Reconnect loop detected in diff execution")
                await self._reconnect()
                # After reconnect, rebuild everything from scratch
                rebuild_ops = _diff_messages(
                    self._history,
                    desired_messages,
                    tools,
                    config,
                    web_search_options=web_search_options,
                )
                await self._execute_diff(
                    rebuild_ops,
                    desired_messages,
                    tools,
                    config,
                    web_search_options=web_search_options,
                    _depth=_depth + 1,
                )
                return
            elif isinstance(op, _SessionUpdateOp):
                await self._send_session_update(op)
            elif isinstance(op, _DeleteOp):
                await self._delete_item(op.item_id)
            elif isinstance(op, _CreateOp):
                await self._create_item(op.item, op.identity)

    # --- Public interface ---

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        *,
        config: LlmConfig,
        **kwargs,
    ) -> AsyncIterable[StreamChunk]:
        """Start a streaming chat completion over the Realtime WS.

        Returns one async iterator per request/response. The provider owns the
        full lifecycle for that iterator: setup, streaming, cancellation/drain,
        and lock release.
        """

        async def _chat():
            async for chunk in _iterate_ws_query(
                self._setup_chat(
                    messages,
                    tools,
                    config,
                    web_search_options=kwargs.get("web_search_options"),
                )
            ):
                yield chunk

        return _chat()

    async def _setup_chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]],
        config: LlmConfig,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Async setup for a single chat stream attempt."""
        await self._lock.acquire()
        try:
            await self._ensure_connected()
            ops = _diff_messages(
                self._history,
                messages,
                tools,
                config,
                web_search_options=web_search_options,
            )
            await self._execute_diff(
                ops,
                messages,
                tools,
                config,
                web_search_options=web_search_options,
            )
            await self._ws.send(json.dumps({"type": "response.create"}))
        except BaseException:
            self._lock.release()
            raise

        # Lock ownership transfers to the stream — released in __aexit__
        def on_response_done(response: Dict[str, Any]) -> None:
            for item in response.get("output", []):
                _track_output_item(self._history, item)

        event_stream = _WsEventStream(self._ws, on_response_done)
        return event_stream, self._ws, self._lock

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[FunctionTool]] = None,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
    ):
        """Pre-establish WS connection and send system prompt + tools.

        Overlaps the ~1100ms handshake with introduction audio so the first
        chat() call sees the connection already open and instructions set.
        """
        tool_defs = build_openai_tool_defs(
            tools,
            web_search_options=web_search_options,
            strict=False,
            responses_api=True,
        )

        async with self._lock:
            await self._ensure_connected()
            context_id = _context_identity(
                config.system_prompt,
                tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            current_context, _ = _split_context_history(self._history)
            if current_context != context_id:
                await self._send_session_update(
                    _SessionUpdateOp(
                        instructions=config.system_prompt,
                        tools=tool_defs,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                )

    async def aclose(self) -> None:
        """Close the WebSocket connection.

        Acquires the lock so that an in-flight ``chat()`` stream finishes
        (or is drained) before the connection is torn down.
        """
        async with self._lock:
            await self._close_ws()


# ---------------------------------------------------------------------------
# State tracking helper
# ---------------------------------------------------------------------------


def _track_output_item(history: List[ConversationEntry], item: Dict[str, Any]) -> None:
    """Update conversation state for a completed output item."""
    if item.get("type") == "function_call":
        history.append(
            (
                (
                    "assistant_tool_call",
                    ((item.get("name", ""), item.get("arguments", ""), item.get("call_id", "")),),
                ),
                item.get("id", item.get("call_id", "")),
            )
        )
    elif item.get("type") == "message" and item.get("role") == "assistant":
        content_parts = item.get("content", [])
        text = "".join(p.get("text", "") for p in content_parts if p.get("type") == "text")
        history.append(
            (
                ("assistant", text, "", ""),
                item.get("id", ""),
            )
        )


def _split_context_history(
    history: List[ConversationEntry],
) -> Tuple[Optional[tuple], List[ConversationEntry]]:
    """Return ``(context_identity, item_history)`` from a history list."""
    if history and history[0][0][0] == "__context__":
        return history[0][0], history[1:]
    return None, history


# ---------------------------------------------------------------------------
# Diff algorithm
# ---------------------------------------------------------------------------


def _diff_messages(
    current_history: List[ConversationEntry],
    desired_messages: List[Message],
    tools: Optional[List[FunctionTool]],
    config: LlmConfig,
    *,
    web_search_options: Optional[Dict[str, Any]] = None,
) -> List[DiffOp]:
    """Compute minimal diff ops to sync WS conversation with desired messages.

    System messages are handled via session.update (instructions), not items.
    """
    ops: List[DiffOp] = []

    # --- Separate system messages from conversation items ---
    desired_instructions, non_system = _extract_instructions_and_messages(desired_messages, config)

    # --- Session update (instructions, tools, sampling params) ---
    tool_defs = build_openai_tool_defs(
        tools,
        web_search_options=web_search_options,
        strict=False,
        responses_api=True,
    )
    desired_context = _context_identity(
        desired_instructions,
        tool_defs,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    current_context, current_items = _split_context_history(current_history)
    if current_context != desired_context:
        ops.append(
            _SessionUpdateOp(
                instructions=desired_instructions,
                tools=tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )

    # --- Diff conversation items ---
    # Flatten messages to (item_dict, identity) pairs — a single Message
    # may expand to multiple items (e.g. parallel tool calls).
    desired_pairs = _expand_messages(non_system, assistant_text_type="text")

    current_identities = [identity for identity, _ in current_items]
    prefix_len, to_create = _compute_divergence(current_identities, desired_pairs)

    # Items to delete (suffix after common prefix)
    to_delete = current_items[prefix_len:]

    if len(to_delete) > RECONNECT_THRESHOLD:
        return [_Reconnect()]

    # Delete in reverse order (last items first)
    for item in reversed(to_delete):
        ops.append(_DeleteOp(item_id=item[1] or ""))

    # Create new items
    for item_dict, identity in to_create:
        ops.append(_CreateOp(item=item_dict, identity=identity))

    return ops
