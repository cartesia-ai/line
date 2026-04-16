"""
OpenAI Realtime WebSocket Provider (text-only) with diff-sync.

Provides a persistent WebSocket connection to the OpenAI Realtime API,
exposing the same chat() -> stream interface as LLMProvider. Uses a diff-sync
mechanism to keep the WS conversation state aligned with the SDK's message list,
which may diverge due to interruptions/TTS truncation.

Protocol reference: https://platform.openai.com/docs/api-reference/realtime
"""

import asyncio
import json
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import uuid

from loguru import logger
import websockets
from websockets.legacy.client import WebSocketClientProtocol

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, ParsedModelId, _extract_instructions_and_messages
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.stream import (
    ConversationEntry,
    HistoryUpdate,
    _AsyncIterableContext,
    _cancel_and_drain,
    _compute_divergence,
    _context_identity,
    _expand_messages,
    _ws_connect,
    _ws_is_closed,
    _WsEventStream,
)
from line.llm_agent.tools.utils import FunctionTool

# If diff requires deleting more items than this, reconnect instead.
RECONNECT_THRESHOLD = 20

WS_URL = "wss://api.openai.com/v1/realtime"

# Timeout for waiting on server acknowledgement events (seconds).
WS_ACK_TIMEOUT = 30


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
        model_id: ParsedModelId,
        api_key: Optional[str] = None,
    ):
        self._model_id = model_id
        self._api_key = api_key or ""
        self._ws_url = f"{WS_URL}?model={model_id.model}"
        self._ws: Optional[WebSocketClientProtocol] = None
        self._history: List[ConversationEntry] = []
        # Lazy-init: asyncio.Lock() requires a running event loop on Python 3.9.
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        self._lock = self._lock or asyncio.Lock()
        return self._lock

    # --- Public interface ---

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        *,
        config: LlmConfig,
        **kwargs,
    ) -> _AsyncIterableContext:
        async def _iter():
            stream = await self._setup_chat(
                messages, tools, config, web_search_options=kwargs.get("web_search_options")
            )
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                try:
                    if not stream.done:
                        await _cancel_and_drain(stream, self._ws)
                finally:
                    self._get_lock().release()

        return _AsyncIterableContext(_iter)

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[FunctionTool]] = None,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with self._get_lock():
            await self._ensure_connected()
            instructions = config.system_prompt or None
            tool_defs = build_openai_tool_defs(
                tools,
                web_search_options=web_search_options,
                strict=config.strict_tool_schemas,
                responses_api=True,
            )
            desired_context = _context_identity(
                instructions,
                tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            current_context, _ = _split_context_history(self._history)
            if current_context == desired_context:
                return

            event, update = _make_session_update(
                instructions=instructions,
                tool_defs=tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            await self._ws.send(json.dumps(event))
            ack = await self._recv_ack()
            self._history = update(self._history, ack)

    async def aclose(self) -> None:
        async with self._get_lock():
            if not _ws_is_closed(self._ws):
                await self._ws.close()
            self._history = []

    # --- Helpers ---

    async def _ensure_connected(self) -> None:
        if _ws_is_closed(self._ws):
            self._ws = await _ws_connect(
                self._ws_url, self._api_key, extra_headers={"OpenAI-Beta": "realtime=v1"}
            )
            self._history = []
            msg = json.loads(await self._ws.recv())
            if msg.get("type") != "session.created":
                raise RuntimeError(f"Expected session.created, got {msg.get('type')}")
            logger.debug("Realtime WS connected, session created")

    async def _reconnect(self) -> None:
        if not _ws_is_closed(self._ws):
            await self._ws.close()
        self._history = []
        await self._ensure_connected()

    async def _setup_chat(self, messages, tools, config, *, web_search_options=None):
        lock = self._get_lock()
        await lock.acquire()
        try:
            await self._ensure_connected()

            plan = _plan_chat(
                history=self._history,
                messages=messages,
                tools=tools,
                config=config,
                web_search_options=web_search_options,
            )
            if plan.reconnect:
                await self._reconnect()

            for event, _ in plan.steps:
                await self._ws.send(json.dumps(event))

            for _, update in plan.steps:
                ack = await self._recv_ack()
                self._history = update(self._history, ack)
            await self._ws.send(json.dumps({"type": "response.create"}))
        except BaseException:
            lock.release()
            raise

        def on_response_done(response):
            if response.get("status") != "completed":
                return
            self._history = _track_output_items(self._history, response.get("output", []))

        return _WsEventStream(self._ws, on_response_done)

    async def _recv_ack(self) -> Dict[str, Any]:
        """Receive the next ack or error event, skipping non-ack events."""
        deadline = asyncio.get_running_loop().time() + WS_ACK_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timeout waiting for ack")
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except websockets.exceptions.ConnectionClosed as e:
                raise RuntimeError(f"WebSocket closed while waiting for ack: {e}") from e
            msg = json.loads(raw)
            if msg.get("type") == "error":
                error_detail = (msg.get("error") or {}).get("message", "unknown error")
                raise RuntimeError(f"Realtime API error: {error_detail}")
            msg_type = msg.get("type", "")
            if msg_type.endswith((".updated", ".created", ".deleted")) and msg_type.startswith(
                ("session.", "conversation.item.")
            ):
                return msg
            logger.debug("Skipping event {} while waiting for ack", msg.get("type"))


# ---------------------------------------------------------------------------
# Helpers for diff-sync planning and history tracking
# ---------------------------------------------------------------------------


def _track_output_items(
    history: List[ConversationEntry], output: List[Dict[str, Any]]
) -> List[ConversationEntry]:
    """Return updated history with entries for completed output items."""
    new_entries: List[ConversationEntry] = []
    for item in output:
        if item.get("type") == "function_call":
            new_entries.append(
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
            new_entries.append(
                (
                    ("assistant", text, "", ""),
                    item.get("id", ""),
                )
            )
    return history + new_entries


def _split_context_history(
    history: List[ConversationEntry],
) -> Tuple[Optional[tuple], List[ConversationEntry]]:
    """Return ``(context_identity, item_history)`` from a history list."""
    if history and history[0][0][0] == "__context__":
        return history[0][0], history[1:]
    return None, history


class _DiffResult(NamedTuple):
    """Pure output of the diff algorithm.

    ``reconnect``: caller must reconnect (clear server state) before sending.
    ``steps``: list of ``(ws_event, history_update)`` pairs.  The caller
    sends all events, then drains acks in order, applying each
    ``history_update(old_history, ack_payload) -> new_history``.

    When ``reconnect`` is True, ``steps`` contains the full rebuild
    (session update + all creates) — no second diff call needed.
    """

    reconnect: bool
    steps: List[Tuple[Dict[str, Any], HistoryUpdate]]


def _plan_chat(
    *,
    history: List[ConversationEntry],
    messages: List[Message],
    tools: Optional[List[FunctionTool]],
    config: LlmConfig,
    web_search_options: Optional[Dict[str, Any]] = None,
) -> _DiffResult:
    """Compute minimal WS messages to sync conversation with desired state."""
    desired_instructions, non_system = _extract_instructions_and_messages(messages, config)

    tool_defs = build_openai_tool_defs(
        tools,
        web_search_options=web_search_options,
        strict=config.strict_tool_schemas,
        responses_api=True,
    )
    desired_context = _context_identity(
        desired_instructions,
        tool_defs,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    current_context, current_items = _split_context_history(history)

    desired_pairs = _expand_messages(non_system, assistant_text_type="text")
    current_identities = [identity for identity, _ in current_items]
    prefix_len, to_create = _compute_divergence(current_identities, desired_pairs)

    to_delete = current_items[prefix_len:]
    if len(to_delete) > RECONNECT_THRESHOLD:
        # Too many deletes — reconnect and rebuild from scratch.
        steps: List[Tuple[Dict[str, Any], HistoryUpdate]] = [
            _make_session_update(
                instructions=desired_instructions,
                tool_defs=tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        ]
        for item_dict, identity in desired_pairs:
            steps.append(_make_create(item_dict, identity))
        return _DiffResult(reconnect=True, steps=steps)

    steps: List[Tuple[Dict[str, Any], HistoryUpdate]] = []

    if current_context != desired_context:
        steps.append(
            _make_session_update(
                instructions=desired_instructions,
                tool_defs=tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )

    for entry in reversed(to_delete):
        steps.append(_make_delete(entry[1] or ""))

    for item_dict, identity in to_create:
        steps.append(_make_create(item_dict, identity))

    return _DiffResult(reconnect=False, steps=steps)


def _make_session_update(
    *,
    instructions: Optional[str],
    tool_defs: Optional[List[Dict[str, Any]]],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[Dict[str, Any], HistoryUpdate]:
    """Build a session.update event and its history update function.

    The Realtime API rejects ``null`` for every field, so we omit
    ``None``-valued fields and use ``""`` / ``[]`` as "clear" sentinels
    for instructions and tools.
    """
    session: Dict[str, Any] = {
        "modalities": ["text"],
        "instructions": instructions or "",
        "tools": tool_defs or [],
    }
    if temperature is not None:
        session["temperature"] = temperature
    if max_tokens is not None:
        session["max_response_output_tokens"] = max_tokens
    event = {"type": "session.update", "session": session}
    context_id = _context_identity(instructions, tool_defs, temperature=temperature, max_tokens=max_tokens)

    def update(history: List[ConversationEntry], _ack: Dict[str, Any]) -> List[ConversationEntry]:
        _, item_history = _split_context_history(history)
        return [(context_id, None)] + item_history

    return event, update


def _make_delete(item_id: str) -> Tuple[Dict[str, Any], HistoryUpdate]:
    """Build a conversation.item.delete event and its history update function."""
    event = {
        "type": "conversation.item.delete",
        "item_id": item_id,
    }

    def update(history: List[ConversationEntry], _ack: Dict[str, Any]) -> List[ConversationEntry]:
        return [entry for entry in history if entry[1] != item_id]

    return event, update


def _make_create(
    item_dict: Dict[str, Any],
    identity: tuple,
) -> Tuple[Dict[str, Any], HistoryUpdate]:
    """Build a conversation.item.create event and its history update function."""
    item_id = f"item_{uuid.uuid4().hex[:12]}"
    item_dict = {**item_dict, "id": item_id}
    event = {
        "type": "conversation.item.create",
        "item": item_dict,
    }

    def update(history: List[ConversationEntry], ack: Dict[str, Any]) -> List[ConversationEntry]:
        actual_id = (ack.get("item") or {}).get("id", item_id)
        return history + [(identity, actual_id)]

    return event, update
