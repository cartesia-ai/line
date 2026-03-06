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
from typing import Any, AsyncIterable, Dict, List, Optional

from loguru import logger
import websockets
from websockets.protocol import State as WsState

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, StreamChunk, _extract_instructions_and_messages
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.stream import (
    _compute_divergence,
    _context_identity,
    _expand_messages,
    _iterate_ws_query,
    _WsEventStream,
)
from line.llm_agent.tools.utils import FunctionTool

WS_URL = "wss://api.openai.com/v1/responses"


# ---------------------------------------------------------------------------
# WebSocketProvider
# ---------------------------------------------------------------------------


class _WebSocketProvider:
    """
    OpenAI WebSocket mode provider (Responses API).

    Exposes the same ``chat() -> stream`` interface as ``LlmProvider`` so it
    can be used as a drop-in replacement in ``LlmAgent``.

    Divergence handling
    -------------------
    In a voice-agent pipeline the SDK may truncate an assistant message when
    the user interrupts (barge-in).  The next ``chat()`` call will carry the
    *truncated* text in its history, while the server — via
    ``previous_response_id`` — still holds the *full* text.

    We maintain a single canonical ``_history`` — a list of
    ``(identity, response_id | None)`` tuples covering every message the
    server knows about.  Model-output entries carry the ``response_id`` that
    completed that turn; input entries have ``None``.

    On each ``chat()`` call we compare the desired identities with
    ``_history``.  If the common prefix is shorter than the history (i.e. a
    previous message was modified or removed) we scan backwards for the
    latest entry with a ``response_id`` and continue from there, preserving
    server-side context up to that point instead of resending the entire
    conversation from scratch.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        default_reasoning_effort: str = "none",
    ):
        self._model = model
        self._api_key = api_key or ""
        self._default_reasoning_effort = default_reasoning_effort
        self._ws: Optional[Any] = None  # websockets ClientConnection

        # Canonical conversation history.  Each entry is
        # (identity_tuple, response_id | None).  Position 0 is the
        # context identity (instructions + tools); subsequent entries
        # correspond to conversation messages.
        self._history: List[tuple] = []

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
            ping_interval=20,
            ping_timeout=20,
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

    async def _close_ws(self) -> None:
        if not self._is_closed():
            await self._ws.close()
        self._reset_response_state()

    def _reset_response_state(self) -> None:
        """Reset all continuation state — next chat() sends full context."""
        self._history = []

    def _finalize_response(
        self,
        response: Dict[str, Any],
        continuation_idx: int,
        desired_ids: List[tuple],
    ) -> None:
        """Update canonical history after a completed response.

        Truncates history to *continuation_idx*, appends the new input
        identities, and appends each model output identity tagged with
        its response_id.
        """
        response_id = response.get("id")
        model_ids = _extract_model_output_identities(response)

        history = self._history[:continuation_idx]
        for ident in desired_ids[continuation_idx:]:
            history.append((ident, None))
        for ident in model_ids:
            history.append((ident, response_id))
        self._history = history

    # --- Public interface ---

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        *,
        config: LlmConfig,
        **kwargs,
    ) -> AsyncIterable[StreamChunk]:
        """Start a streaming chat completion over WebSocket mode.

        Returns one async iterator per request/response. The provider owns the
        full lifecycle for that iterator, including one retry when the server
        has forgotten a ``previous_response_id`` before emitting any content.
        """

        async def _chat():
            attempt = 0
            while True:
                emitted_any = False
                try:
                    async for chunk in _iterate_ws_query(
                        self._setup_chat(
                            messages,
                            tools,
                            config,
                            web_search_options=kwargs.get("web_search_options"),
                        )
                    ):
                        emitted_any = True
                        yield chunk
                    return
                except RuntimeError as exc:
                    if "previous_response_not_found" not in str(exc) or emitted_any or attempt >= 1:
                        raise
                    attempt += 1
                    logger.debug(
                        "Responses API lost previous_response_id; retrying current turn from scratch"
                    )

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

            # -- Separate system messages (→ instructions) from conversation
            instructions, non_system = _extract_instructions_and_messages(messages, config)
            tool_defs = build_openai_tool_defs(
                tools,
                web_search_options=web_search_options,
                strict=False,
                responses_api=True,
            )

            # -- Context identity sits at position 0 of desired_ids.
            # If instructions or tools change, prefix matching fails at
            # position 0 and the entire chain is invalidated (correct
            # for prefix-cache keying).
            context_id = _context_identity(
                instructions,
                tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            desired_pairs = _expand_messages(non_system, assistant_text_type="output_text")
            desired_ids = [context_id] + [identity for _, identity in desired_pairs]
            if self._history and self._history[0][0] == context_id:
                current_identities = [identity for identity, _ in self._history[1:]]
                shared_item_count, _ = _compute_divergence(current_identities, desired_pairs)
                prefix_len = 1 + shared_item_count
            else:
                prefix_len = 0

            # Scan backwards within the shared prefix for the latest
            # history entry that carries a response_id (i.e. a completed
            # model turn or warmup we can continue from).
            continuation_idx = 0
            continuation_resp_id: Optional[str] = None
            for i in range(min(prefix_len, len(self._history)) - 1, -1, -1):
                if self._history[i][1] is not None:
                    continuation_idx = i + 1
                    continuation_resp_id = self._history[i][1]
                    break

            if prefix_len < len(self._history):
                logger.debug(
                    "WebSocket conversation diverged at index %d/%d, rolling back to checkpoint at %d",
                    prefix_len,
                    len(self._history),
                    continuation_idx,
                )

            # desired_ids[0] is the context identity; subsequent entries map
            # one-to-one to the expanded desired_pairs stream.
            input_pairs = desired_pairs[max(0, continuation_idx - 1) :]
            request = _build_request(
                model=self._model,
                default_reasoning_effort=self._default_reasoning_effort,
                instructions=instructions,
                tool_defs=tool_defs,
                cfg=config,
                input=[item for item, _ in input_pairs],
                previous_response_id=continuation_resp_id,
            )

            await self._ws.send(json.dumps(request))

        except BaseException:
            self._lock.release()
            raise

        # Lock ownership transfers to the stream — released in __aexit__
        def on_response_done(response: Dict[str, Any]) -> None:
            if response.get("error", {}).get("code") == "previous_response_not_found":
                self._reset_response_state()
            elif response.get("status") == "completed":
                self._finalize_response(response, continuation_idx, desired_ids)

        event_stream = _WsEventStream(self._ws, on_response_done)
        return event_stream, self._ws, self._lock

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[FunctionTool]] = None,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
    ):
        """Pre-populate the server's prefix cache with instructions and tools.

        Sends a ``generate: false`` request so the server caches the prompt
        prefix (instructions + tools) without producing any output.  The
        resulting response ID is stored in ``_history`` at position 0 (the
        context identity slot) so the first ``chat()`` call chains from it
        via normal prefix matching.
        """
        async with self._lock:
            await self._ensure_connected()
            instructions = config.system_prompt or None
            tool_defs = build_openai_tool_defs(
                tools,
                web_search_options=web_search_options,
                strict=False,
                responses_api=True,
            )
            context_id = _context_identity(
                instructions,
                tool_defs,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            # Already warmed up with this exact context
            if self._history and self._history[0][0] == context_id and self._history[0][1] is not None:
                return

            request = _build_request(
                model=self._model,
                default_reasoning_effort=self._default_reasoning_effort,
                instructions=instructions,
                tool_defs=tool_defs,
                cfg=config,
                generate=False,
            )
            await self._ws.send(json.dumps(request))

            # Drain via _WsEventStream — reuse existing event handling
            warmup_resp_id: Optional[str] = None

            def on_done(response: Dict[str, Any]) -> None:
                nonlocal warmup_resp_id
                if response.get("status") == "completed":
                    warmup_resp_id = response.get("id")

            stream = _WsEventStream(self._ws, on_done)
            async for _ in stream:
                pass

            if warmup_resp_id is not None:
                self._history = [(context_id, warmup_resp_id)]

    async def aclose(self) -> None:
        """Close the WebSocket connection.

        Acquires the lock so that an in-flight ``chat()`` stream finishes
        (or is drained) before the connection is torn down.
        """
        async with self._lock:
            await self._close_ws()


def _normalize_openai_model_name(model: str) -> str:
    """Strip LiteLLM-style ``openai/`` prefixes for direct OpenAI API calls."""
    if model.lower().startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _build_request(
    *,
    model: str,
    default_reasoning_effort: str,
    instructions: Optional[str],
    tool_defs: Optional[List[Dict[str, Any]]],
    cfg: LlmConfig,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a ``response.create`` request dict.

    Callers pass request-specific fields (``input``, ``previous_response_id``,
    ``generate``, etc.) via ``**extra``.
    """
    request: Dict[str, Any] = {
        "type": "response.create",
        "model": _normalize_openai_model_name(model),
        "store": True,
        **extra,
    }
    if instructions is not None:
        request["instructions"] = instructions
    if tool_defs is not None:
        request["tools"] = tool_defs
    reasoning_effort = cfg.reasoning_effort or default_reasoning_effort
    request["reasoning"] = {"effort": reasoning_effort}
    if cfg.temperature is not None:
        request["temperature"] = cfg.temperature
    if cfg.max_tokens is not None:
        request["max_output_tokens"] = cfg.max_tokens
    if cfg.top_p is not None:
        request["top_p"] = cfg.top_p
    return request


def _extract_model_output_identities(response: Dict[str, Any]) -> List[tuple]:
    """Derive ordered message identities from a Responses API output.

    Mirrors the SDK's stored history shape: assistant text and tool calls are
    tracked as separate messages/events, so mixed turns must preserve each
    output item in order for continuation prefix-matching to remain stable.
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
