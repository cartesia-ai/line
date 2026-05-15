"""
OpenAI WebSocket Mode Provider for the Responses API.

Uses a persistent WebSocket connection to wss://api.openai.com/v1/responses
for lower-latency agentic workflows.  Instead of separate HTTP requests per
turn, we keep the socket open and use ``previous_response_id`` for efficient
multi-turn continuations.

Key differences from the Realtime API (``realtime_provider.py``):
  * Works with standard text models (openai/gpt-5.2, openai/gpt-5.2-pro, …)
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
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from websockets.legacy.client import WebSocketClientProtocol

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, ParsedModelId
from line.llm_agent.provider_utils import (
    ConversationEntry,
    HistoryUpdate,
    _AsyncIterableContext,
    _build_responses_body,
    _cancel_and_drain,
    _context_identity,
    _plan_responses_chat,
    _ws_connect,
    _ws_is_closed,
    _WsEventStream,
)
from line.llm_agent.schema_converter import build_openai_tool_defs
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
        model_id: ParsedModelId,
        api_key: Optional[str] = None,
        default_reasoning_effort: Optional[str] = "none",
    ):
        self._model_id = model_id
        self._default_reasoning_effort = default_reasoning_effort
        self._api_key = api_key or ""
        self._ws: Optional[WebSocketClientProtocol] = None
        self._history: List[ConversationEntry] = []
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
        """Start a streaming chat completion over WebSocket mode.

        Retries once when the server has forgotten a ``previous_response_id``
        before any content was emitted.
        """
        ws_kwargs = {"web_search_options": kwargs.get("web_search_options")}

        async def _iter():
            attempt = 0
            while True:
                emitted_any = False
                try:
                    stream = await self._setup_chat(messages, tools, config, **ws_kwargs)
                    try:
                        async for chunk in stream:
                            emitted_any = True
                            yield chunk
                    finally:
                        try:
                            if not stream.done:
                                await _cancel_and_drain(stream, self._ws)
                        finally:
                            self._get_lock().release()
                    return
                except RuntimeError as exc:
                    if "previous_response_not_found" not in str(exc) or emitted_any or attempt >= 1:
                        raise
                    self._history = []
                    attempt += 1
                    logger.debug(
                        "Responses API lost previous_response_id; retrying current turn from scratch"
                    )

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
                model_id=self._model_id,
                default_reasoning_effort=self._default_reasoning_effort,
                instructions=instructions,
                tool_defs=tool_defs,
                cfg=config,
                generate=False,
                input=[],  # No input for warmup
            )
            await self._ws.send(json.dumps(request))

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
        async with self._get_lock():
            if not _ws_is_closed(self._ws):
                await self._ws.close()
            self._history = []

    # --- Helpers ---

    async def _ensure_connected(self) -> None:
        if _ws_is_closed(self._ws):
            self._ws = await _ws_connect(WS_URL, self._api_key)
            self._history = []
            logger.debug("WebSocket mode connected to Responses API")

    async def _setup_chat(self, messages, tools, config, *, web_search_options=None):
        lock = self._get_lock()
        await lock.acquire()
        try:
            await self._ensure_connected()

            request, update_history = _plan_chat(
                history=self._history,
                model_id=self._model_id,
                default_reasoning_effort=self._default_reasoning_effort,
                messages=messages,
                tools=tools,
                config=config,
                web_search_options=web_search_options,
            )
            await self._ws.send(json.dumps(request))

        except BaseException:
            lock.release()
            raise

        def on_response_done(response):
            if response.get("status") != "completed":
                return
            self._history = update_history(self._history, response)

        return _WsEventStream(self._ws, on_response_done)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _plan_chat(
    *,
    history: List[ConversationEntry],
    model_id: ParsedModelId,
    default_reasoning_effort: Optional[str],
    messages: List[Message],
    tools: Optional[List[FunctionTool]],
    config: LlmConfig,
    web_search_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], HistoryUpdate]:
    """Compute the WS request to send and the history-update callback.

    Wraps the shared :func:`_plan_responses_chat` planner with the
    WS-protocol ``"type": "response.create"`` envelope.
    """
    body, update = _plan_responses_chat(
        history=history,
        model_id=model_id,
        default_reasoning_effort=default_reasoning_effort,
        messages=messages,
        tools=tools,
        config=config,
        web_search_options=web_search_options,
    )
    request = {"type": "response.create", **body}
    return request, update


def _build_request(
    *,
    model_id: ParsedModelId,
    default_reasoning_effort: Optional[str],
    instructions: Optional[str],
    tool_defs: Optional[List[Dict[str, Any]]],
    cfg: LlmConfig,
    input: Optional[List[Dict[str, Any]]] = None,
    previous_response_id: Optional[str] = None,
    generate: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a ``response.create`` WebSocket request dict.

    Wraps the shared :func:`_build_responses_body` with the WS-protocol
    ``type`` envelope and the WS-only ``generate`` flag.
    """
    body = _build_responses_body(
        model_id=model_id,
        default_reasoning_effort=default_reasoning_effort,
        instructions=instructions,
        tool_defs=tool_defs,
        cfg=cfg,
        input=input,
        previous_response_id=previous_response_id,
    )
    request: Dict[str, Any] = {"type": "response.create", **body}
    if generate is not None:
        request["generate"] = generate
    return request
