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
from line.llm_agent.provider import Message, _extract_instructions_and_messages
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.stream import (
    ConversationEntry,
    HistoryUpdate,
    _AsyncIterableContext,
    _cancel_and_drain,
    _compute_divergence,
    _context_identity,
    _expand_messages,
    _normalize_openai_model_name,
    _ws_connect,
    _ws_is_closed,
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
        default_reasoning_effort: Optional[str] = "none",
    ):
        self._model = model
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
                model=self._model,
                default_reasoning_effort=self._default_reasoning_effort,
                instructions=instructions,
                tool_defs=tool_defs,
                cfg=config,
                generate=False,
                input=[] # No input for warmup
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
                model=self._model,
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
            error_code = (response.get("error") or {}).get("code")
            if error_code == "previous_response_not_found":
                self._history = []
                return
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
    model: str,
    default_reasoning_effort: Optional[str],
    messages: List[Message],
    tools: Optional[List[FunctionTool]],
    config: LlmConfig,
    web_search_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], HistoryUpdate]:
    """Compute the request to send and the history update callback.

    Returns ``(request_dict, update_fn)`` where
    ``update_fn(history, response) -> new_history``.
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
            "WebSocket conversation diverged at index %d/%d, rolling back to checkpoint at %d",
            prefix_len,
            len(history),
            continuation_idx,
        )

    input_pairs = desired_pairs[max(0, continuation_idx - 1) :]
    request = _build_request(
        model=model,
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

    return request, update


def _build_request(
    *,
    model: str,
    default_reasoning_effort: Optional[str],
    instructions: Optional[str],
    tool_defs: Optional[List[Dict[str, Any]]],
    cfg: LlmConfig,
    input: Optional[List[Dict[str, Any]]] = None,
    previous_response_id: Optional[str] = None,
    generate: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a ``response.create`` request dict."""
    request: Dict[str, Any] = {
        "type": "response.create",
        "model": _normalize_openai_model_name(model),
        "store": True,
    }
    if input is not None:
        request["input"] = input
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    if generate is not None:
        request["generate"] = generate
    if instructions is not None:
        request["instructions"] = instructions
    if tool_defs is not None:
        request["tools"] = tool_defs
    reasoning_effort = cfg.reasoning_effort or default_reasoning_effort
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}
    if cfg.temperature is not None:
        request["temperature"] = cfg.temperature
    if cfg.max_tokens is not None:
        request["max_output_tokens"] = cfg.max_tokens
    if cfg.top_p is not None:
        request["top_p"] = cfg.top_p
    return request


def _extract_model_output_identities(response: Dict[str, Any]) -> List[tuple]:
    """Derive ordered message identities from a Responses API output."""
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
