"""
HTTPS-based provider for OpenAI's Responses API.

Opt-in alternative to the default LiteLLM HTTP path (``acompletion``)
for ``gpt-5.2`` / ``gpt-5.4-*`` models.  Select with
``LlmProvider(..., backend="http_responses")``.

Speaks the Responses API directly via ``litellm.aresponses`` instead of
going through the Chat-Completions → Responses bridge that
``acompletion`` activates for these models.  That bridge runs an
``OpenAiResponsesToChatCompletionStreamIterator`` translator which
silently flattens commentary + final-answer ``message`` items into one
``Delta(content=...)`` stream, producing duplicated TTS output for
``gpt-5.4+`` reasoning models when both phases carry similar text.

Mirrors the design of ``_WebSocketProvider`` (same identity-based
history, same ``previous_response_id`` continuation, same
``_plan_responses_chat`` planner) but over HTTPS rather than the WS
endpoint — useful when the ``wss://api.openai.com/v1/responses``
endpoint is less reliable than HTTPS in practice.

Phase handling
--------------
The Responses API can emit two ``message`` items per response: one
with ``phase: "commentary"`` (preamble before tool calls) and one with
``phase: "final_answer"`` (the completed answer).  For a voice/TTS
context, the commentary item's text is almost always undesirable to
speak — and when it is similar/identical to the final answer it causes
double-speak.  This provider tracks ``output_index → phase`` from
``response.output_item.added`` events and suppresses
``response.output_text.delta`` events for items marked
``"commentary"``.
"""

import asyncio
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from litellm import aresponses
from loguru import logger

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, ParsedModelId, StreamChunk, ToolCall
from line.llm_agent.provider_utils import (
    ConversationEntry,
    _AsyncIterableContext,
    _plan_responses_chat,
)
from line.llm_agent.tools.utils import FunctionTool

# Terminal event types in the Responses streaming protocol.
_TERMINAL_EVENTS = frozenset(
    {
        "response.completed",
        "response.failed",
        "response.incomplete",
    }
)


class _HttpResponseEventStream:
    """Reads Responses-API streaming events from ``litellm.aresponses`` and
    yields :class:`StreamChunk` objects.

    Phase filtering: when an ``output_item.added`` event arrives with
    ``item.type == "message"`` and ``item.phase == "commentary"``, all
    subsequent ``output_text.delta`` events with that ``output_index``
    are dropped.  Tool calls and final-answer text pass through.

    On terminal events the ``on_response_done`` callback is invoked
    with the response dict so the provider can update its history.
    """

    def __init__(
        self,
        iterator: AsyncIterator[Any],
        on_response_done: Callable[[Dict[str, Any]], None],
    ):
        self._iter = iterator
        self._on_response_done = on_response_done
        self.done = False

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        tool_calls: Dict[str, ToolCall] = {}
        commentary_indices: set[int] = set()
        received_content = False

        async for event in self._iter:
            event_type = _event_type(event)
            if not event_type:
                continue

            if event_type == "response.output_item.added":
                received_content = True
                item = event.item
                output_index = event.output_index
                item_type = item.type
                if item_type == "message":
                    phase = getattr(item, "phase", None)
                    if phase == "commentary":
                        commentary_indices.add(int(output_index))
                        logger.debug(
                            "Responses HTTP: suppressing commentary message at output_index=%d",
                            int(output_index),
                        )
                elif item_type == "function_call":
                    call_id = item.call_id
                    name = item.name
                    if call_id and call_id not in tool_calls:
                        tool_calls[call_id] = ToolCall(id=call_id, name=name, arguments="")

            elif event_type == "response.output_text.delta":
                output_index = event.output_index
                if int(output_index) in commentary_indices:
                    continue
                delta = event.delta
                if delta:
                    received_content = True
                    yield StreamChunk(text=delta)

            elif event_type == "response.function_call_arguments.delta":
                # The Responses streaming protocol identifies the active
                # function call by ``item_id`` here, not ``call_id``.  But
                # we keyed ``tool_calls`` by ``call_id`` from
                # ``output_item.added``.  Look up the call by output_index
                # via the ordered insertion of ``tool_calls``.  Simpler:
                # accumulate by item_id and resolve on output_item.done.
                item_id = event.item_id
                delta = event.delta
                # Find the most-recent tool call whose item_id matches, or
                # fall back to keying by item_id directly.
                tc = tool_calls.get(item_id)
                if tc is None:
                    tc = ToolCall(id=item_id, name="", arguments="")
                    tool_calls[item_id] = tc
                tc.arguments += delta

            elif event_type == "response.output_item.done":
                item = event.item
                if item.type == "function_call":
                    call_id = item.call_id
                    name = item.name
                    args = item.arguments
                    item_id = item.id
                    # The item may have been tracked under item_id in the
                    # delta handler; rekey to call_id now that we have it.
                    if call_id:
                        tc = tool_calls.pop(item_id, None)
                        if tc is None:
                            tc = tool_calls.get(call_id) or ToolCall(id=call_id, name=name, arguments=args)
                        tc.id = call_id
                        tc.name = name or tc.name
                        # Server-canonical arguments win over our accumulated.
                        tc.arguments = args or tc.arguments
                        tc.is_complete = True
                        tool_calls[call_id] = tc

            elif event_type in _TERMINAL_EVENTS:
                self.done = True
                response = event.response
                response_dict = _to_dict(response)
                status = response_dict.get("status")
                self._on_response_done(response_dict)

                error = response_dict.get("error")
                if event_type == "response.failed" or status == "failed":
                    raise RuntimeError(
                        f"OpenAI Responses API error "
                        f"({error.get('code', '') if isinstance(error, dict) else ''}): "
                        f"{error.get('message', 'unknown error') if isinstance(error, dict) else error}"
                    )

                if status == "completed":
                    for tc in tool_calls.values():
                        tc.is_complete = True
                else:
                    details = response_dict.get("incomplete_details")
                    logger.warning(
                        "Non-completed response from Responses API: status=%s, reason=%s, response_id=%s",
                        status,
                        details.get("reason", "") if isinstance(details, dict) else "",
                        response_dict.get("id", ""),
                    )

                yield StreamChunk(
                    tool_calls=list(tool_calls.values()) if tool_calls else [],
                    is_final=True,
                )
                return

            elif event_type == "error":
                self.done = True
                error = event.error
                err_dict = _to_dict(error) if not isinstance(error, dict) else error
                raise RuntimeError(
                    f"OpenAI Responses API error "
                    f"({err_dict.get('code', '')}): {err_dict.get('message', 'unknown error')}"
                )

            # All other event types (response.created, response.in_progress,
            # response.content_part.added/done, response.output_text.done,
            # reasoning_summary_*, etc.) are safely ignorable for our
            # streaming output.

        # Stream ended without a terminal event.
        if not received_content:
            raise RuntimeError("Responses API stream closed before delivering any response content")


def _event_type(event: Any) -> Optional[str]:
    """Coerce a Responses-API stream event's ``type`` to its wire string.

    LiteLLM models events as pydantic ``BaseLiteLLMOpenAIResponseObject``
    instances whose ``type`` field is a ``ResponsesAPIStreamEvents(str, Enum)``
    member.  ``str(enum_member)`` returns the *qualified name* form
    ``"ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED"``, not the wire-format
    value ``"response.output_item.added"`` — so naive string comparison
    silently fails to match.  Use ``.value`` (or the enum's str-mixin)
    to get the actual wire string.
    """
    t = getattr(event, "type", None)
    if t is None and isinstance(event, dict):
        t = event.get("type")
    if t is None:
        return None
    value = getattr(t, "value", None)
    if value is not None:
        return str(value)
    return str(t)


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort dict conversion for pydantic responses or raw dicts."""
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - defensive
            pass
    return dict(getattr(obj, "__dict__", {}) or {})


# ---------------------------------------------------------------------------
# _HttpResponsesProvider
# ---------------------------------------------------------------------------


class _HttpResponsesProvider:
    """OpenAI Responses-API provider over HTTPS via ``litellm.aresponses``.

    Same interface as the other providers (``chat``, ``warmup``, ``aclose``)
    and the same identity-based history bookkeeping as
    ``_WebSocketProvider`` — they share the planner in
    :func:`_plan_responses_chat`.  The transport is the only material
    difference.
    """

    def __init__(
        self,
        model_id: ParsedModelId,
        api_key: Optional[str] = None,
        default_reasoning_effort: Optional[str] = "low",
    ):
        self._model_id = model_id
        self._api_key = api_key or ""
        self._default_reasoning_effort = default_reasoning_effort
        self._history: List[ConversationEntry] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        self._lock = self._lock or asyncio.Lock()
        return self._lock

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        *,
        config: LlmConfig,
        **kwargs,
    ) -> _AsyncIterableContext:
        """Start a streaming Responses-API chat over HTTPS.

        Retries once when the server has forgotten a
        ``previous_response_id`` before any content was emitted.
        """
        web_search_options = kwargs.get("web_search_options")

        async def _iter():
            attempt = 0
            while True:
                emitted_any = False
                try:
                    lock = self._get_lock()
                    await lock.acquire()
                    try:
                        body, update = _plan_responses_chat(
                            history=self._history,
                            model_id=self._model_id,
                            default_reasoning_effort=self._default_reasoning_effort,
                            messages=messages,
                            tools=tools,
                            config=config,
                            web_search_options=web_search_options,
                        )

                        request_kwargs: Dict[str, Any] = dict(body)
                        request_kwargs["model"] = str(self._model_id)
                        request_kwargs["stream"] = True
                        if self._api_key:
                            request_kwargs["api_key"] = self._api_key
                        if config.timeout:
                            request_kwargs["timeout"] = config.timeout

                        iterator = await aresponses(**request_kwargs)

                        def on_response_done(response_dict: Dict[str, Any], _update=update) -> None:
                            if response_dict.get("status") != "completed":
                                return
                            self._history = _update(self._history, response_dict)

                        stream = _HttpResponseEventStream(iterator, on_response_done)

                        async for chunk in stream:
                            emitted_any = True
                            yield chunk
                    finally:
                        lock.release()
                    return
                except RuntimeError as exc:
                    if "previous_response_not_found" not in str(exc) or emitted_any or attempt >= 1:
                        raise
                    self._history = []
                    attempt += 1
                    logger.debug(
                        "Responses API lost previous_response_id; retrying current turn from scratch"
                    )
                except Exception as exc:
                    # litellm surfaces previous_response_not_found as
                    # ``BadRequestError`` / generic Exception with the
                    # code in the message.  Match by substring to mirror
                    # the WS provider's behavior.
                    if "previous_response_not_found" not in str(exc) or emitted_any or attempt >= 1:
                        raise
                    self._history = []
                    attempt += 1
                    logger.debug(
                        "Responses API lost previous_response_id (via %s); retrying current turn",
                        type(exc).__name__,
                    )

        return _AsyncIterableContext(_iter)

    async def warmup(
        self,
        config: LlmConfig,
        tools: Optional[List[FunctionTool]] = None,
        *,
        web_search_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op for the HTTPS provider — no persistent connection to warm."""
        return None

    async def aclose(self) -> None:
        """Reset history. No persistent connection to close."""
        async with self._get_lock():
            self._history = []
