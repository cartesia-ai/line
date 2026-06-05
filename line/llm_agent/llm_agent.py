"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

import asyncio
from contextlib import suppress
import inspect
import json
import time
import traceback
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    get_args,
)

from loguru import logger

from line.agent import AgentCallable, TurnEnv
from line.events import (
    AgentEndCall,
    AgentHandedOff,
    AgentSendDtmf,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTransferCall,
    CallEnded,
    CallStarted,
    CustomHistoryEntry,
    HistoryEvent,
    InputEvent,
    LogMetric,
    OutputEvent,
    UserTextSent,
    UserTurnEnded,
)
from line.llm_agent.background_queue import BackgroundQueue
from line.llm_agent.config import LlmConfig, _merge_configs, _normalize_config
from line.llm_agent.history import _HISTORY_EVENT_TYPES, History
from line.llm_agent.provider import (
    LlmProvider,
    Message,
    ToolCall,
    _get_model_config,
    parse_model_id,
)
from line.llm_agent.tools.system import EndCallTool, TransferCallTool, VoicemailTool, WebSearchTool
from line.llm_agent.tools.utils import (
    FunctionTool,
    ToolEnv,
    ToolType,
    _merge_tools,
    _normalize_tools,
)
from line.llm_agent.voicemail_detection import VoicemailDetectionConfig, _VoicemailDetector

T = TypeVar("T")

# Concrete OutputEvent types for isinstance checks (OutputEvent itself is a Union).
_OUTPUT_EVENT_TYPES: Tuple[type, ...] = get_args(OutputEvent)

# Output events the user actually sees/hears. Only these are buffered behind the
# voicemail-detection gate; metrics/logs always pass through immediately.
_USER_VISIBLE_OUTPUT_TYPES: Tuple[type, ...] = (
    AgentSendText,
    AgentEndCall,
    AgentTransferCall,
    AgentSendDtmf,
)

# Type alias for tools that can be passed to LlmAgent.
# Plain callables are automatically wrapped via the @tool decorator.
ToolSpec = Union[FunctionTool, WebSearchTool, EndCallTool, TransferCallTool, VoicemailTool, Callable]


def _is_voicemail_tool(tool: Any) -> bool:
    """Return True if *tool* is the built-in voicemail tool (instance or FunctionTool)."""
    if isinstance(tool, VoicemailTool):
        return True
    return getattr(tool, "name", None) == "voicemail"


class LlmAgent:
    """
    Agent wrapping LLM providers via LiteLLM with tool calling support.

    Supports loopback, passthrough, and handoff tool paradigms.
    Also supports web search via native LLM capabilities or fallback to DuckDuckGo.

    See README.md for examples.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        tools: Optional[List[ToolSpec]] = None,
        config: Optional[LlmConfig] = None,
        max_tool_iterations: int = 10,
        backend: Optional[str] = None,
        voicemail_detection: Optional[VoicemailDetectionConfig] = None,
        voicemail_tool_active_turns: Optional[int] = 1,
    ):
        """
        Args:
            voicemail_detection: Opt-in cheap-LM voicemail detection sidecar
                (Approach 2). When set, a separate classifier runs concurrently
                with the main LM on each completed user turn; a ``voicemail``
                verdict within the gate suppresses the main output and ends the
                call with ``reason="voicemail_detected"``.
            voicemail_tool_active_turns: For the built-in ``voicemail`` tool
                (Approach 1) — how many completed user turns the tool stays
                available before the agent drops it from its options (the
                conversation is "deemed started"). Defaults to ``1`` (available
                only for the first user turn). ``None`` keeps the tool for the
                whole call. No-op when no voicemail tool is present.
        """
        if not api_key:
            raise ValueError("Missing API key in LLmAgent initialization")

        model_id = parse_model_id(model)
        model_config = _get_model_config(model_id, backend=backend)

        # Resolve the base config to insert default values for any _UNSET sentinels.
        effective_config = _normalize_config(config or LlmConfig())
        if (
            effective_config.reasoning_effort is not None
            and effective_config.reasoning_effort != "none"
            and not model_config.supports_reasoning_effort
        ):
            raise ValueError(
                f"Model {str(model_id)} does not support reasoning_effort. "
                "Remove reasoning_effort from your LlmConfig or use a model that supports it."
            )

        self._model_id = model_id
        self._api_key = api_key
        self._config = effective_config
        self._max_tool_iterations = max_tool_iterations

        self._tools: List[ToolSpec] = list(tools or [])
        effective_tools, web_search_options = _normalize_tools(self._tools, model_id=model_id)

        self._llm = LlmProvider(
            model=str(model_id),
            api_key=api_key,
            config=effective_config,
            tools=self._tools,
            backend=backend,
        )

        # Approach 1: drop the built-in voicemail tool once the conversation is
        # "deemed started" (after `voicemail_tool_active_turns` completed turns).
        self._voicemail_tool_active_turns = voicemail_tool_active_turns
        self._user_turns_seen = 0
        self._voicemail_tool_removed = False

        # Approach 2: opt-in cheap-LM voicemail detection sidecar.
        self._voicemail_detection = voicemail_detection
        self._voicemail_detector = (
            _VoicemailDetector(voicemail_detection) if voicemail_detection is not None else None
        )

        self._introduction_sent = False
        self.history = History()
        self._handoff_target: Optional[AgentCallable] = None  # Normalized process function
        # Queue for events from backgrounded tools that need to trigger loopback
        self._background_event_queue: BackgroundQueue[Tuple[AgentToolCalled, AgentToolReturned]] = (
            BackgroundQueue()
        )
        # Cache for thought signatures (Gemini 3+ models)
        # Maps tool_call_id -> thought_signature
        self._tool_signatures: Dict[str, str] = {}

        tool_names = [t.name for t in effective_tools] + (["web_search"] if web_search_options else [])
        logger.info(f"LlmAgent initialized with model={self._model_id}, tools={tool_names}")

    def _get_background_event_queue(
        self,
    ) -> "BackgroundQueue[Tuple[AgentToolCalled, AgentToolReturned]]":
        return self._background_event_queue

    def set_tools(self, tools: List[ToolSpec]) -> None:
        """Replace the agent's tools with a new list."""
        self._tools = tools
        self._llm._set_tools(tools)

    def set_config(self, config: LlmConfig) -> None:
        """Replace the agent's config."""
        self._config = _normalize_config(config)

    async def process(
        self,
        env: TurnEnv,
        event: InputEvent,
        *,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[ToolSpec]] = None,
        context: Union[str, List[HistoryEvent], None] = None,
        history: Optional[List[HistoryEvent]] = None,
    ) -> AsyncIterable[OutputEvent]:
        """Process an input event and yield output events.

        Args:
            env: The turn environment.
            event: The input event to process.
            config: Optional LlmConfig to merge with self._config for this #process invocation
            tools: Optional tools to use for this #process invocation. Tools with matching names replace
                those in self._tools; other tools from self._tools are preserved.
            context: Extra context for this #process invocation only. If a string, converted to a
                system CustomHistoryEntry. If a list of HistoryEvents, used as-is.
                Appended to the end of history for message building. Not persisted.
            history: Override the managed history for this #process invocation only. When provided,
                _build_messages uses this list instead of self.history. The managed
                self.history still receives _set_input and _append_local as usual.

        Raises:
            TypeError: If config, tools, context, or history have invalid types.
        """
        async for output in self._process_impl(
            env, event, config=config, tools=tools, context=context, history=history
        ):
            yield _set_responding_to(output, event.event_id)

    async def _process_impl(
        self,
        env: TurnEnv,
        event: InputEvent,
        *,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[ToolSpec]] = None,
        context: Union[str, List[HistoryEvent], None] = None,
        history: Optional[List[HistoryEvent]] = None,
    ) -> AsyncIterable[OutputEvent]:
        """Internal implementation of process(). All yielded events are stamped
        with responding_to by the process() wrapper."""
        turn_start_time = time.perf_counter()

        self._validate_config(config)
        self._validate_tools(tools)
        self._validate_context(context)
        self._validate_history(history)

        # Track the event_id of the triggering input event
        # The triggering event is the last element in event.history
        current_event_id = event.history[-1].event_id if event.history else ""
        self.history._set_input(event.history or [], current_event_id)

        # Compute effective config and tools for this #process invocation
        effective_config = _merge_configs(self._config, config) if config else self._config
        effective_tools = _merge_tools(self._tools, tools)

        # If handoff is active, call the handed-off process function
        if self._handoff_target is not None:
            async for output in self._handoff_target(env, event):
                self.history._append_local(output)
                yield output
            # Keep turn timing consistent across all process paths, including handoffs.
            yield LogMetric(name="agent_turn_ms", value=(time.perf_counter() - turn_start_time) * 1000)
            return

        # Handle CallStarted
        if isinstance(event, CallStarted):
            warmup_task = asyncio.create_task(
                self._llm.warmup(config=effective_config, tools=effective_tools)
            )
            if effective_config.introduction and not self._introduction_sent:
                output = AgentSendText(text=effective_config.introduction)
                self.history._append_local(output)
                self._introduction_sent = True
                yield output
            yield LogMetric(name="agent_turn_ms", value=(time.perf_counter() - turn_start_time) * 1000)
            try:
                await warmup_task
            except asyncio.CancelledError:
                raise  # Warmup task continues as a separate asyncio.Task
            except Exception as e:
                logger.warning(f"Provider warmup failed: {e}")
            return

        # Handle CallEnded
        if isinstance(event, CallEnded):
            await self.cleanup()
            yield LogMetric(name="agent_turn_ms", value=(time.perf_counter() - turn_start_time) * 1000)
            return

        # A non-handoff, non-lifecycle event drives an agent response turn.
        # Count it and, per Approach 1, drop the voicemail tool once the
        # conversation is "deemed started".
        self._user_turns_seen += 1
        self._maybe_remove_voicemail_tool()
        # Recompute in case the voicemail tool was just removed.
        effective_tools = _merge_tools(self._tools, tools)

        gen = self._generate_response(
            env, event, effective_tools, effective_config, context=context, history=history
        )
        # Approach 2: gate the main response behind the voicemail detector for
        # completed user turns that carry a non-empty transcript.
        if self._voicemail_detector is not None and isinstance(event, UserTurnEnded):
            transcript = _extract_user_transcript(event)
            if transcript:
                gen = self._wrap_with_voicemail_detection(gen, transcript)

        async for output in gen:
            yield output

        yield LogMetric(name="agent_turn_ms", value=(time.perf_counter() - turn_start_time) * 1000)

    async def _generate_response(
        self,
        env: TurnEnv,
        event: InputEvent,
        tool_specs: List[ToolSpec],
        config: LlmConfig,
        *,
        context: Union[str, List[HistoryEvent], None] = None,
        history: Optional[List[HistoryEvent]] = None,
    ) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM.

        Args:
            env: The turn environment.
            event: The input event to process.
            tool_specs: ToolSpecs to use for the current #process invocation
            config: The effective LlmConfig for the current #process invocation
            context: Extra context to append to history for the current #process invocation only.
            history: Override history for the current #process invocation only.
        """
        tools, web_search_options = _normalize_tools(tool_specs, model_id=self._model_id)
        tool_map: Dict[str, FunctionTool] = {t.name: t for t in tools}

        is_first_iteration = True
        should_loopback = False

        # Timing metrics - measured from start of _generate_response, emitted once
        response_start_time = time.perf_counter()
        first_chunk_logged = False
        first_text_logged = False

        for _iteration in range(self._max_tool_iterations):
            # ==== LOOPBACK MANAGMENT ==== #
            # First, yield any pending events from backgrounded tools
            # These events were produced since the last iteration (or from previous process() invocations)
            if is_first_iteration or should_loopback:
                # Drain any immediately available events (non-blocking)
                while (pair := self._get_background_event_queue().get_nowait()) is not None:
                    called_evt, returned_evt = pair
                    yield called_evt
                    yield returned_evt
            else:
                # Otherwise wait for either: all sources complete OR new event arrives
                result = await self._get_background_event_queue().get()
                if result is None:
                    # All background sources completed with no more events
                    # this generation process is completed - exit loop
                    break
                called_evt, returned_evt = result
                yield called_evt
                yield returned_evt

            is_first_iteration = False
            should_loopback = False
            # ==== END LOOPBACK MANAGMENT ==== #

            # ==== GENERATION CALL ==== #
            messages = await self._build_messages(context=context, history=history)

            tool_calls_dict: Dict[str, ToolCall] = {}

            # Build kwargs for LLM chat, including web_search_options if available
            chat_kwargs: Dict[str, Any] = {}
            if web_search_options:
                chat_kwargs["web_search_options"] = web_search_options

            stream = self._llm.chat(
                messages,
                tools or None,
                config=config,
                **chat_kwargs,
            )
            async for chunk in stream:
                # Track time to first chunk (text or tool call)
                if not first_chunk_logged and (chunk.text or chunk.tool_calls):
                    first_chunk_ms = (time.perf_counter() - response_start_time) * 1000
                    logger.info(f"Time to first chunk: {first_chunk_ms:.2f}ms")
                    yield LogMetric(name="llm_first_chunk_ms", value=first_chunk_ms)
                    first_chunk_logged = True

                if chunk.text:
                    output = AgentSendText(text=chunk.text)
                    self.history._append_local(output)

                    # Track time to first text
                    if not first_text_logged:
                        first_text_ms = (time.perf_counter() - response_start_time) * 1000
                        logger.info(f"Time to first text: {first_text_ms:.2f}ms")
                        yield LogMetric(name="llm_first_text_ms", value=first_text_ms)
                        first_text_logged = True

                    yield output

                if chunk.tool_calls:
                    # Tool call streaming differs by provider:
                    # - OpenAI: sends args incrementally ("{\"ci", "ty\":", "\"Tokyo\"}")
                    # - Anthropic: incremental chunks like OpenAI
                    # - Gemini: sends complete args each chunk ("{\"city\":\"Tokyo\"}")
                    # Provider handles accumulation; we just replace with latest version.
                    for tc in chunk.tool_calls:
                        tool_calls_dict[tc.id] = tc
            # ==== END GENERATION CALL ==== #

            # ==== TOOL CALLS ==== #
            # Store thought signatures for Gemini 3+ models before processing
            for tc in tool_calls_dict.values():
                if tc.thought_signature:
                    self._tool_signatures[tc.id] = tc.thought_signature

            ctx = ToolEnv(turn_env=env)

            # Track before tool calls are processed so backgrounded tools can reference the triggering event
            triggering_event_id = event.event_id

            for tc in tool_calls_dict.values():
                if not tc.is_complete:
                    continue

                tool = tool_map.get(tc.name)
                if not tool:
                    logger.warning(f"Unknown tool: {tc.name}")
                    continue

                tool_args = json.loads(tc.arguments) if tc.arguments else {}

                normalized_func = _normalize_to_async_gen(tool.func)

                # For backgrounded tools, we emit AgentToolCalled/AgentToolReturned pairs
                # inside _execute_backgroundable_tool, not here
                if tool.tool_type == ToolType.GENERAL and tool.is_background:
                    # Backgroundable tool: run in a shielded task that survives cancellation
                    # Each yielded value triggers a loopback with AgentToolCalled/AgentToolReturned pair
                    self._execute_backgroundable_tool(
                        normalized_func, ctx, tool_args, tc.id, tc.name, triggering_event_id
                    )
                    continue

                if tool.tool_type == ToolType.GENERAL:
                    # Branch per yielded value:
                    #   - OutputEvent → emit directly to the user (passthrough)
                    #   - raw value  → wrap as a synthetic tool result and trigger loopback
                    n = 0
                    yielded_raw = False
                    closed = False
                    try:
                        try:
                            async for value in normalized_func(ctx, **tool_args):
                                if isinstance(value, _OUTPUT_EVENT_TYPES):
                                    self.history._append_local(value)
                                    yield value
                                else:
                                    yielded_raw = True
                                    should_loopback = True
                                    call_id = f"{tc.id}-{n}"
                                    tool_called_output, tool_returned_output = _construct_tool_events(
                                        call_id, tc.name, tool_args, value
                                    )
                                    self.history._append_local(tool_called_output)
                                    self.history._append_local(tool_returned_output)
                                    yield tool_called_output
                                    yield tool_returned_output
                                    n += 1
                        except Exception as e:
                            # Use negative limit to show last 10 frames (most relevant)
                            logger.error(
                                f'Error in Tool Call to "{tc.name}":\n{traceback.format_exc(limit=-10)}'
                            )
                            tool_called_output, tool_returned_output = _construct_tool_events(
                                f"{tc.id}-{n}", tc.name, tool_args, f"error: {e}"
                            )
                            self.history._append_local(tool_called_output)
                            self.history._append_local(tool_returned_output)
                            yield tool_called_output
                            yield tool_returned_output
                            yielded_raw = True
                            should_loopback = True

                        # If the tool yielded only OutputEvents (or nothing at all), close out the
                        # LLM-issued tool_call_id with a success pair so the LLM has a tool result anchor.
                        if not yielded_raw:
                            tool_called_output, tool_returned_output = _construct_tool_events(
                                tc.id, tc.name, tool_args, "success"
                            )
                            self.history._append_local(tool_called_output)
                            self.history._append_local(tool_returned_output)
                            yield tool_called_output
                            yield tool_returned_output
                        closed = True
                    finally:
                        if not closed and not yielded_raw:
                            # CancelledError before any tool result was produced — record a
                            # cancelled pair in history so the LLM doesn't see an orphan call.
                            self.history._append_local(
                                AgentToolCalled(
                                    tool_call_id=tc.id,
                                    tool_name=tc.name,
                                    tool_args=tool_args,
                                )
                            )
                            self.history._append_local(
                                AgentToolReturned(
                                    tool_call_id=tc.id,
                                    tool_name=tc.name,
                                    tool_args=tool_args,
                                    result="cancelled",
                                )
                            )

                elif tool.tool_type == ToolType.HANDOFF:
                    # Emit AgentToolCalled before executing
                    tool_called_output = AgentToolCalled(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                    )
                    self.history._append_local(tool_called_output)
                    yield tool_called_output

                    # AgentHandedOff input event is passed to the handoff target to execute the tool
                    handed_off_event = AgentHandedOff()
                    event = AgentHandedOff(
                        history=event.history + [handed_off_event],
                        **{k: v for k, v in handed_off_event.model_dump().items() if k != "history"},
                    )
                    self.history._append_local(event)
                    tool_returned = False
                    try:
                        async for item in normalized_func(ctx, **tool_args, event=event):
                            self.history._append_local(item)
                            yield item
                        # Emit AgentToolReturned after successful completion
                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )
                        self.history._append_local(tool_returned_output)
                        tool_returned = True
                        yield tool_returned_output

                        # Format the handoff target to be called on all future events
                        # Use default args to bind loop variables
                        def handoff_target(
                            env: TurnEnv,
                            event: InputEvent,
                            _tool_args=tool_args,
                            _normalized_func=normalized_func,
                        ) -> AsyncIterable[OutputEvent]:
                            tool_env = ToolEnv(turn_env=env)
                            return _normalized_func(tool_env, **_tool_args.copy(), event=event)

                        self._handoff_target = handoff_target
                    except Exception as e:
                        # Use negative limit to show last 10 frames (most relevant)
                        logger.error(f'Error in Tool Call to "{tc.name}":\n{traceback.format_exc(limit=-10)}')
                        # Emit AgentToolReturned with error
                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result=f"error: {e}",
                        )
                        self.history._append_local(tool_returned_output)
                        tool_returned = True
                        yield tool_returned_output
                    finally:
                        if not tool_returned:
                            # CancelledError (BaseException) - ensure history stays consistent
                            self.history._append_local(
                                AgentToolReturned(
                                    tool_call_id=tc.id,
                                    tool_name=tc.name,
                                    tool_args=tool_args,
                                    result="cancelled",
                                )
                            )

            # ==== END TOOL CALLS ==== #

            if not (should_loopback or self._get_background_event_queue().is_active):
                break

    async def _build_messages(
        self,
        *,
        context: Union[str, List[HistoryEvent], None] = None,
        history: Optional[List[HistoryEvent]] = None,
    ) -> List[Message]:
        """Build LLM messages from conversation history.

        Uses self.history to get the merged history, then converts to LLM messages.

        The full_history contains HistoryEvent items:
        - InputEvent for events from input_history (matchable OutputEvents converted to InputEvent
          counterparts)
        - AgentToolCalled/AgentToolReturned for tool interactions from local_history
        - CustomHistoryEntry for injected history entries from local_history

        Args:
            context: Extra context to append to history for this call only.
                If a string, converted to a system CustomHistoryEntry.
                If a list of HistoryEvents, appended as-is.
            history: Override the managed history for this call only.
        """
        if history is not None:
            full_history = list(history)
        else:
            full_history = list(self.history)

        if context is not None:
            if isinstance(context, str):
                full_history.append(CustomHistoryEntry(content=context))
            else:
                full_history.extend(context)

        # First pass: collect all tool_call_ids that have matching AgentToolReturned
        returned_tool_call_ids: Set[str] = set()
        for event in full_history:
            if isinstance(event, AgentToolReturned):
                returned_tool_call_ids.add(event.tool_call_id)

        messages = []
        for event in full_history:
            # Handle InputEvent types
            if isinstance(event, UserTextSent):
                messages.append(Message(role="user", content=event.content or ""))
            elif isinstance(event, AgentTextSent):
                messages.append(Message(role="assistant", content=event.content or ""))
            # Handle CustomHistoryEntry (injected history entries)
            elif isinstance(event, CustomHistoryEntry):
                # Don't filter - could create invalid message sequences
                messages.append(Message(role=event.role, content=event.content))
            # Handle tool events from local_history
            elif isinstance(event, AgentToolCalled):
                # Look up thought_signature from cache (for Gemini 3+ models)
                # The tool_call_id may have a suffix like "-0", "-1" for streaming tools
                # Try exact match first, then try base ID without suffix
                thought_sig = self._tool_signatures.get(event.tool_call_id)
                if not thought_sig and "-" in event.tool_call_id:
                    base_id = event.tool_call_id.rsplit("-", 1)[0]
                    thought_sig = self._tool_signatures.get(base_id)

                messages.append(
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=event.tool_call_id,
                                name=event.tool_name,
                                arguments=json.dumps(event.tool_args),
                                thought_signature=thought_sig,
                            )
                        ],
                    )
                )

                # If this tool call doesn't have a matching result, add a pending result
                if event.tool_call_id not in returned_tool_call_ids:
                    messages.append(
                        Message(
                            role="tool",
                            content="pending",
                            tool_call_id=event.tool_call_id,
                            name=event.tool_name,
                        )
                    )
            elif isinstance(event, AgentToolReturned):
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(event.result)
                        if not isinstance(event.result, str)
                        else event.result,
                        tool_call_id=event.tool_call_id,
                        name=event.tool_name,
                    )
                )
        return messages

    def _execute_backgroundable_tool(
        self,
        normalized_func: Callable[..., AsyncIterable[Any]],
        ctx: ToolEnv,
        tool_args: Dict[str, Any],
        tc_id: str,
        tc_name: str,
        triggering_event_id: str,
    ) -> None:
        """Execute a backgroundable tool via the background queue.

        Each value yielded by the tool produces a pair of:
        - AgentToolCalled with tool_call_id = "{tc_id}-{n}"
        - AgentToolReturned with the same tool_call_id

        The source is subscribed to the background queue, which shields it
        from cancellation. Events are tagged with the CURRENT event_id at
        yield time so background tool results appear at the end of history
        when yielded after a new process() call has started.

        responding_to is set to the triggering_event_id (captured at subscription
        time) so background results reference the event that originally triggered
        the tool, not whatever event happens to be processing when results are drained.
        """

        async def generate_events() -> AsyncIterable[Tuple[AgentToolCalled, AgentToolReturned]]:
            n = 0
            try:
                async for value in normalized_func(ctx, **tool_args):
                    call_id = f"{tc_id}-{n}"
                    called, returned = _construct_tool_events(call_id, tc_name, tool_args, value)
                    called.responding_to = triggering_event_id
                    returned.responding_to = triggering_event_id
                    self.history._append_local(called)
                    self.history._append_local(returned)
                    yield (called, returned)
                    n += 1
            except Exception as e:
                logger.error(f"Error in Tool Call {tc_name}: {e}\n{traceback.format_exc(limit=-10)}")
                called, returned = _construct_tool_events(f"{tc_id}-{n}", tc_name, tool_args, f"error: {e}")
                called.responding_to = triggering_event_id
                returned.responding_to = triggering_event_id
                self.history._append_local(called)
                self.history._append_local(returned)
                yield (called, returned)

        self._get_background_event_queue().subscribe(generate_events())

    # ------------------------------------------------------------------
    # Approach 1: voicemail tool removal once the conversation has started
    # ------------------------------------------------------------------

    def _maybe_remove_voicemail_tool(self) -> None:
        """Drop the built-in voicemail tool once enough user turns have elapsed.

        The voicemail tool is only useful at the very start of a call (the
        greeting). Once the conversation is "deemed started" — after
        ``voicemail_tool_active_turns`` completed user turns — we remove it so the
        main LM can't accidentally hang up mid-conversation. No-op when no
        voicemail tool is configured or when removal is disabled (``None``).
        """
        if (
            self._voicemail_tool_removed
            or self._voicemail_tool_active_turns is None
            or self._user_turns_seen <= self._voicemail_tool_active_turns
        ):
            return

        remaining = [t for t in self._tools if not _is_voicemail_tool(t)]
        if len(remaining) != len(self._tools):
            logger.info(
                f"Removing voicemail tool after {self._voicemail_tool_active_turns} user turn(s) "
                "(conversation deemed started)"
            )
            self.set_tools(remaining)
        # Set the flag regardless so we only ever attempt removal once.
        self._voicemail_tool_removed = True

    # ------------------------------------------------------------------
    # Approach 2: cheap-LM voicemail detection sidecar
    # ------------------------------------------------------------------

    async def _wrap_with_voicemail_detection(
        self,
        gen: AsyncIterable[OutputEvent],
        transcript: str,
    ) -> AsyncIterable[OutputEvent]:
        """Gate the main response (``gen``) behind the voicemail detector.

        Runs the detector concurrently with the main LM. Buffers the main LM's
        first user-visible output until either the detector returns or the
        ``initial_gate_ms`` gate elapses:

        - ``voicemail`` within the gate → suppress the main output, close the
          main generation, and emit the optional voicemail message plus an
          uninterruptible ``AgentEndCall(reason="voicemail_detected")``.
        - ``human`` / ``unknown`` / error / arrives after the gate → release the
          buffered output and continue normally. A late ``voicemail`` verdict
          after the first released output is ignored.
        """
        assert self._voicemail_detector is not None and self._voicemail_detection is not None
        detector_task = asyncio.create_task(self._voicemail_detector.classify(transcript))
        gate_deadline = time.monotonic() + self._voicemail_detection.initial_gate_ms / 1000.0

        buffered: List[OutputEvent] = []
        released = False
        voicemail_detected = False
        agen = gen.__aiter__()
        try:
            async for output in agen:
                if released or not isinstance(output, _USER_VISIBLE_OUTPUT_TYPES):
                    # Metrics/logs pass through immediately; once released,
                    # everything streams straight to the user.
                    yield output
                    continue

                # First user-visible output while still gating: hold it and
                # resolve the detector exactly once.
                buffered.append(output)
                verdict = await self._resolve_voicemail_gate(detector_task, gate_deadline)
                if verdict == "voicemail":
                    voicemail_detected = True
                    break
                released = True
                for buffered_output in buffered:
                    yield buffered_output
                buffered = []

            if voicemail_detected:
                # Cancel/close the main generation before speaking the fixed message.
                await agen.aclose()
                async for output in self._emit_voicemail_response():
                    yield output
        finally:
            await agen.aclose()
            if not detector_task.done():
                detector_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await detector_task

    async def _resolve_voicemail_gate(
        self,
        detector_task: "asyncio.Task[Any]",
        gate_deadline: float,
    ) -> str:
        """Return the detector verdict if it lands within the gate, else ``"unknown"``.

        Treats timeouts/late results as ``"unknown"`` (release). Uses ``shield``
        so a gate timeout never cancels the in-flight detector — cleanup owns
        cancellation.
        """
        remaining = gate_deadline - time.monotonic()
        if remaining <= 0:
            if detector_task.done():
                return self._detector_classification(detector_task)
            return "unknown"
        try:
            await asyncio.wait_for(asyncio.shield(detector_task), timeout=remaining)
        except Exception:
            # TimeoutError (gate elapsed) or any detector error → fail open.
            return "unknown"
        return self._detector_classification(detector_task)

    @staticmethod
    def _detector_classification(detector_task: "asyncio.Task[Any]") -> str:
        """Best-effort read of a completed detector task's classification."""
        if not detector_task.done() or detector_task.cancelled():
            return "unknown"
        exc = detector_task.exception()
        if exc is not None:
            return "unknown"
        return detector_task.result().classification

    async def _emit_voicemail_response(self) -> AsyncIterable[OutputEvent]:
        """Emit the optional fixed voicemail message, then end the call."""
        assert self._voicemail_detection is not None
        message = self._voicemail_detection.message
        if message:
            text_output = AgentSendText(text=message, interruptible=False)
            self.history._append_local(text_output)
            yield text_output
        end_output = AgentEndCall(reason="voicemail_detected", interruptible=False)
        self.history._append_local(end_output)
        yield end_output

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._handoff_target = None
        await self._get_background_event_queue().wait()
        await self._llm.aclose()
        if self._voicemail_detector is not None:
            await self._voicemail_detector.aclose()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(config: Optional[LlmConfig]) -> None:
        """Validate the config argument passed to process().

        Raises TypeError if config is not None or an LlmConfig instance.
        """
        if config is not None and not isinstance(config, LlmConfig):
            raise TypeError(f"config must be an LlmConfig instance, got {type(config).__name__}")

    @staticmethod
    def _validate_tools(tools: Optional[List[ToolSpec]]) -> None:
        """Validate the tools argument passed to process().

        Raises TypeError if tools is not None, not a list, or contains invalid items.
        """
        if tools is not None:
            if not isinstance(tools, list):
                raise TypeError(f"tools must be a list, got {type(tools).__name__}")
            for i, tool in enumerate(tools):
                if not (
                    isinstance(
                        tool, (FunctionTool, WebSearchTool, EndCallTool, TransferCallTool, VoicemailTool)
                    )
                    or callable(tool)
                ):
                    raise TypeError(
                        f"tools[{i}] must be a FunctionTool, WebSearchTool, EndCallTool, "
                        f"TransferCallTool, VoicemailTool, or callable, got {type(tool).__name__}"
                    )

    @staticmethod
    def _validate_context(context: Union[str, List[HistoryEvent], None]) -> None:
        """Validate the context argument passed to process().

        Raises TypeError if context is not None, a string, or a list of HistoryEvents.
        """
        if context is not None and not isinstance(context, str):
            if not isinstance(context, list):
                raise TypeError(
                    f"context must be a string, list of HistoryEvents, or None, got {type(context).__name__}"
                )
            for i, item in enumerate(context):
                if not isinstance(item, _HISTORY_EVENT_TYPES):
                    raise TypeError(
                        f"context[{i}] must be a HistoryEvent "
                        f"(e.g. UserTextSent, AgentTextSent, AgentToolCalled, CustomHistoryEntry), "
                        f"got {type(item).__name__}"
                    )

    @staticmethod
    def _validate_history(history: Optional[List[HistoryEvent]]) -> None:
        """Validate the history argument passed to process().

        Raises TypeError if history is not None, not a list, or contains non-HistoryEvent items.
        """
        if history is not None:
            if not isinstance(history, list):
                raise TypeError(f"history must be a list of HistoryEvents, got {type(history).__name__}")
            for i, item in enumerate(history):
                if not isinstance(item, _HISTORY_EVENT_TYPES):
                    raise TypeError(
                        f"history[{i}] must be a HistoryEvent "
                        f"(e.g. UserTextSent, AgentTextSent, AgentToolCalled, CustomHistoryEntry), "
                        f"got {type(item).__name__}"
                    )


async def _normalize_result(
    result: Union[AsyncIterable[T], Awaitable[T], T],
) -> AsyncIterable[T]:
    """Normalize any result type to an async iterable.

    Converts: AsyncIterable[T] | Awaitable[T] | T => AsyncIterable[T]
    """
    if inspect.iscoroutine(result) or inspect.isawaitable(result):
        yield await result  # type: ignore[misc]
    elif hasattr(result, "__aiter__"):
        async for item in result:  # type: ignore[union-attr]
            yield item
    else:
        yield result  # type: ignore[misc]


def _normalize_to_async_gen(
    func: Callable[..., Union[AsyncIterable[T], Awaitable[T], T]],
) -> Callable[..., AsyncIterable[T]]:
    """Wrap a function to always return an async generator.

    Converts: Callable[..., AsyncIterable[T] | Awaitable[T] | T] => Callable[..., AsyncIterable[T]]
    """

    async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterable[T]:
        result = func(*args, **kwargs)
        async for item in _normalize_result(result):
            yield item

    return wrapper


def _construct_tool_events(
    tool_call_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
) -> Tuple[AgentToolCalled, AgentToolReturned]:
    """Construct a pair of AgentToolCalled and AgentToolReturned events."""
    called = AgentToolCalled(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
    )
    returned = AgentToolReturned(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
    )
    return called, returned


def _extract_user_transcript(event: UserTurnEnded) -> str:
    """Join the text parts of a completed user turn into a single transcript string."""
    parts = [item.content for item in event.content if isinstance(item, UserTextSent) and item.content]
    return " ".join(parts).strip()


def _set_responding_to(event: OutputEvent, event_id: str) -> OutputEvent:
    """Set responding_to on harness-facing events if not already set.

    Called at the process() yield boundary so the harness knows which input event
    triggered each output event. When event_id is empty string (e.g., no history available),
    responding_to is left unset. Skips events that already have responding_to set
    (e.g., from a custom agent or handed-off agent that set it explicitly).
    """
    if event_id and event.responding_to is None:
        event.responding_to = event_id
    return event
