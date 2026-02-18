"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

import asyncio
from collections import defaultdict
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
    Literal,
    Optional,
    TypeVar,
    Union,
)

from litellm import get_supported_openai_params
from loguru import logger

from line.agent import AgentCallable, TurnEnv
from line.events import (
    AgentDtmfSent,
    AgentEndCall,
    AgentHandedOff,
    AgentSendDtmf,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTransferCall,
    AgentTurnEnded,
    AgentTurnStarted,
    AgentUpdateCall,
    CallEnded,
    CallStarted,
    CustomHistoryEntry,
    HistoryEvent,
    InputEvent,
    LogMessage,
    LogMetric,
    OutputEvent,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)
from line.llm_agent.config import LlmConfig, _merge_configs, _normalize_config
from line.llm_agent.provider import LLMProvider, Message, ToolCall
from line.llm_agent.tools.decorators import loopback_tool
from line.llm_agent.tools.system import EndCallTool, WebSearchTool
from line.llm_agent.tools.utils import FunctionTool, ToolEnv, ToolType, construct_function_tool

T = TypeVar("T")

# Type alias for tools that can be passed to LlmAgent
# Plain callables are automatically wrapped as loopback tools
ToolSpec = Union[FunctionTool, WebSearchTool, EndCallTool, Callable]

# Type for events stored in local history (OutputEvent or CustomHistoryEntry)
_LocalEvent = Union[OutputEvent, CustomHistoryEntry]

# Sentinel event_id used before the first process() call.
# _build_full_history prepends entries tagged with this ID at the start of the history.
_INIT_EVENT_ID = "__init__"


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
    ):
        if not api_key:
            raise ValueError("Missing API key in LLmAgent initialization")
        supported_params = get_supported_openai_params(model=model)
        if supported_params is None:
            raise ValueError(
                f"Model {model} is not supported. See https://models.litellm.ai/ for supported models."
            )

        # Resolve the base config to insert default values for any _UNSET sentinels.
        effective_config = _normalize_config(config or LlmConfig())
        if effective_config.reasoning_effort is not None and "reasoning_effort" not in supported_params:
            raise ValueError(
                f"Model {model} does not support reasoning_effort. "
                "Remove reasoning_effort from your LlmConfig or use a model that supports it."
            )

        self._model = model
        self._api_key = api_key
        self._config = effective_config
        self._max_tool_iterations = max_tool_iterations

        self._tools: List[ToolSpec] = list(tools or [])

        self._llm = LLMProvider(
            model=self._model,
            api_key=self._api_key,
            config=self._config,
        )

        self._introduction_sent = False
        # Local history annotated with (triggering_event_id, event)
        # The event_id is the stable UUID of the triggering input event
        self._local_history: List[tuple[str, _LocalEvent]] = []
        # Event ID of the current triggering input event (set on each process() call)
        self._current_event_id: str = _INIT_EVENT_ID
        self._handoff_target: Optional[AgentCallable] = None  # Normalized process function
        # Background task for backgrounded tools - None means no pending work
        self._background_task: Optional[asyncio.Task[None]] = None
        # Queue for events from backgrounded tools that need to trigger loopback
        self._background_event_queue: asyncio.Queue[tuple[AgentToolCalled, AgentToolReturned]] = (
            asyncio.Queue()
        )
        # Cache for thought signatures (Gemini 3+ models)
        # Maps tool_call_id -> thought_signature
        self._tool_signatures: Dict[str, str] = {}
        # Registered history transform (called in _build_messages after _build_full_history)
        self._process_history_fn: Optional[
            Callable[[List[HistoryEvent]], Union[List[HistoryEvent], Awaitable[List[HistoryEvent]]]]
        ] = None

        resolved_tools, web_search_options = self._resolve_tools(self._tools)
        tool_names = [t.name for t in resolved_tools] + (["web_search"] if web_search_options else [])
        logger.info(f"LlmAgent initialized with model={self._model}, tools={tool_names}")

    def set_tools(self, tools: List[ToolSpec]) -> None:
        """Replace the agent's tools with a new list."""
        self._tools = tools

    def set_config(self, config: LlmConfig) -> None:
        """Replace the agent's config."""
        self._config = _normalize_config(config)

    def set_history_processor(
        self,
        fn: Callable[[List[HistoryEvent]], Union[List[HistoryEvent], Awaitable[List[HistoryEvent]]]],
    ) -> None:
        """Register a transform that processes the history before message building.

        The transform is called on the history before it's passed to the LLM
        It receives the original history and can filter, reorder, or inject events.
        """
        self._process_history_fn = fn

    def add_history_entry(self, content: str, role: Literal["system", "user"] = "system") -> None:
        """Insert a CustomHistoryEntry event into local history.

        The entry appears as a message with the given role ("system" by default) in the
        LLM conversation
        """
        event = CustomHistoryEntry(content=content, role=role)
        self._append_to_local_history(event)

    async def process(
        self,
        env: TurnEnv,
        event: InputEvent,
        *,
        config: Optional[LlmConfig] = None,
        tools: Optional[List[ToolSpec]] = None,
    ) -> AsyncIterable[OutputEvent]:
        """Process an input event and yield output events.

        Args:
            env: The turn environment.
            event: The input event to process.
            config: Optional LlmConfig to merge with self._config for this call.
            tools: Optional tools to use for this call. Tools with matching names replace
                those in self._tools; other tools from self._tools are preserved.
        """
        # Track the event_id of the triggering input event
        # The triggering event is the last element in event.history
        self._current_event_id = event.history[-1].event_id if event.history else ""

        # Compute effective config and tools for this call
        effective_config = _merge_configs(self._config, config) if config else self._config
        effective_tools = self._merge_tools(self._tools, tools) if tools else self._tools

        # If handoff is active, call the handed-off process function
        if self._handoff_target is not None:
            async for output in self._handoff_target(env, event):
                self._append_to_local_history(output)
                yield output
            return

        # Handle CallStarted
        if isinstance(event, CallStarted):
            if effective_config.introduction and not self._introduction_sent:
                output = AgentSendText(text=effective_config.introduction)
                self._append_to_local_history(output)
                self._introduction_sent = True
                yield output
            return

        # Handle CallEnded
        if isinstance(event, CallEnded):
            await self.cleanup()
            return

        async for output in self._generate_response(env, event, effective_tools, effective_config):
            yield output

    def _get_tool_name(self, tool: ToolSpec) -> str:
        """Extract the name from a ToolSpec.

        Args:
            tool: A ToolSpec (FunctionTool, WebSearchTool, EndCallTool, McpTool, or Callable)

        Returns:
            The name of the tool
        """
        if isinstance(tool, WebSearchTool):
            return "web_search"
        elif isinstance(tool, EndCallTool):
            return tool.name
        elif isinstance(tool, FunctionTool):
            return tool.name
        else:  # Plain callable
            return tool.__name__

    def _merge_tools(
        self, base_tools: List[ToolSpec], override_tools: Optional[List[ToolSpec]]
    ) -> List[ToolSpec]:
        """Merge two tool lists, with override_tools replacing base_tools by name.

        Args:
            base_tools: The base list of tools (typically self._tools)
            override_tools: Tools to merge in, replacing any with matching names

        Returns:
            A merged list where tools from override_tools replace those in base_tools
            that have the same name, and all other base_tools are preserved.
        """
        if not override_tools:
            return base_tools

        # Build a set of names from override_tools
        override_names = {self._get_tool_name(tool) for tool in override_tools}

        # Filter base_tools to exclude any with names in override_names
        filtered_base = [tool for tool in base_tools if self._get_tool_name(tool) not in override_names]

        # Return filtered base + override tools
        return filtered_base + override_tools

    def _resolve_tools(
        self, tool_specs: List[ToolSpec]
    ) -> tuple[List[FunctionTool], Optional[Dict[str, Any]]]:
        """Resolve ToolSpecs into FunctionTools and web_search_options.

        Separates WebSearchTool from other tools, converts plain callables to
        FunctionTools via loopback_tool, and decides whether to use native web
        search or a fallback tool based on model support.

        Returns:
            (function_tools, web_search_options)
        """
        function_tools: List[FunctionTool] = []
        web_search_tool: Optional[WebSearchTool] = None

        for tool in tool_specs:
            if isinstance(tool, WebSearchTool):
                web_search_tool = tool
            elif isinstance(tool, EndCallTool):
                function_tools.append(tool.as_function_tool())
            elif isinstance(tool, FunctionTool):
                function_tools.append(tool)
            else:
                function_tools.append(loopback_tool(tool))

        web_search_options: Optional[Dict[str, Any]] = None
        if web_search_tool is not None:
            if _check_web_search_support(self._model) and not function_tools:
                web_search_options = web_search_tool.get_web_search_options()
            else:
                function_tools.append(_web_search_tool_to_function_tool(web_search_tool))

        return function_tools, web_search_options

    async def _generate_response(
        self,
        env: TurnEnv,
        event: InputEvent,
        tool_specs: List[ToolSpec],
        config: LlmConfig,
    ) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM.

        Args:
            env: The turn environment.
            event: The input event to process.
            tool_specs: ToolSpecs to resolve and use for this call.
            config: The effective LlmConfig for this call.
        """
        tools, web_search_options = self._resolve_tools(tool_specs)
        tool_map: Dict[str, FunctionTool] = {t.name: t for t in tools}

        is_first_iteration = True
        should_loopback = False
        for _iteration in range(self._max_tool_iterations):
            # ==== LOOPBACK MANAGMENT ==== #
            # First, yield any pending events from backgrounded tools
            # These events were produced since the last iteration (or from previous process() calls)
            if is_first_iteration or should_loopback:
                # Drain any immediately available events (non-blocking)
                while not self._background_event_queue.empty():
                    called_evt, returned_evt = self._background_event_queue.get_nowait()
                    yield called_evt
                    yield returned_evt
            else:
                # Otherwise wait for either: background task completes OR new event arrives
                result = await self._maybe_await_background_event()
                if result is None:
                    # Background task completed with no more events
                    # this generation process is completed - exit loop
                    break
                called_evt, returned_evt = result
                yield called_evt
                yield returned_evt

            is_first_iteration = False
            should_loopback = False
            # ==== END LOOPBACK MANAGMENT ==== #

            # ==== GENERATION CALL ==== #
            messages = await self._build_messages(event.history, self._local_history, self._current_event_id)
            tool_calls_dict: Dict[str, ToolCall] = {}

            # Build kwargs for LLM chat, including web_search_options if available
            chat_kwargs: Dict[str, Any] = {}
            if web_search_options:
                chat_kwargs["web_search_options"] = web_search_options

            # Timing metrics
            request_start_time = time.perf_counter()
            first_token_logged = False
            first_agent_text_logged = False

            stream = self._llm.chat(
                messages,
                tools or None,
                config=config,
                **chat_kwargs,
            )
            async with stream:
                async for chunk in stream:
                    # Track time to first chunk (text or tool call)
                    if not first_token_logged and (chunk.text or chunk.tool_calls):
                        first_chunk_ms = (time.perf_counter() - request_start_time) * 1000
                        logger.info(f"Time to first chunk: {first_chunk_ms:.2f}ms")
                        yield LogMetric(name="llm_first_chunk_ms", value=first_chunk_ms)
                        first_token_logged = True

                    if chunk.text:
                        output = AgentSendText(text=chunk.text)
                        self._append_to_local_history(output)

                        # Track time to first text
                        if not first_agent_text_logged:
                            first_text_ms = (time.perf_counter() - request_start_time) * 1000
                            logger.info(f"Time to first text: {first_text_ms:.2f}ms")
                            yield LogMetric(name="llm_first_text_ms", value=first_text_ms)
                            first_agent_text_logged = True

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
                if tool.tool_type == ToolType.LOOPBACK and tool.is_background:
                    # Backgroundable tool: run in a shielded task that survives cancellation
                    # Each yielded value triggers a loopback with AgentToolCalled/AgentToolReturned pair
                    self._execute_backgroundable_tool(normalized_func, ctx, tool_args, tc.id, tc.name)
                    continue

                if tool.tool_type == ToolType.LOOPBACK:
                    should_loopback = True
                    # Regular loopback tool: collect results to send back to LLM
                    n = 0
                    try:
                        async for value in normalized_func(ctx, **tool_args):
                            call_id = f"{tc.id}-{n}"
                            tool_called_output, tool_returned_output = _construct_tool_events(
                                call_id, tc.name, tool_args, value
                            )
                            self._append_to_local_history(tool_called_output)
                            self._append_to_local_history(tool_returned_output)
                            yield tool_called_output
                            yield tool_returned_output
                            n += 1
                    except Exception as e:
                        # Use negative limit to show last 10 frames (most relevant)
                        logger.error(f'Error in Tool Call to "{tc.name}":\n{traceback.format_exc(limit=-10)}')
                        tool_called_output, tool_returned_output = _construct_tool_events(
                            f"{tc.id}-{n}", tc.name, tool_args, f"error: {e}"
                        )
                        self._append_to_local_history(tool_called_output)
                        self._append_to_local_history(tool_returned_output)
                        yield tool_called_output
                        yield tool_returned_output

                elif tool.tool_type == ToolType.PASSTHROUGH:
                    # Emit AgentToolCalled before executing
                    tool_called_output = AgentToolCalled(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                    )
                    self._append_to_local_history(tool_called_output)
                    yield tool_called_output

                    try:
                        async for evt in normalized_func(ctx, **tool_args):
                            self._append_to_local_history(evt)
                            yield evt
                        # Emit AgentToolReturned after successful completion
                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )
                        self._append_to_local_history(tool_returned_output)
                        yield tool_returned_output
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
                        self._append_to_local_history(tool_returned_output)
                        yield tool_returned_output

                elif tool.tool_type == ToolType.HANDOFF:
                    # Emit AgentToolCalled before executing
                    tool_called_output = AgentToolCalled(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                    )
                    self._append_to_local_history(tool_called_output)
                    yield tool_called_output

                    # AgentHandedOff input event is passed to the handoff target to execute the tool
                    handed_off_event = AgentHandedOff()
                    event = AgentHandedOff(
                        history=event.history + [handed_off_event],
                        **{k: v for k, v in handed_off_event.model_dump().items() if k != "history"},
                    )
                    self._append_to_local_history(event)
                    try:
                        async for item in normalized_func(ctx, **tool_args, event=event):
                            self._append_to_local_history(item)
                            yield item
                        # Emit AgentToolReturned after successful completion
                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )
                        self._append_to_local_history(tool_returned_output)
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
                        self._append_to_local_history(tool_returned_output)
                        yield tool_returned_output

            # ==== END TOOL CALLS ==== #

            has_background_events = not self._background_event_queue.empty()
            has_background_tasks = self._background_task is not None and not self._background_task.done()
            if not (should_loopback or has_background_events or has_background_tasks):
                break

    async def _build_messages(
        self,
        input_history: List[InputEvent],
        local_history: List[tuple[str, _LocalEvent]],
        current_event_id: str,
    ) -> List[Message]:
        """Build LLM messages from conversation history.

        Merges input_history (canonical) with local_history using the following rules:
        1. Input history is the source of truth for all events
        2. Local history events are interpolated based on which input event triggered them
        3. Matchable events are matched between local and input history
        4. Non-matchable events (tool calls, custom text) are interpolated relative to matchables
        5. Tool calls without matching results get a "pending" result

        The full_history contains HistoryEvent items:
        - InputEvent for events from input_history (matchable OutputEvents converted to InputEvent
          counterparts)
        - AgentToolCalled/AgentToolReturned for tool interactions from local_history
        - CustomHistoryEntry for injected history entries from local_history
        """
        full_history = _build_full_history(input_history, local_history, current_event_id)

        # Apply registered history transform
        if self._process_history_fn is not None:
            try:
                result = self._process_history_fn(full_history)
                if inspect.isawaitable(result):
                    result = await result
                full_history = _validate_processed_history(result)
            except Exception:
                logger.error(
                    f"History processor failed, using unprocessed history:\n{traceback.format_exc(limit=-10)}"
                )
                # full_history is unchanged — fall back to unprocessed version

        # First pass: collect all tool_call_ids that have matching AgentToolReturned
        returned_tool_call_ids: set[str] = set()
        for event in full_history:
            if isinstance(event, AgentToolReturned):
                returned_tool_call_ids.add(event.tool_call_id)

        messages = []
        for event in full_history:
            # Handle InputEvent types
            if isinstance(event, UserTextSent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, AgentTextSent):
                messages.append(Message(role="assistant", content=event.content))
            # Handle CustomHistoryEntry (injected history entries)
            elif isinstance(event, CustomHistoryEntry):
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
    ) -> None:
        """Execute a backgroundable tool in a shielded task, streaming events.

        The task is protected from cancellation. If the calling coroutine is
        cancelled, the task continues running and stores results to local_history.

        Each value yielded by the tool produces a pair of:
        - AgentToolCalled with tool_call_id = "{tc_id}-{n}"
        - AgentToolReturned with the same tool_call_id

        Events are added to _background_event_queue for loopback processing.
        If the caller is cancelled, events continue to be produced and queued
        for processing on the next process() call.
        """
        # Capture the event_id at the start - this is the triggering event
        triggering_event_id = self._current_event_id

        async def generate_events() -> None:
            n = 0
            try:
                async for value in normalized_func(ctx, **tool_args):
                    call_id = f"{tc_id}-{n}"
                    called, returned = _construct_tool_events(call_id, tc_name, tool_args, value)

                    # Add to local history with the triggering event_id
                    self._local_history.append((triggering_event_id, called))
                    self._local_history.append((triggering_event_id, returned))
                    # Add to queue for loopback processing
                    await self._background_event_queue.put((called, returned))
                    n += 1
            except Exception as e:
                # Use negative limit to show last 10 frames (most relevant)
                logger.error(f"Error in Tool Call {tc_name}: {e}\n{traceback.format_exc(limit=-10)}")
                called, returned = _construct_tool_events(f"{tc_id}-{n}", tc_name, tool_args, f"error: {e}")
                # Add to local history with the triggering event_id
                self._local_history.append((triggering_event_id, called))
                self._local_history.append((triggering_event_id, returned))
                # Add to queue for loopback processing
                await self._background_event_queue.put((called, returned))

        # Chain this task after the current background task
        # Use shield to protect from cancellation
        future = asyncio.shield(generate_events())
        old_background_task = self._background_task

        async def _new_background_task() -> None:
            if old_background_task is not None:
                await old_background_task
            await future

        self._background_task = asyncio.ensure_future(_new_background_task())

    def _append_to_local_history(self, event: _LocalEvent) -> None:
        """Append an event to local history, annotated with the triggering event_id."""
        self._local_history.append((self._current_event_id, event))

    async def _maybe_await_background_event(self) -> Union[None, tuple[AgentToolCalled, AgentToolReturned]]:
        """Wait for either a background event or background task completion.

        Cleans up get_event if the background task completes first.
        Intentionally does not clean up the background task if get_event completes first,
        since #cleanup handles that

        Returns:
            - (AgentToolCalled, AgentToolReturned) if a new event is available
            - None if the background task completed with no more events
        """
        # If no background task, there's nothing to wait for
        if self._background_task is None:
            return None

        get_event_task = asyncio.ensure_future(self._background_event_queue.get())
        done, _ = await asyncio.wait(
            [get_event_task, self._background_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Check if the get_event task completed
        if get_event_task in done:
            return get_event_task.result()

        # Background task completed first - cancel the get_event task
        get_event_task.cancel()
        try:
            await get_event_task
        except asyncio.CancelledError:
            pass
        return None

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._handoff_target = None
        # Wait for any remaining background task to complete
        if self._background_task is not None:
            await self._background_task

        await self._llm.aclose()


def _check_web_search_support(model: str) -> bool:
    """Check if a model supports native web search via litellm.

    Returns True if the model supports web_search_options, False otherwise.
    """
    try:
        import litellm

        return litellm.supports_web_search(model=model)
    except (ImportError, AttributeError, Exception):
        # If litellm doesn't have supports_web_search or any error occurs,
        # fall back to the tool-based approach
        return False


def _web_search_tool_to_function_tool(web_search_tool: WebSearchTool) -> FunctionTool:
    """Convert a WebSearchTool to a FunctionTool for use as a fallback.

    When the LLM doesn't support native web search, we use the WebSearchTool's
    search method as a regular loopback tool.
    """
    return construct_function_tool(
        func=web_search_tool.search,
        name="web_search",
        description="Search the web for real-time information."
        + " Use this when you need current information that may not be in your training data.",
        tool_type=ToolType.LOOPBACK,
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
) -> tuple[AgentToolCalled, AgentToolReturned]:
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


def _build_full_history(
    input_history: List[InputEvent],
    local_history: List[tuple[str, _LocalEvent]],
    current_event_id: str,
) -> List[HistoryEvent]:
    """
    We have a split brain situation, where "input_history" (maintained by the external harness) is the source
    of truth for what actually happened in the conversation, but "local_history" (maintained by the agent)
    contains additional events that we want to include in the history (tool calls, custom entries). we need
    to merge them together in a coherent way before building messages for the LLM.

    To that end, we have certain events that are "matchable". They show up in both input and local history,
    but the input version is considered "canonical"
    (it's what actually occured in the conversation)

    Invariants:
    1. Input history ordering is preserved exactly
    2. Every local non-matchable event appears exactly once in output
    3. Matchable locals appear after their triggering input event
    4. Non-matchable locals maintain their original relative order
    5. Matchable events use canonical (input) version
    6. Unmatched local matchables are dropped

    Algorithm:
    1) For each input event, if it triggered any output events, load a queue
    of those local output events.
    2) Unmatchable input events are output directly.
    3) Matchable input events are matched against the queue, with non-matchable locals (tool calls, custom
    entries) drained alongside their matched matchable.

    Examples:

    1) Simple turn with a tool call:

        input_history:  [UserTextSent("hi", id=A), AgentTextSent("hello", id=B)]
        local_history:  [(A, AgentToolCalled(...)), (A, AgentToolReturned(...)), (A, AgentSendText("hello"))]

        result: [UserTextSent("hi"), AgentToolCalled(...), AgentToolReturned(...), AgentTextSent("hello")]

        UserTextSent is non-matchable so it's emitted directly. AgentTextSent is matchable and matches
        the local AgentSendText("hello"), so the canonical input version is used. The non-matchable tool
        events are drained alongside it.

    2) Multi-turn with interleaved tool calls:

        input_history:  [UserTextSent("weather?", id=A), AgentTextSent("It's sunny", id=B),
                         UserTextSent("thanks", id=C), AgentTextSent("You're welcome", id=D)]
        local_history:  [(A, AgentToolCalled(weather)), (A, AgentToolReturned(weather)),
                         (A, AgentSendText("It's sunny")),
                         (C, AgentSendText("You're welcome"))]

        result: [UserTextSent("weather?"), AgentToolCalled(weather), AgentToolReturned(weather),
                 AgentTextSent("It's sunny"), UserTextSent("thanks"), AgentTextSent("You're welcome")]

        When we reach UserTextSent("thanks", id=C), its id is in local_by_event_id so we flush the
        remaining queue from the previous trigger (nothing left) and load the new queue. The tool events
        from the first turn are properly interleaved before the matched AgentTextSent.

    """
    # Split local history into prior and current based on event_id
    prior_local = [(eid, e) for eid, e in local_history if eid != current_event_id]
    current_local = [e for eid, e in local_history if eid == current_event_id]

    # Build map from event_id to list of responsive local events
    local_by_event_id: dict[str, List[_LocalEvent]] = defaultdict(list)
    for eid, event in prior_local:
        local_by_event_id[eid].append(event)

    # Prepend init entries (added before the first process() call)
    init_events = local_by_event_id.pop(_INIT_EVENT_ID, [])

    result: List[Union[InputEvent, _LocalEvent]] = list(init_events)
    queue: List[_LocalEvent] = []

    for input_evt in input_history:
        # If this input event generated any output events
        # flush old output events (they are semantically "prior" to this input event)
        # and load new queue of output events triggered by this input event
        if input_evt.event_id in local_by_event_id:
            result.extend(_flush_queue(queue))
            local_slice = local_by_event_id[input_evt.event_id]
            queue = _concat_contiguous_agent_send_text(local_slice)

        if not _is_input_matchable(input_evt):
            # Non-matchable input (UserTextSent, etc.) — emit directly
            result.append(input_evt)
        else:
            # Matchable input — match against queue
            emitted, queue = _match_matchable(input_evt, queue)
            result.extend(emitted)

    # Flush any remaining locals from the last trigger
    result.extend(_flush_queue(queue))

    # Append current-turn events (not yet observed, use local version)
    result.extend(current_local)

    # Convert matchable OutputEvents to InputEvent counterparts
    return [h for e in result if (h := _to_history_event(e)) is not None]


def _flush_queue(queue: List[_LocalEvent]) -> List[_LocalEvent]:
    """Return non-matchable events from queue, discarding unmatched matchables."""
    return [e for e in queue if not _is_local_matchable(e)]


def _split_leading_non_matchables(
    queue: List[_LocalEvent],
) -> tuple[List[_LocalEvent], List[_LocalEvent]]:
    """Split queue into leading non-matchables and the rest.

    Returns:
        (non_matchables, remaining) where remaining starts at the first matchable
        or is empty if none exist.
    """
    for i, event in enumerate(queue):
        if _is_local_matchable(event):
            return queue[:i], queue[i:]
    return queue, []


def _match_matchable(
    input_evt: InputEvent,
    queue: List[_LocalEvent],
) -> tuple[List[Union[InputEvent, _LocalEvent]], List[_LocalEvent]]:
    """Match a matchable input event against the local queue.

    Splits off leading non-matchables, then tries to match the head matchable.
    On match, emits canonical input version and splits off trailing non-matchables.
    On prefix match, also prepends the suffix to the remaining queue.
    On no match, discards the local matchable and retries.
    If queue exhausted with no match, emits input as-is.

    Returns:
        (emitted_events, remaining_queue)
    """
    remaining = queue
    emitted: List[Union[InputEvent, _LocalEvent]] = []

    while remaining:
        non_obs, rest = _split_leading_non_matchables(remaining)
        emitted.extend(non_obs)

        if not rest:
            break

        head_local = rest[0]
        match_result = _try_match_events(head_local, input_evt)

        if match_result is not None:
            matched_input, suffix_event = match_result
            emitted.append(matched_input)
            # Split off non-matchables that follow the matched matchable
            trailing_non_obs, after = _split_leading_non_matchables(rest[1:])
            emitted.extend(trailing_non_obs)
            # On prefix match, prepend suffix to remaining queue
            remaining = ([suffix_event] + after) if suffix_event is not None else after
            return emitted, remaining

        # No match — discard this local matchable and try next
        remaining = rest[1:]

    # Queue exhausted with no match — emit input as-is
    emitted.append(input_evt)
    return emitted, []


_HISTORY_EVENT_TYPES = (
    # InputEvent types
    CallStarted,
    CallEnded,
    AgentHandedOff,
    UserTurnStarted,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    AgentTurnStarted,
    AgentTextSent,
    AgentDtmfSent,
    AgentTurnEnded,
    # Tool events
    AgentToolCalled,
    AgentToolReturned,
    # Custom entries
    CustomHistoryEntry,
)


def _validate_processed_history(history: Any) -> List[HistoryEvent]:
    """Validate that set_history_processor returned a List[HistoryEvent].

    Raises TypeError if the return value is not a list or contains non-HistoryEvent items.
    """
    if not isinstance(history, list):
        raise TypeError(
            f"set_history_processor callback must return List[HistoryEvent], got {type(history).__name__}"
        )
    for i, event in enumerate(history):
        if not isinstance(event, _HISTORY_EVENT_TYPES):
            raise TypeError(
                f"set_history_processor callback returned invalid event at index {i}: "
                f"expected HistoryEvent, got {type(event).__name__}"
            )
    return history


def _to_history_event(event: Any) -> Optional[HistoryEvent]:
    """Convert an event to a HistoryEvent.

    Matchable OutputEvents are converted to their InputEvent counterparts.
    Non-history OutputEvents (LogMetric, etc.) are filtered out (returns None).
    All other events (InputEvent, AgentToolCalled, AgentToolReturned, CustomHistoryEntry)
    pass through unchanged.
    """
    # Matchable OutputEvents → convert to InputEvent counterparts
    if isinstance(event, AgentSendText):
        return AgentTextSent(content=event.text)
    elif isinstance(event, AgentSendDtmf):
        return AgentDtmfSent(button=event.button)
    elif isinstance(event, AgentEndCall):
        return CallEnded()
    # HistoryEvent pass-through (tool events, custom entries)
    elif isinstance(event, (AgentToolCalled, AgentToolReturned, CustomHistoryEntry)):
        return event
    # InputEvent types pass through
    elif isinstance(
        event,
        (
            CallStarted,
            CallEnded,
            AgentHandedOff,
            UserTurnStarted,
            UserDtmfSent,
            UserTextSent,
            UserTurnEnded,
            AgentTurnStarted,
            AgentTextSent,
            AgentDtmfSent,
            AgentTurnEnded,
        ),
    ):
        return event
    # Non-history OutputEvents are filtered out
    elif isinstance(event, (AgentTransferCall, LogMetric, LogMessage, AgentUpdateCall)):
        return None
    else:
        raise ValueError(f"Unknown event type in history: {type(event).__name__}")


def _concat_contiguous_agent_send_text(local_history: List[_LocalEvent]) -> List[_LocalEvent]:
    """
    Since the LLM streams output, we likely will have many AgentSendText events that are part of the same
    logical "message" from the LLM. This concats them into a single AgentSendText event for easier matching
    to the input history, which only has one AgentTextSent per LLM message.
    """
    if not local_history:
        return []
    result: List[_LocalEvent] = []
    current = local_history[0]
    for event in local_history[1:]:
        if isinstance(current, AgentSendText) and isinstance(event, AgentSendText):
            current = AgentSendText(text=current.text + event.text)
        else:
            result.append(current)
            current = event
    result.append(current)
    return result


# Matchable OutputEvent types - these can be matched between local and input history
# Corresponds to events that the external system tracks/observes
MATCHABLE_OUTPUT_EVENT_TYPES = (
    AgentSendDtmf,  # => AgentDtmfSent
    AgentSendText,  # => AgentTextSent
    AgentEndCall,  # => CallEnded
)


def _is_local_matchable(event: _LocalEvent) -> bool:
    """Check if a local event is matchable (can be matched to input history)."""
    return isinstance(event, MATCHABLE_OUTPUT_EVENT_TYPES)


MATCHABLE_INPUT_EVENT_TYPES = (
    AgentDtmfSent,
    AgentTextSent,
    CallEnded,
)


def _is_input_matchable(event: InputEvent) -> bool:
    """Check if an InputEvent is matchable (can be matched to local history)."""
    return isinstance(event, MATCHABLE_INPUT_EVENT_TYPES)


def _try_match_events(
    local: _LocalEvent, input_evt: InputEvent
) -> Optional[tuple[InputEvent, Optional[_LocalEvent]]]:
    """Try to match a local matchable event to an input matchable event.

    Returns:
        None: No match
        (input_evt, None): Exact match - use input_evt as canonical
        (input_evt, suffix_event): Prefix match - use input_evt and carry forward suffix_event

    For text events, supports prefix matching (input is prefix of local).
    For DTMF and EndCall events, only exact matching is supported.
    """
    if isinstance(local, AgentSendText) and isinstance(input_evt, AgentTextSent):
        if local.text == input_evt.content:
            return (input_evt, None)
        if local.text.startswith(input_evt.content):
            suffix = local.text[len(input_evt.content) :]
            return (input_evt, AgentSendText(text=suffix))
    elif isinstance(local, AgentSendDtmf) and isinstance(input_evt, AgentDtmfSent):
        if local.button == input_evt.button:
            return (input_evt, None)
    elif isinstance(local, AgentEndCall) and isinstance(input_evt, CallEnded):
        return (input_evt, None)
    return None
