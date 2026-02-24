"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

import asyncio
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
    AgentHandedOff,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    CallEnded,
    CallStarted,
    CustomHistoryEntry,
    HistoryEvent,
    InputEvent,
    LogMetric,
    OutputEvent,
    UserTextSent,
)
from line.llm_agent.config import LlmConfig, _merge_configs, _normalize_config
from line.llm_agent.history import _HISTORY_EVENT_TYPES, History
from line.llm_agent.provider import LLMProvider, Message, ToolCall
from line.llm_agent.tools.decorators import loopback_tool
from line.llm_agent.tools.system import EndCallTool, WebSearchTool
from line.llm_agent.tools.utils import FunctionTool, ToolEnv, ToolType, construct_function_tool

T = TypeVar("T")

# Type alias for tools that can be passed to LlmAgent
# Plain callables are automatically wrapped as loopback tools
ToolSpec = Union[FunctionTool, WebSearchTool, EndCallTool, Callable]


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
        self.history = History()
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

        resolved_tools, web_seach_options = self._resolve_tools(self._tools)
        tool_names = [t.name for t in resolved_tools] + (["web_search"] if web_seach_options else [])
        logger.info(f"LlmAgent initialized with model={self._model}, tools={tool_names}")

    def set_tools(self, tools: List[ToolSpec]) -> None:
        """Replace the agent's tools with a new list."""
        self._tools = tools

    def set_config(self, config: LlmConfig) -> None:
        """Replace the agent's config."""
        self._config = _normalize_config(config)

    def add_history_entry(self, content: str, role: Literal["system", "user"] = "system") -> None:
        """Insert a CustomHistoryEntry event into local history.

        The entry appears as a message with the given role ("system" by default) in the
        LLM conversation
        """
        self.history.add_entry(content, role)

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
        effective_tools = self._merge_tools(self._tools, tools) if tools else self._tools

        # If handoff is active, call the handed-off process function
        if self._handoff_target is not None:
            async for output in self._handoff_target(env, event):
                self.history._append_local(output)
                yield output
            return

        # Handle CallStarted
        if isinstance(event, CallStarted):
            if effective_config.introduction and not self._introduction_sent:
                output = AgentSendText(text=effective_config.introduction)
                self.history._append_local(output)
                self._introduction_sent = True
                yield output
            return

        # Handle CallEnded
        if isinstance(event, CallEnded):
            await self.cleanup()
            return

        async for output in self._generate_response(
            env, event, effective_tools, effective_config, context=context, history=history
        ):
            yield output

    def _get_tool_name(self, tool: ToolSpec) -> str:
        """Extract the name from a ToolSpec.

        Args:
            tool: A ToolSpec (FunctionTool, WebSearchTool, EndCallTool, or Callable)

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
        """Resolve ToolSpecs into FunctionTools and optional web_search_options.

        Separates WebSearchTool from other tools, converts plain callables to
        FunctionTools via loopback_tool, and decides whether to use native web
        search or a fallback tool based on model support.

        Returns:
            (function_tools, web_search_options) — web_search_options is set only
            when the model supports native web search and there are no other
            function tools; otherwise the WebSearchTool is converted to a fallback
            FunctionTool included in the first list.
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
        tools, web_search_options = self._resolve_tools(tool_specs)
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
            messages = await self._build_messages(context=context, history=history)
            tool_calls_dict: Dict[str, ToolCall] = {}

            # Build kwargs for LLM chat, including web_search_options if available
            chat_kwargs: Dict[str, Any] = {}
            if web_search_options:
                chat_kwargs["web_search_options"] = web_search_options

            stream = self._llm.chat(
                messages,
                tools if tools else None,
                config=config,
                **chat_kwargs,
            )
            async with stream:
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
                            self.history._append_local(tool_called_output)
                            self.history._append_local(tool_returned_output)
                            yield tool_called_output
                            yield tool_returned_output
                            n += 1
                    except Exception as e:
                        # Use negative limit to show last 10 frames (most relevant)
                        logger.error(f'Error in Tool Call to "{tc.name}":\n{traceback.format_exc(limit=-10)}')
                        tool_called_output, tool_returned_output = _construct_tool_events(
                            f"{tc.id}-{n}", tc.name, tool_args, f"error: {e}"
                        )
                        self.history._append_local(tool_called_output)
                        self.history._append_local(tool_returned_output)
                        yield tool_called_output
                        yield tool_returned_output

                elif tool.tool_type == ToolType.PASSTHROUGH:
                    # Emit AgentToolCalled before executing
                    tool_called_output = AgentToolCalled(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                    )
                    self.history._append_local(tool_called_output)
                    yield tool_called_output

                    try:
                        async for evt in normalized_func(ctx, **tool_args):
                            self.history._append_local(evt)
                            yield evt
                        # Emit AgentToolReturned after successful completion
                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )
                        self.history._append_local(tool_returned_output)
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
                        self.history._append_local(tool_returned_output)
                        yield tool_returned_output

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
                        yield tool_returned_output

            # ==== END TOOL CALLS ==== #

            has_background_events = not self._background_event_queue.empty()
            has_background_tasks = self._background_task is not None and not self._background_task.done()
            if not (should_loopback or has_background_events or has_background_tasks):
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
        triggering_event_id = self.history._current_event_id

        async def generate_events() -> None:
            n = 0
            try:
                async for value in normalized_func(ctx, **tool_args):
                    call_id = f"{tc_id}-{n}"
                    called, returned = _construct_tool_events(call_id, tc_name, tool_args, value)

                    # Add to local history with the triggering event_id
                    self.history._append_local_with_event_id(called, triggering_event_id)
                    self.history._append_local_with_event_id(returned, triggering_event_id)
                    # Add to queue for loopback processing
                    await self._background_event_queue.put((called, returned))
                    n += 1
            except Exception as e:
                # Use negative limit to show last 10 frames (most relevant)
                logger.error(f"Error in Tool Call {tc_name}: {e}\n{traceback.format_exc(limit=-10)}")
                called, returned = _construct_tool_events(f"{tc_id}-{n}", tc_name, tool_args, f"error: {e}")
                # Add to local history with the triggering event_id
                self.history._append_local_with_event_id(called, triggering_event_id)
                self.history._append_local_with_event_id(returned, triggering_event_id)
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
                if not (isinstance(tool, (FunctionTool, WebSearchTool, EndCallTool)) or callable(tool)):
                    raise TypeError(
                        f"tools[{i}] must be a FunctionTool, WebSearchTool, EndCallTool, "
                        f"or callable, got {type(tool).__name__}"
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
