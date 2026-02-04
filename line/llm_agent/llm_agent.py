"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

import asyncio
import inspect
import json
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

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
    CallEnded,
    CallStarted,
    InputEvent,
    OutputEvent,
    UserTextSent,
)
from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import LLMProvider, Message, ToolCall
from line.llm_agent.tools.decorators import loopback_tool
from line.llm_agent.tools.system import WebSearchTool
from line.llm_agent.tools.utils import FunctionTool, ToolEnv, ToolType, construct_function_tool

T = TypeVar("T")

# Type alias for tools that can be passed to LlmAgent
# Plain callables are automatically wrapped as loopback tools
ToolSpec = Union[FunctionTool, WebSearchTool, Callable]


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
        api_key: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        config: Optional[LlmConfig] = None,
        max_tool_iterations: int = 10,
    ):
        self._model = model
        self._api_key = api_key
        self._config = config or LlmConfig()
        self._max_tool_iterations = max_tool_iterations

        # Process tools: separate WebSearchTool from regular FunctionTools
        self._web_search_options: Optional[Dict[str, Any]] = None
        self._tools: List[FunctionTool] = []

        for tool in tools or []:
            if isinstance(tool, WebSearchTool):
                # Check if model supports native web search
                if _check_web_search_support(model):
                    # Use native web search via web_search_options
                    self._web_search_options = tool.get_web_search_options()
                    logger.info(f"Model {model} supports native web search, using web_search_options")
                else:
                    # Fall back to tool-based web search
                    fallback_tool = _web_search_tool_to_function_tool(tool)
                    self._tools.append(fallback_tool)
                    logger.info(f"Model {model} doesn't support native web search, using fallback tool")
            elif isinstance(tool, FunctionTool):
                self._tools.append(tool)
            else:
                # Plain callable - wrap as loopback tool
                self._tools.append(loopback_tool(tool))

        self._tool_map: Dict[str, FunctionTool] = {t.name: t for t in self._tools}
        self._llm = LLMProvider(
            model=self._model,
            api_key=self._api_key,
            config=self._config,
            num_retries=self._config.num_retries,
            fallbacks=self._config.fallbacks,
            timeout=self._config.timeout,
        )

        self._introduction_sent = False
        # Local history annotated with (triggering_event_id, event)
        # The event_id is the stable UUID of the triggering input event
        self._local_history: List[tuple[str, OutputEvent]] = []
        # Event ID of the current triggering input event (set on each process() call)
        self._current_event_id: str = ""
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

        logger.info(f"LlmAgent initialized with model={self._model}, tools={[t.name for t in self._tools]}")

    @property
    def model(self) -> str:
        return self._model

    @property
    def tools(self) -> List[FunctionTool]:
        return self._tools

    @property
    def config(self) -> LlmConfig:
        return self._config

    @property
    def handoff_target(self) -> Optional[AgentCallable]:
        """The normalized process function we've handed off to, if any."""
        return self._handoff_target

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Process an input event and yield output events."""
        # Track the event_id of the triggering input event
        # The triggering event is the last element in event.history
        self._current_event_id = event.history[-1].event_id if event.history else ""

        # If handoff is active, call the handed-off process function
        if self._handoff_target is not None:
            async for output in self._handoff_target(env, event):
                self._append_to_local_history(output)
                yield output
            return

        # Handle CallStarted
        if isinstance(event, CallStarted):
            if self._config.introduction and not self._introduction_sent:
                output = AgentSendText(text=self._config.introduction)
                self._append_to_local_history(output)
                self._introduction_sent = True
                yield output
            return

        # Handle CallEnded
        if isinstance(event, CallEnded):
            await self.cleanup()
            return

        async for output in self._generate_response(env, event):
            yield output

    async def _generate_response(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM."""

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
            messages = self._build_messages(event.history, self._local_history, self._current_event_id)
            tool_calls_dict: Dict[str, ToolCall] = {}

            # Build kwargs for LLM chat, including web_search_options if available
            chat_kwargs: Dict[str, Any] = {}
            if self._web_search_options:
                chat_kwargs["web_search_options"] = self._web_search_options

            stream = self._llm.chat(
                messages,
                self._tools if self._tools else None,
                **chat_kwargs,
            )
            async with stream:
                async for chunk in stream:
                    if chunk.text:
                        output = AgentSendText(text=chunk.text)
                        self._append_to_local_history(output)
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

                tool = self._tool_map.get(tc.name)
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

                try:
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
                            logger.error(f"Loopback tool execution error: {e}")
                            tool_called_output, tool_returned_output = _construct_tool_events(
                                f"{tc.id}-{n}", tc.name, tool_args, f"error: {e}"
                            )
                            self._append_to_local_history(tool_called_output)
                            self._append_to_local_history(tool_returned_output)
                            yield tool_called_output
                            yield tool_returned_output

                    elif tool.tool_type == ToolType.PASSTHROUGH:
                        # Emit AgentToolCalled before executing
                        # This is a hack to deal with the fact that most LLMs
                        # expect a tool call/tool call response pair
                        # to come back to back
                        # for better error management, we should wait till the passthrough
                        # completes before emitting the returned event
                        # and include the error in it if any:
                        # https://linear.app/cartesia/issue/PRO-1669/fix-error-management-for-tool-calls
                        tool_called_output, tool_returned_output = _construct_tool_events(
                            tc.id, tc.name, tool_args, "success"
                        )
                        self._append_to_local_history(tool_called_output)
                        self._append_to_local_history(tool_returned_output)
                        yield tool_called_output
                        yield tool_returned_output

                        try:
                            async for evt in normalized_func(ctx, **tool_args):
                                self._append_to_local_history(evt)
                                yield evt
                        except Exception as e:
                            logger.error(f"Passthrough tool execution error: {e}")

                    elif tool.tool_type == ToolType.HANDOFF:
                        # Emit AgentToolCalled before executing
                        # This is a hack to deal with the fact that most LLMs
                        # expect a tool call/tool call response pair
                        # to come back to back
                        # for better error management, we should wait till the handoff generation
                        # completes before emitting the returned event
                        # and include the error in it if any:
                        # https://linear.app/cartesia/issue/PRO-1669/fix-error-management-for-tool-calls
                        tool_called_output, tool_returned_output = _construct_tool_events(
                            tc.id, tc.name, tool_args, "success"
                        )
                        self._append_to_local_history(tool_called_output)
                        self._append_to_local_history(tool_returned_output)
                        yield tool_called_output
                        yield tool_returned_output

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
                        except Exception as e:
                            logger.error(f"Handoff tool execution error: {e}")

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
                    logger.error(f"Tool execution error: {e}")
            # ==== END TOOL CALLS ==== #

            has_background_events = not self._background_event_queue.empty()
            has_background_tasks = self._background_task is not None and not self._background_task.done()
            if not (should_loopback or has_background_events or has_background_tasks):
                break

    def _build_messages(
        self,
        input_history: List[InputEvent],
        local_history: List[tuple[str, OutputEvent]],
        current_event_id: str,
    ) -> List[Message]:
        """Build LLM messages from conversation history.

        Merges input_history (canonical) with local_history using the following rules:
        1. Input history is the source of truth for all events
        2. Local history events are interpolated based on which input event triggered them
        3. Observable events are matched between local and input history
        4. Unobservable events (tool calls) are interpolated relative to observables

        The full_history contains:
        - InputEvent (with history=None) for events from input_history
        - OutputEvent for unobservable/current events from local_history
        """
        full_history = _build_full_history(input_history, local_history, current_event_id)

        messages = []
        for event in full_history:
            # Handle InputEvent types (from input_history)
            if isinstance(event, UserTextSent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, AgentTextSent):
                messages.append(Message(role="assistant", content=event.content))
            # Handle OutputEvent types (unobservable events from local_history)
            elif isinstance(event, AgentSendText):
                messages.append(Message(role="assistant", content=event.text))
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
                logger.error(f"Background tool execution error: {e}")
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

    def _append_to_local_history(self, event: OutputEvent) -> None:
        """Append an output event to local history, annotated with the triggering event_id."""
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


def _build_full_history(
    input_history: List[InputEvent],
    local_history: List[tuple[str, OutputEvent]],
    current_event_id: str,
) -> List[Union[InputEvent, OutputEvent]]:
    """
    Build full history by merging input_history (canonical) with local_history.

    Args:
        input_history: Canonical history from the external harness (source of truth)
        local_history: Local events annotated with (triggering_event_id, event)
        current_event_id: Event ID of the current input event being processed

    Algorithm:
    1. Split local history into prior (already observed) and current (not yet observed)
    2. Build a map from event_id to list of responsive local events
    3. Iterate over input events:
       - If the event has responsive local events, process a "slice" of input
         from this event to the next triggering event using interpolation logic
       - If no responsive local events, output the event as-is
    4. Append current local events at the end

    Rules:
    - Input history is the source of truth for all observable events
    - Observable events: AgentSendText, AgentSendDtmf, AgentEndCall (and their input counterparts)
    - Unobservable events (tool calls) are interpolated relative to observables
    - Prefix matching: if input text is prefix of local text, match and carry forward suffix
    """
    from collections import defaultdict

    # Split local history into prior and current based on event_id
    prior_local = [(eid, e) for eid, e in local_history if eid != current_event_id]
    current_local = [e for eid, e in local_history if eid == current_event_id]

    # Build map from event_id to list of responsive local events
    local_by_event_id: dict[str, List[OutputEvent]] = defaultdict(list)
    for eid, event in prior_local:
        local_by_event_id[eid].append(event)

    # Build a set of event_ids that have responsive local events
    trigger_event_ids = set(local_by_event_id.keys())

    # Process input in slices
    result: List[Union[InputEvent, OutputEvent]] = []
    i = 0

    while i < len(input_history):
        current_evt = input_history[i]
        if current_evt.event_id in trigger_event_ids:
            # Find the end of this slice (next trigger index or end of input)
            slice_end = len(input_history)
            for j in range(i + 1, len(input_history)):
                if input_history[j].event_id in trigger_event_ids:
                    slice_end = j
                    break

            # Get the slice of input and the responsive local events
            input_slice = input_history[i:slice_end]
            local_slice = local_by_event_id[current_evt.event_id]

            # Preprocess local slice for concatenation
            preprocessed_local_slice = _concat_contiguous_agent_send_text(local_slice)

            # Apply interpolation logic to this slice
            slice_result = _build_history_rec(input_slice, preprocessed_local_slice)
            result.extend(slice_result)

            i = slice_end
        else:
            # No responsive local events, output as-is
            result.append(input_history[i])
            i += 1

    # Append current local events (not yet observed, use local version)
    result.extend(current_local)

    return result


def _concat_contiguous_agent_send_text(local_history: List[OutputEvent]) -> List[OutputEvent]:
    """Concatenate contiguous AgentSendText events in local history."""

    def reduce_texts(a: OutputEvent, b: OutputEvent) -> List[OutputEvent]:
        if isinstance(a, AgentSendText) and isinstance(b, AgentSendText):
            return [AgentSendText(text=a.text + b.text)]
        return [a, b]

    return _reduce_windowed(local_history, reduce_texts)


def _build_history_rec(
    input_history: List[InputEvent],
    local_history: List[OutputEvent],
) -> List[Union[InputEvent, OutputEvent]]:
    """
    Recursive implementation of history merging.

    Algorithm:
    1. Base case: if both histories are empty, return []
    2. If head of local is non-observable: output it, recurse with rest of local
    3. If head of input is non-observable: output it, recurse with rest of input
    4. (Now both heads are observable, or one/both missing)
    5. If both exist and match exactly: drain any trailing unobservables from rest_local,
       output them + head_input (canonical), recurse with rest of both
    6. If input text is a prefix of local text: same as above, with suffix prepended to local
    7. If head_local exists (no match or no input): skip it, recurse with rest of local
    8. If head_input exists (no local left): output it, recurse with rest of input
    """
    # Base case: both empty
    if not input_history and not local_history:
        return []

    head_input = _safe_head(input_history)
    rest_input = input_history[1:] if input_history else []
    head_local = _safe_head(local_history)
    rest_local = local_history[1:] if local_history else []

    # If head_input is non-observable: output it, continue with same local, rest of input
    if head_input is not None and not _is_input_observable(head_input):
        return [head_input] + _build_history_rec(rest_input, local_history)
    # If head_local is non-observable: output it, continue with rest of local, same input
    if head_local is not None and not _is_local_observable(head_local):
        return [head_local] + _build_history_rec(input_history, rest_local)

    # Now both heads are observable (or one/both missing)

    # Try to match: exact match or prefix match
    if head_local is not None and head_input is not None:
        match_result = _try_match_events(head_local, head_input)
        if match_result is not None:
            matched_input, suffix_event = match_result
            # Drain any leading unobservable events from rest_local after outputting the match.
            # This ensures tool results are grouped with the agent's text that preceded them
            # in local_history, preserving the original ordering of events.
            drained_unobservables, remaining_local = _drain_leading_unobservables(rest_local)
            new_local = ([suffix_event] if suffix_event else []) + list(remaining_local)
            return [matched_input] + drained_unobservables + _build_history_rec(rest_input, new_local)

    # If head_local exists but no match (or head_input missing): skip head_local
    if head_local is not None:
        return _build_history_rec(input_history, rest_local)

    # If head_input exists but no head_local left: output head_input (canonical)
    if head_input is not None:
        return [head_input] + _build_history_rec(rest_input, local_history)

    # Both are None - should have been caught by base case
    return []


def _drain_leading_unobservables(
    local_history: List[OutputEvent],
) -> tuple[List[OutputEvent], List[OutputEvent]]:
    """Drain leading non-observable events from local_history.

    Only drains AgentToolReturned events (not AgentToolCalled). This is because
    AgentToolReturned completes a tool call that was already started (and output)
    before the current match, so it should be grouped with the match. But
    AgentToolCalled starts a NEW tool call that belongs to a later part of the
    conversation and should not be drained.

    Returns:
        (drained_events, remaining_events)
    """
    drained: List[OutputEvent] = []
    remaining = list(local_history)
    while remaining:
        head = remaining[0]
        # Only drain AgentToolReturned - stop at anything else (including AgentToolCalled)
        if isinstance(head, AgentToolReturned):
            drained.append(remaining.pop(0))
        else:
            break
    return drained, remaining


# Observable OutputEvent types - these can be matched between local and input history
# Corresponds to events that the external system tracks/observes
OBSERVABLE_OUTPUT_EVENT_TYPES = (
    AgentSendDtmf,  # => AgentDtmfSent
    AgentSendText,  # => AgentTextSent
    AgentEndCall,  # => CallEnded
)


def _is_local_observable(event: OutputEvent) -> bool:
    """Check if an OutputEvent is observable (can be matched to input history)."""
    return isinstance(event, OBSERVABLE_OUTPUT_EVENT_TYPES)


OBSERVABLE_INPUT_EVENT_TYPES = (
    AgentDtmfSent,
    AgentTextSent,
    CallEnded,
)


def _is_input_observable(event: InputEvent) -> bool:
    """Check if an InputEvent is observable (can be matched to local history)."""
    return isinstance(event, OBSERVABLE_INPUT_EVENT_TYPES)


def _try_match_events(
    local: OutputEvent, input_evt: InputEvent
) -> Optional[tuple[InputEvent, Optional[OutputEvent]]]:
    """Try to match a local observable event to an input observable event.

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


def _safe_head(lst: list) -> Optional[Any]:
    """Return the first element of a list, or None if empty."""
    return lst[0] if lst else None


def _reduce_windowed(lst: List[T], reduce: Callable[[T, T], List[T]]) -> List[T]:
    """Reduce a list by applying a function to consecutive pairs.

    The reduce function takes two consecutive elements and returns:
    - A single-element list if they should be merged
    - A two-element list [a, b] if they should remain separate

    The function processes the list left-to-right, using the result of each
    reduction as the left element for the next pair.
    """
    if len(lst) <= 1:
        return lst.copy()

    result: List[T] = []
    current = lst[0]
    for i in range(1, len(lst)):
        reduced = reduce(current, lst[i])
        current = reduced[-1]
        if len(reduced) == 2:
            # Not merged: output current, use second element as new current
            result.append(reduced[0])

    # Don't forget the last current element
    result.append(current)
    return result
