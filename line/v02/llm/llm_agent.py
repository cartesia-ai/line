"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

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

from line.v02.events import AgentEndCall, SpecificCallEnded
from line.v02.llm.agent import (
    AgentCallable,
    AgentHandedOff,
    AgentSendDtmf,
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallEnded,
    CallStarted,
    InputEvent,
    OutputEvent,
    SpecificAgentDtmfSent,
    SpecificAgentTextSent,
    SpecificInputEvent,
    SpecificUserTextSent,
    ToolEnv,
    TurnEnv,
)
from line.v02.llm.config import LlmConfig
from line.v02.llm.provider import LLMProvider, Message, ToolCall
from line.v02.llm.tool_types import loopback_tool
from line.v02.llm.tool_utils import FunctionTool, ToolType, construct_function_tool
from line.v02.llm.tools import WebSearchTool

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
        self._local_history: List[OutputEvent] = []
        self._handoff_target: Optional[AgentCallable] = None  # Normalized process function

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
            return

        async for output in self._generate_response(env, event):
            yield output

    async def _generate_response(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM."""
        messages = self._build_messages(event.history, self._local_history)

        for _iteration in range(self._max_tool_iterations):
            text_buffer = ""
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
                        text_buffer += chunk.text
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

            should_continue = False
            ctx = ToolEnv(turn_env=env)

            for tc in tool_calls_dict.values():
                if not tc.is_complete:
                    continue

                tool = self._tool_map.get(tc.name)
                if not tool:
                    logger.warning(f"Unknown tool: {tc.name}")
                    continue

                tool_args = json.loads(tc.arguments) if tc.arguments else {}

                tool_called_output = AgentToolCalled(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_args=tool_args,
                )
                self._append_to_local_history(tool_called_output)
                yield tool_called_output

                try:
                    normalized_func = _normalize_to_async_gen(tool.func)

                    if tool.tool_type == ToolType.LOOPBACK:
                        # Collect results to send back to LLM
                        results = []
                        async for value in normalized_func(ctx, **tool_args):
                            results.append(value)
                            tool_returned_output = AgentToolReturned(
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                tool_args=tool_args,
                                result=value,
                            )
                            self._append_to_local_history(tool_returned_output)
                            yield tool_returned_output

                        # Add to messages for next iteration
                        result_str = str(results[0]) if len(results) == 1 else str(results)
                        messages.append(
                            Message(
                                role="assistant",
                                content=text_buffer if text_buffer else None,
                                tool_calls=[tc],
                            )
                        )
                        messages.append(
                            Message(
                                role="tool",
                                content=result_str,
                                tool_call_id=tc.id,
                                name=tc.name,
                            )
                        )
                        should_continue = True
                        text_buffer = ""

                    elif tool.tool_type == ToolType.PASSTHROUGH:
                        async for evt in normalized_func(ctx, **tool_args):
                            self._append_to_local_history(evt)
                            yield evt
                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )
                        self._append_to_local_history(tool_returned_output)
                        yield tool_returned_output

                    elif tool.tool_type == ToolType.HANDOFF:
                        # AgentHandedOff input event is passed to the handoff target to execute the tool
                        async for item in normalized_func(ctx, **tool_args, event=AgentHandedOff()):
                            self._append_to_local_history(item)
                            yield item

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

                        tool_returned_output = AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )
                        self._append_to_local_history(tool_returned_output)
                        yield tool_returned_output

                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    error_output = AgentToolReturned(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                        result=f"error: {e}",
                    )
                    self._append_to_local_history(error_output)
                    yield error_output

            if not should_continue:
                break

    def _build_messages(
        self, input_history: List[SpecificInputEvent], local_history: List[OutputEvent]
    ) -> List[Message]:
        """Build LLM messages from conversation history.

        Merges input_history (canonical) with local_history using the following rules:
        1. Observable events can be matched between local and input history
        2. Unobservable events are interpolated based on their relative position
           to observable events
        3. Unobserved observable events are excluded from the merged history
           (because the audio harness is the source of truth for them)

        The full_history contains:
        - SpecificInputEvent for events from input_history (including matched observables)
        - OutputEvent for unobservable events from local_history
        """
        full_history = _build_full_history(input_history, local_history)

        messages = []
        for event in full_history:
            # Handle SpecificInputEvent types (from input_history)
            if isinstance(event, SpecificUserTextSent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, SpecificAgentTextSent):
                messages.append(Message(role="assistant", content=event.content))
            # Handle OutputEvent types (unobservable events from local_history)
            elif isinstance(event, AgentSendText):
                messages.append(Message(role="assistant", content=event.text))
            elif isinstance(event, AgentToolCalled):
                messages.append(
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=event.tool_call_id,
                                name=event.tool_name,
                                arguments=json.dumps(event.tool_args),
                                is_complete=True,
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

    def _append_to_local_history(self, event: OutputEvent) -> None:
        """Append an output event to local history."""
        self._local_history.append(event)

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._handoff_target = None
        await self._llm.aclose()


def _build_full_history(
    input_history: List[SpecificInputEvent],
    local_history: List[OutputEvent],
) -> List[Union[SpecificInputEvent, OutputEvent]]:
    """
    Build full history by merging input_history (canonical) with local_history.

    Preprocessing:
    - Concatenates contiguous AgentSendText events in local_history
    - Concatenates contiguous SpecificAgentTextSent events in input_history

    Rules:
    1 ) An event is considered "observable" if the external harness tracks it and
        returns it to us in input_history: AgentSendDtmf, AgentSendText, AgentEndCall
         - These can be matched between local and input history
    2. Other events are unobservable (e.g., AgentToolReturned)
    3. Match observable local events to input_history events ("observed")
    4. Input history is canonical; interpolate unobservable events based on
       their relative position to observable events
    5. Unobserved observable events are excluded from full history
    6. For text events, if input is a prefix of local, match and carry forward
       the unmatched suffix
    """
    # Preprocess: concatenate contiguous text events
    preprocessed_local = _concat_contiguous_agent_send_text(local_history)
    preprocessed_input = _concat_contiguous_agent_text_sent(input_history)

    return _build_full_history_rec(preprocessed_input, preprocessed_local)


def _concat_contiguous_agent_send_text(local_history: List[OutputEvent]) -> List[OutputEvent]:
    """Concatenate contiguous AgentSendText events in local history."""

    def reduce_texts(a: OutputEvent, b: OutputEvent) -> List[OutputEvent]:
        if isinstance(a, AgentSendText) and isinstance(b, AgentSendText):
            return [AgentSendText(text=a.text + b.text)]
        return [a, b]

    return _reduce_windowed(local_history, reduce_texts)


def _concat_contiguous_agent_text_sent(
    input_history: List[SpecificInputEvent],
) -> List[SpecificInputEvent]:
    """Concatenate contiguous SpecificAgentTextSent events in input history."""

    def reduce_texts(a: SpecificInputEvent, b: SpecificInputEvent) -> List[SpecificInputEvent]:
        if isinstance(a, SpecificAgentTextSent) and isinstance(b, SpecificAgentTextSent):
            return [SpecificAgentTextSent(content=a.content + b.content)]
        return [a, b]

    return _reduce_windowed(input_history, reduce_texts)


def _build_full_history_rec(
    input_history: List[SpecificInputEvent],
    local_history: List[OutputEvent],
) -> List[Union[SpecificInputEvent, OutputEvent]]:
    """
    Recursive implementation of history merging.

    Algorithm:
    1. Base case: if both histories are empty, return []
    2. If head of input is non-observable: output it, recurse with rest of input
    3. If head of local is non-observable: output it, recurse with rest of local
    4. (Now both heads are observable, or one/both missing)
    5. If both exist and match exactly: output head_input (canonical), recurse with rest of both
    6. If input text is a prefix of local text: output head_input, recurse with suffix prepended to local
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
        return [head_input] + _build_full_history_rec(rest_input, local_history)

    # If head_local is non-observable: output it, continue with rest of local, same input
    if head_local is not None and not _is_local_observable(head_local):
        return [head_local] + _build_full_history_rec(input_history, rest_local)

    # Now both heads are observable (or one/both missing)

    # Try to match: exact match or prefix match
    if head_local is not None and head_input is not None:
        match_result = _try_match_events(head_local, head_input)
        if match_result is not None:
            matched_input, suffix_event = match_result
            new_local = ([suffix_event] if suffix_event else []) + list(rest_local)
            return [matched_input] + _build_full_history_rec(rest_input, new_local)

    # If head_local exists but no match (or head_input missing): skip head_local
    if head_local is not None:
        return _build_full_history_rec(input_history, rest_local)

    # If head_input exists but no head_local left: output head_input (canonical)
    if head_input is not None:
        return [head_input] + _build_full_history_rec(rest_input, local_history)

    # Both are None - should have been caught by base case
    return []


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
    SpecificAgentDtmfSent,
    SpecificAgentTextSent,
    SpecificCallEnded,
)


def _is_input_observable(event: SpecificInputEvent) -> bool:
    """Check if a SpecificInputEvent is observable (can be matched to local history)."""
    return isinstance(event, OBSERVABLE_INPUT_EVENT_TYPES)


def _try_match_events(
    local: OutputEvent, input_evt: SpecificInputEvent
) -> Optional[tuple[SpecificInputEvent, Optional[OutputEvent]]]:
    """Try to match a local observable event to an input observable event.

    Returns:
        None: No match
        (input_evt, None): Exact match - use input_evt as canonical
        (input_evt, suffix_event): Prefix match - use input_evt and carry forward suffix_event

    For text events, supports prefix matching (input is prefix of local).
    For DTMF and EndCall events, only exact matching is supported.
    """
    if isinstance(local, AgentSendText) and isinstance(input_evt, SpecificAgentTextSent):
        if local.text == input_evt.content:
            return (input_evt, None)
        if local.text.startswith(input_evt.content):
            suffix = local.text[len(input_evt.content) :]
            return (input_evt, AgentSendText(text=suffix))
    elif isinstance(local, AgentSendDtmf) and isinstance(input_evt, SpecificAgentDtmfSent):
        if local.button == input_evt.button:
            return (input_evt, None)
    elif isinstance(local, AgentEndCall) and isinstance(input_evt, SpecificCallEnded):
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
