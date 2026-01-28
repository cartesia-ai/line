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
        # Track input events we've "seen" via incrementing ID
        self._current_input_id: int = 0
        # Local history annotated with the input ID each output is responsive to
        self._local_history: List[tuple[int, OutputEvent]] = []
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
        # Increment input ID to track this new input event
        self._current_input_id += 1

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

        for _iteration in range(self._max_tool_iterations):
            messages = self._build_messages(
                event.history, self._local_history, self._current_input_id
            )
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
                        should_continue = True
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
        self,
        input_history: List[SpecificInputEvent],
        local_history: List[tuple[int, OutputEvent]],
        current_input_id: int,
    ) -> List[Message]:
        """Build LLM messages from conversation history.

        Merges input_history (canonical) with local_history using the following rules:
        1. Events responsive to prior inputs use input_history (canonical) for observables
        2. ToolCalled/ToolCallReturned are interpolated relative to observable events
        3. Events responsive to current input use local_history directly

        The full_history contains:
        - SpecificInputEvent for events from input_history (for prior turns)
        - OutputEvent for current turn events from local_history
        """
        full_history = _build_full_history(input_history, local_history, current_input_id)

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
        """Append an output event to local history, annotated with the current input ID."""
        self._local_history.append((self._current_input_id, event))

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._handoff_target = None
        await self._llm.aclose()


def _build_full_history(
    input_history: List[SpecificInputEvent],
    local_history: List[tuple[int, OutputEvent]],
    current_input_id: int,
) -> List[Union[SpecificInputEvent, OutputEvent]]:
    """
    Build full history by merging input_history (canonical) with local_history.

    Args:
        input_history: Canonical history from the external harness
        local_history: Local events annotated with (input_id, event)
        current_input_id: The ID of the current input being processed

    Algorithm:
    1. Split local_history by input_id:
       - prior_local: events where input_id < current_input_id
       - current_local: events where input_id == current_input_id
    2. Split input_history at the last user input:
       - prior_input: everything up to and including the last user input
       - after_current: observables after the last user input (for current turn)
    3. For prior events: use prior_input (canonical) for observables,
       interpolate ToolCalled/ToolCallReturned relative to observable events
    4. For current events: use local_history directly (not yet observed)

    Preprocessing:
    - Concatenates contiguous AgentSendText events in prior local history
    - Concatenates contiguous SpecificAgentTextSent events in input history
    """
    # Split local history by input_id
    prior_local = [event for input_id, event in local_history if input_id < current_input_id]
    current_local = [event for input_id, event in local_history if input_id == current_input_id]

    # Build history for prior events (canonical from input_history)
    prior_result = _build_prior_history_rec(input_history, prior_local)
    # Append current local events (not yet observed, use local version)
    return prior_result + current_local


def _build_prior_history_rec(
    input_history: List[SpecificInputEvent],
    prior_local: List[OutputEvent],
) -> List[Union[SpecificInputEvent, OutputEvent]]:
    """
    Recursive implementation of prior history merging.

    Merges input_history with prior_local (events responsive to prior inputs).

    Args:
        input_history: Canonical history from external harness
        prior_local: Local events responsive to prior inputs

    Algorithm:
    1. Base case: if both histories are empty, return []
    2. If head of local is non-observable: output it, recurse with rest of local
    3. If head of input is non-observable: output it, recurse with rest of input
    4. (Now both heads are observable, or one/both missing)
    5. If both exist and match exactly: drain any trailing unobservables from rest_local,
       output them + head_input (canonical), recurse with rest of both
    6. If input text is a prefix of local text: same as above, with suffix prepended to local
    7. If head_local exists (no match or no input): skip head_local, recurse with rest of local
    8. If head_input exists but no match in prior_local:
       - If we have local observables: EXCLUDE it (we're generating our own response)
       - If we have no local observables (fresh agent): INCLUDE it (from before we started)
    """
    # Base case: both empty
    if not input_history and not prior_local:
        return []

    head_input = _safe_head(input_history)
    rest_input = input_history[1:] if input_history else []
    head_local = _safe_head(prior_local)
    rest_local = prior_local[1:] if prior_local else []

    # If head_input is non-observable: output it, continue with same local, rest of input
    if head_input is not None and not _is_input_observable(head_input):
        return [head_input] + _build_prior_history_rec( rest_input, prior_local)
    # If head_local is non-observable: output it, continue with rest of local, same input
    if head_local is not None and not _is_local_observable(head_local):
        return [head_local] + _build_prior_history_rec( input_history, rest_local)

    # Now both heads are observable (or one/both missing)

    # Try to match: exact match or prefix match
    if head_local is not None and head_input is not None:
        match_result = _try_match_events(head_local, head_input)
        if match_result is not None:
            matched_input, suffix_event = match_result
            new_local = ([suffix_event] if suffix_event else []) + list(rest_local)
            return [matched_input] + _build_prior_history_rec(rest_input, new_local)

    # If head_local exists but no match (or head_input missing): skip head_local
    if head_local is not None:
        return _build_prior_history_rec(input_history, rest_local)

    # head_local is None (exhausted or never had any), head_input is observable
    if head_input is not None:
        # We have local observables, so exclude unmatched input observables
        # (they're stale/partial or will be replaced by our current output)
        return _build_prior_history_rec(rest_input, prior_local)

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
