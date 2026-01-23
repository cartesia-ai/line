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

from line.v02.llm.agent import (
    AgentCallable,
    AgentHandedOff,
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallEnded,
    CallStarted,
    InputEvent,
    OutputEvent,
    SpecificAgentTextSent,
    SpecificInputEvent,
    SpecificUserTextSent,
    ToolEnv,
    TurnEnv,
)
from line.v02.llm.config import LlmConfig
from line.v02.llm.provider import LLMProvider, Message, ToolCall
from line.v02.llm.tool_utils import FunctionTool, ToolType, construct_function_tool
from line.v02.llm.tools import WebSearchTool

T = TypeVar("T")

# Type alias for tools that can be passed to LlmAgent
ToolSpec = Union[FunctionTool, WebSearchTool]


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
            else:
                self._tools.append(tool)

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
                yield output
            return

        # Handle CallStarted
        if isinstance(event, CallStarted):
            if self._config.introduction and not self._introduction_sent:
                self._introduction_sent = True
                yield AgentSendText(text=self._config.introduction)
            return

        # Handle CallEnded
        if isinstance(event, CallEnded):
            return

        async for output in self._generate_response(env, event):
            yield output

    async def _generate_response(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM."""
        messages = self._build_messages(event.history)

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
                        yield AgentSendText(text=chunk.text)

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

                yield AgentToolCalled(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_args=tool_args,
                )

                try:
                    normalized_func = _normalize_to_async_gen(tool.func)

                    if tool.tool_type == ToolType.LOOPBACK:
                        # Collect results to send back to LLM
                        results = []
                        async for value in normalized_func(ctx, **tool_args):
                            results.append(value)
                            yield AgentToolReturned(
                                tool_call_id=tc.id,
                                tool_name=tc.name,
                                tool_args=tool_args,
                                result=value,
                            )

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
                            yield evt
                        yield AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )

                    elif tool.tool_type == ToolType.HANDOFF:
                        # AgentHandedOff input event is passed to the handoff target to execute the tool
                        async for item in normalized_func(ctx, **tool_args, event=AgentHandedOff()):
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

                        yield AgentToolReturned(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            tool_args=tool_args,
                            result="success",
                        )

                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    yield AgentToolReturned(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                        result=f"error: {e}",
                    )

            if not should_continue:
                break

    def _build_messages(self, history: List[SpecificInputEvent]) -> List[Message]:
        """Build LLM messages from conversation history."""
        messages = []
        for event in history:
            if isinstance(event, SpecificUserTextSent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, SpecificAgentTextSent):
                messages.append(Message(role="assistant", content=event.content))
        return messages

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._handoff_target = None
        await self._llm.aclose()
