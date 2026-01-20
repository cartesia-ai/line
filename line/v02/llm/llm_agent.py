"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

import inspect
import json
from typing import Any, AsyncIterable, Callable, Dict, List, Optional, TypeVar

from loguru import logger

from line.v02.llm.agent import (
    AgentCallable,
    AgentHandedOff,
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    InputEvent,
    OutputEvent,
    SpecificAgentTextSent,
    SpecificInputEvent,
    SpecificUserTextSent,
    ToolContext,
    TurnEnv,
)
from line.v02.llm.config import LlmConfig
from line.v02.llm.function_tool import FunctionTool, ToolType
from line.v02.llm.provider import LLMProvider, Message, ToolCall

T = TypeVar("T")


async def _normalize_result(result: Any) -> AsyncIterable[Any]:
    """Normalize any result type to an async iterable.

    Converts: AsyncIterable[T] | Future[T] | T => AsyncIterable[T]
    """
    if inspect.iscoroutine(result) or inspect.isawaitable(result):
        yield await result
    elif hasattr(result, "__aiter__"):
        async for item in result:
            yield item
    else:
        yield result


def _normalize_to_async_gen(func: Callable[..., Any]) -> Callable[..., AsyncIterable[Any]]:
    """Wrap a function to always return an async generator.

    Converts: Callable[Args, AsyncIterable[T] | Future[T] | T] => Callable[Args, AsyncIterable[T]]
    """

    async def wrapper(*args, **kwargs) -> AsyncIterable[Any]:
        result = func(*args, **kwargs)
        async for item in _normalize_result(result):
            yield item

    return wrapper


def _is_handoff_target(obj: Any) -> bool:
    """Check if an object can be a handoff target (has process method or is a callable with 2+ params)."""
    if hasattr(obj, "process") and callable(obj.process):
        return True
    if callable(obj) and not isinstance(obj, type):
        try:
            sig = inspect.signature(obj)
            return len(sig.parameters) >= 2
        except (ValueError, TypeError):
            pass
    return False


def _normalize_to_callable(target: Any) -> AgentCallable:
    """Normalize a handoff target to a callable process function.

    Accepts either an object with .process() method or a callable (env, event) -> AsyncIterable.
    Returns a normalized callable.
    """
    if hasattr(target, "process") and callable(target.process):
        return target.process
    elif callable(target):
        return target
    else:
        raise TypeError(f"Cannot normalize {type(target)} to process function")


class LlmAgent:
    """
    Agent wrapping LLM providers via LiteLLM with tool calling support.

    Supports loopback, passthrough, and handoff tool paradigms.
    See README.md for examples.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        tools: Optional[List[FunctionTool]] = None,
        config: Optional[LlmConfig] = None,
        max_tool_iterations: int = 10,
    ):
        self._model = model
        self._api_key = api_key
        self._tools = tools or []
        self._config = config or LlmConfig()
        self._max_tool_iterations = max_tool_iterations

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

        async for output in self._generate_response(env, event):
            yield output

    async def _generate_response(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM."""
        messages = self._build_messages(event.history)

        for _iteration in range(self._max_tool_iterations):
            text_buffer = ""
            tool_calls_dict: Dict[str, ToolCall] = {}

            stream = self._llm.chat(messages, self._tools if self._tools else None)
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
            ctx = ToolContext(turn_env=env)

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
                        # Execute and filter: yield events, capture handoff target
                        handoff_target = None
                        async for item in normalized_func(ctx, **tool_args):
                            if _is_handoff_target(item):
                                handoff_target = item
                            else:
                                yield item

                        if handoff_target:
                            self._handoff_target = _normalize_to_callable(handoff_target)
                            target_name = (
                                getattr(handoff_target, "__name__", None) or type(handoff_target).__name__
                            )
                            yield AgentHandedOff(target=target_name)

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
