"""
LlmAgent - An Agent implementation wrapping 100+ LLM providers via LiteLLM.

See README.md for examples and documentation.
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Dict, List, Optional

from loguru import logger

from line.v02.llm.agent import (
    Agent,
    AgentHandoff,
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    InputEvent,
    OutputEvent,
    SpecificAgentTextSent,
    SpecificInputEvent,
    SpecificUserTextSent,
    TurnEnv,
    UserTextSent,
    UserTurnEnded,
)
from line.v02.llm.config import LlmConfig
from line.v02.llm.function_tool import FunctionTool, ToolType
from line.v02.llm.provider import LLMProvider, Message, ToolCall


@dataclass
class _ToolResult:
    """Internal result from executing a tool."""

    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    events: List[Any] = field(default_factory=list)
    handoff_target: Any = None

    @property
    def success(self) -> bool:
        return self.error is None


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

        # Build tool lookup
        self._tool_map: Dict[str, FunctionTool] = {t.name: t for t in self._tools}

        # Create the LLM instance
        self._llm = self._create_llm()

        # Track state
        self._introduction_sent = False
        self._handoff_target: Optional[Agent] = None

        logger.info(f"LlmAgent initialized with model={self._model}, tools={[t.name for t in self._tools]}")

    def _create_llm(self) -> LLMProvider:
        """Create the LiteLLM provider instance."""
        return LLMProvider(
            model=self._model,
            api_key=self._api_key,
            config=self._config,
            num_retries=self._config.num_retries,
            fallbacks=self._config.fallbacks,
            timeout=self._config.timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def tools(self) -> List[FunctionTool]:
        return self._tools

    @property
    def config(self) -> LlmConfig:
        return self._config

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Process an input event and yield output events."""
        # If handoff is active, delegate
        if self._handoff_target is not None:
            async for output in self._delegate_to_handoff(env, event):
                yield output
            return

        # Handle CallStarted
        if isinstance(event, CallStarted):
            if self._config.introduction and not self._introduction_sent:
                self._introduction_sent = True
                yield AgentSendText(text=self._config.introduction)
            return

        # Handle UserTextSent
        if isinstance(event, UserTextSent):
            async for output in self._generate_response(env, event.history, event.content):
                yield output
            return

        # Handle UserTurnEnded
        if isinstance(event, UserTurnEnded):
            user_text = ""
            for content_item in event.content:
                if isinstance(content_item, SpecificUserTextSent):
                    user_text += content_item.content
            if user_text:
                async for output in self._generate_response(env, event.history, user_text):
                    yield output
            return

        logger.debug(f"LlmAgent received unhandled event: {type(event).__name__}")

    async def _delegate_to_handoff(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """Delegate processing to the handoff target agent."""
        if self._handoff_target is None:
            return

        if hasattr(self._handoff_target, "process"):
            async for output in self._handoff_target.process(env, event):
                yield output
        elif callable(self._handoff_target):
            async for output in self._handoff_target(env, event):
                yield output
        else:
            logger.error(f"Invalid handoff target: {type(self._handoff_target)}")

    async def _generate_response(
        self, env: TurnEnv, history: List[SpecificInputEvent], user_text: str
    ) -> AsyncIterable[OutputEvent]:
        """Generate a response using the LLM."""
        messages = self._build_messages(history, user_text)

        iteration = 0
        while iteration < self._max_tool_iterations:
            iteration += 1

            text_buffer = ""
            tool_calls: List[ToolCall] = []

            stream = self._llm.chat(messages, self._tools if self._tools else None)
            async with stream:
                async for chunk in stream:
                    if chunk.text:
                        text_buffer += chunk.text
                        yield AgentSendText(text=chunk.text)

                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            existing = next((t for t in tool_calls if t.id == tc.id), None)
                            if existing:
                                existing.arguments += tc.arguments
                                existing.is_complete = tc.is_complete
                            else:
                                tool_calls.append(tc)

            if not tool_calls:
                break

            should_continue = False
            for tc in tool_calls:
                if not tc.is_complete:
                    continue

                tool = self._tool_map.get(tc.name)
                if not tool:
                    logger.warning(f"Unknown tool: {tc.name}")
                    continue

                tool_args = json.loads(tc.arguments) if tc.arguments else {}

                # Emit tool called event
                yield AgentToolCalled(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_args=tool_args,
                )

                # Execute the tool
                result = await self._execute_tool(tool, tc, tool_args, history, env)

                if tool.tool_type == ToolType.LOOPBACK:
                    # Emit result event
                    yield AgentToolReturned(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                        result=result.result if result.success else result.error,
                    )

                    # Add to messages for next iteration
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
                            content=str(result.result) if result.result else result.error,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

                    should_continue = True
                    text_buffer = ""

                elif tool.tool_type == ToolType.PASSTHROUGH:
                    for evt in result.events:
                        yield evt

                    yield AgentToolReturned(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                        result=result.result if result.success else result.error,
                    )

                elif tool.tool_type == ToolType.HANDOFF:
                    for evt in result.events:
                        yield evt

                    if result.handoff_target:
                        self._handoff_target = result.handoff_target
                        target_name = (
                            getattr(result.handoff_target, "__name__", None)
                            or type(result.handoff_target).__name__
                        )
                        yield AgentHandoff(
                            target_agent=target_name,
                            reason=f"Handoff from tool {tc.name}",
                        )

                    yield AgentToolReturned(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        tool_args=tool_args,
                        result=result.result if result.success else result.error,
                    )

            if not should_continue:
                break

    async def _execute_tool(
        self,
        tool: FunctionTool,
        tool_call: ToolCall,
        tool_args: Dict[str, Any],
        history: List[SpecificInputEvent],
        env: TurnEnv,
    ) -> _ToolResult:
        """Execute a tool and return the result."""
        try:
            # Build context dict for tool
            ctx = {
                "conversation_history": list(history),
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "turn_env": env,
                "config": self._config,
            }

            if tool.tool_type == ToolType.PASSTHROUGH:
                events = []
                async for evt in tool.func(ctx, **tool_args):
                    events.append(evt)
                return _ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_args=tool_args,
                    events=events,
                )

            elif tool.tool_type == ToolType.HANDOFF:
                events = []
                handoff_target = None
                async for evt in tool.func(ctx, **tool_args):
                    if self._is_agent(evt):
                        handoff_target = evt
                    else:
                        events.append(evt)
                return _ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_args=tool_args,
                    events=events,
                    handoff_target=handoff_target,
                )

            else:
                raw_result = tool.func(ctx, **tool_args)
                result = await self._resolve_loopback_result(raw_result)
                return _ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_args=tool_args,
                    result=result,
                )

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return _ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                tool_args=tool_args,
                error=str(e),
            )

    def _is_agent(self, obj: Any) -> bool:
        """Check if an object is an Agent."""
        if hasattr(obj, "process") and callable(obj.process):
            return True
        if callable(obj) and not isinstance(obj, type):
            try:
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                return len(params) >= 2
            except (ValueError, TypeError):
                pass
        return False

    async def _resolve_loopback_result(self, raw_result: Any) -> Any:
        """Resolve a loopback tool result to a final value."""
        if inspect.iscoroutine(raw_result):
            return await raw_result

        if inspect.isawaitable(raw_result):
            return await raw_result

        if hasattr(raw_result, "__aiter__"):
            results = []
            async for item in raw_result:
                results.append(item)
            return results[0] if len(results) == 1 else results

        return raw_result

    def _build_messages(self, history: List[SpecificInputEvent], current_user_text: str) -> List[Message]:
        """Build LLM messages from conversation history."""
        messages = []

        for event in history:
            if isinstance(event, SpecificUserTextSent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, SpecificAgentTextSent):
                messages.append(Message(role="assistant", content=event.content))

        messages.append(Message(role="user", content=current_user_text))

        return messages

    def reset_handoff(self) -> None:
        """Reset the handoff state."""
        self._handoff_target = None

    @property
    def handoff_target(self) -> Optional[Agent]:
        """Get the current handoff target, if any."""
        return self._handoff_target

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._handoff_target = None
        await self._llm.aclose()
