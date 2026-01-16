"""
LlmAgent - An Agent implementation that wraps LLM providers via LiteLLM.

This module provides the LlmAgent class that abstracts over different LLM
providers (OpenAI, Anthropic, Google, and 100+ more) using LiteLLM and provides
a unified interface for tool calling with three paradigms: loopback, passthrough,
and handoff.

Model naming convention (LiteLLM format):
- OpenAI: "gpt-4o", "gpt-4o-mini", "o1"
- Anthropic: "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-opus-20240229"
- Google: "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"

Example:
    ```python
    from line.v02.llm import LlmAgent, LlmConfig, function_tool, Field, passthrough_tool
    from line.v02.llm import AgentSendText, AgentEndCall
    from typing import Annotated

    @function_tool
    async def get_weather(
        ctx: ToolContext,
        city: Annotated[str, Field(description="The city name")]
    ) -> str:
        '''Get the current weather'''
        return f"72°F in {city}"

    @passthrough_tool
    async def end_call(
        ctx: ToolContext,
        message: Annotated[str, Field(description="Goodbye message")]
    ):
        '''End the call'''
        yield AgentSendText(text=message)
        yield AgentEndCall()

    agent = LlmAgent(
        model="gpt-4o",
        tools=[get_weather, end_call],
        config=LlmConfig(
            system_prompt="You are a helpful weather assistant.",
            introduction="Hello! I can help you check the weather.",
            temperature=0.7,
            num_retries=3,
            fallbacks=["anthropic/claude-3-5-sonnet-20241022"],
        ),
    )
    ```
"""

import inspect
import json
from typing import Any, AsyncIterable, Dict, List, Optional

from loguru import logger

from line.v02.llm.agent import (
    Agent,
    AgentHandoff,
    AgentSendText,
    CallStarted,
    InputEvent,
    OutputEvent,
    SpecificAgentTextSent,
    SpecificInputEvent,
    SpecificUserTextSent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnv,
    UserTextSent,
    UserTurnEnded,
)
from line.v02.llm.config import LlmConfig
from line.v02.llm.function_tool import FunctionTool, ToolType
from line.v02.llm.providers.base import LLM, Message, ToolCall
from line.v02.llm.tool_context import ToolContext, ToolResult


class LlmAgent:
    """
    An Agent that wraps LLM providers for voice agents using LiteLLM.

    LlmAgent provides:
    1. Abstraction over 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Google, etc.)
    2. Automatic context management (conversation history, tool calls)
    3. Three tool calling paradigms (loopback, passthrough, handoff)
    4. Built-in retry logic and fallback support
    5. Seamless integration with the Cartesia Agent Harness

    Model naming (LiteLLM format):
    - OpenAI: "gpt-4o", "gpt-4o-mini"
    - Anthropic: "anthropic/claude-3-5-sonnet-20241022"
    - Google: "gemini/gemini-2.0-flash"

    Example:
        ```python
        agent = LlmAgent(
            model="gpt-4o",
            tools=[get_weather, end_call],
            config=LlmConfig(
                system_prompt="You are a helpful assistant.",
                introduction="Hello!",
                temperature=0.7,
                num_retries=3,
                fallbacks=["anthropic/claude-3-5-sonnet-20241022"],
            ),
        )

        # In get_agent
        def get_agent(ctx, call_request):
            return agent
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        tools: Optional[List[FunctionTool]] = None,
        config: Optional[LlmConfig] = None,
        max_tool_iterations: int = 10,
    ):
        """
        Initialize the LlmAgent.

        Args:
            model: Model identifier in LiteLLM format (e.g., "gpt-4o",
                "anthropic/claude-3-5-sonnet-20241022", "gemini/gemini-2.0-flash").
            api_key: API key for the provider. Can also be set via environment variables.
            tools: List of FunctionTool instances for the agent to use.
            config: LLM configuration (system prompt, temperature, retries, fallbacks, etc.).
            max_tool_iterations: Safety limit for loopback tool cycles per turn. Loopback tools
                feed results back to the LLM, which may trigger more tool calls. This prevents
                infinite loops and runaway API costs. Default 10.
        """
        self._model = model
        self._api_key = api_key
        self._tools = tools or []
        self._config = config or LlmConfig()
        self._max_tool_iterations = max_tool_iterations

        # Build tool lookup
        self._tool_map: Dict[str, FunctionTool] = {t.name: t for t in self._tools}

        # Create the LLM instance using LiteLLM
        self._llm = self._create_llm()

        # Track if first message has been sent
        self._introduction_sent = False

        # Handoff state: when set, all subsequent process() calls are delegated
        # to the handoff target agent instead of this LlmAgent
        self._handoff_target: Optional[Agent] = None

        logger.info(f"LlmAgent initialized with model={self._model}, tools={[t.name for t in self._tools]}")

    def _create_llm(self) -> LLM:
        """Create the LiteLLM provider instance."""
        from line.v02.llm.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(
            model=self._model,
            api_key=self._api_key,
            config=self._config,
            num_retries=self._config.num_retries,
            fallbacks=self._config.fallbacks,
            timeout=self._config.timeout,
        )

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def tools(self) -> List[FunctionTool]:
        """Get the list of tools."""
        return self._tools

    @property
    def config(self) -> LlmConfig:
        """Get the LLM configuration."""
        return self._config

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """
        Process an input event and yield output events.

        This is the main entry point called by the Cartesia Agent Harness.
        It handles:
        - Delegation to handoff target if active
        - First message on CallStarted (if configured)
        - LLM generation on UserTextSent or UserTurnEnded
        - Tool execution with loopback/passthrough/handoff handling

        Args:
            env: The turn environment.
            event: The input event to process (includes history).

        Yields:
            Output events (AgentSendText, AgentEndCall, ToolCallEvent, etc.)
        """
        # If handoff is active, delegate all events to the handoff target
        if self._handoff_target is not None:
            async for output in self._delegate_to_handoff(env, event):
                yield output
            return

        # Handle CallStarted - send first message if configured
        if isinstance(event, CallStarted):
            if self._config.introduction and not self._introduction_sent:
                self._introduction_sent = True
                yield AgentSendText(text=self._config.introduction)
            return

        # Handle UserTextSent - generate LLM response
        if isinstance(event, UserTextSent):
            async for output in self._generate_response(event.history, event.content):
                yield output
            return

        # Handle UserTurnEnded - extract text content and generate response
        if isinstance(event, UserTurnEnded):
            # Extract text from the user turn content
            user_text = ""
            for content_item in event.content:
                if isinstance(content_item, SpecificUserTextSent):
                    user_text += content_item.content
            if user_text:
                async for output in self._generate_response(event.history, user_text):
                    yield output
            return

        # Other events can be handled here as needed
        logger.debug(f"LlmAgent received unhandled event: {type(event).__name__}")

    async def _delegate_to_handoff(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        """
        Delegate processing to the handoff target agent.

        Args:
            env: The turn environment.
            event: The input event.

        Yields:
            Output events from the handoff target.
        """
        if self._handoff_target is None:
            return

        # Call the handoff target's process method
        if hasattr(self._handoff_target, "process"):
            # Class-based agent
            async for output in self._handoff_target.process(env, event):
                yield output
        elif callable(self._handoff_target):
            # Function-based agent
            async for output in self._handoff_target(env, event):
                yield output
        else:
            logger.error(f"Invalid handoff target: {type(self._handoff_target)}")

    async def _generate_response(
        self, history: List[SpecificInputEvent], user_text: str
    ) -> AsyncIterable[OutputEvent]:
        """
        Generate a response to user input using the LLM.

        Args:
            history: The conversation history (list of SpecificInputEvent).
            user_text: The user's transcribed text.

        Yields:
            Output events from LLM generation and tool execution.
        """
        # Build messages from conversation history
        messages = self._build_messages(history, user_text)

        # Main generation loop with tool handling
        iteration = 0
        while iteration < self._max_tool_iterations:
            iteration += 1

            # Stream LLM response
            text_buffer = ""
            tool_calls: List[ToolCall] = []

            stream = self._llm.chat(messages, self._tools if self._tools else None)
            async with stream:
                async for chunk in stream:
                    # Yield text as it streams
                    if chunk.text:
                        text_buffer += chunk.text
                        yield AgentSendText(text=chunk.text)

                    # Collect tool calls
                    if chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            existing = next((t for t in tool_calls if t.id == tc.id), None)
                            if existing:
                                existing.arguments += tc.arguments
                                existing.is_complete = tc.is_complete
                            else:
                                tool_calls.append(tc)

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Process tool calls
            should_continue = False
            for tc in tool_calls:
                if not tc.is_complete:
                    continue

                tool = self._tool_map.get(tc.name)
                if not tool:
                    logger.warning(f"Unknown tool: {tc.name}")
                    continue

                # Yield tool call event for observability
                yield ToolCallEvent(
                    tool_name=tc.name,
                    tool_args=json.loads(tc.arguments) if tc.arguments else {},
                    tool_call_id=tc.id,
                )

                # Execute the tool
                result = await self._execute_tool(tool, tc, history)

                # Handle based on tool type
                if tool.tool_type == ToolType.LOOPBACK:
                    # Yield result for observability
                    yield ToolResultEvent(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=result.result,
                        error=result.error,
                    )

                    # Add assistant message with tool call to messages
                    messages.append(
                        Message(
                            role="assistant",
                            content=text_buffer if text_buffer else None,
                            tool_calls=[tc],
                        )
                    )

                    # Add tool result message
                    messages.append(
                        Message(
                            role="tool",
                            content=str(result.result) if result.result else result.error,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

                    should_continue = True
                    text_buffer = ""  # Reset for next iteration

                elif tool.tool_type == ToolType.PASSTHROUGH:
                    # Yield events directly (bypass LLM)
                    for evt in result.events:
                        yield evt

                    # Yield result for observability
                    yield ToolResultEvent(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=result.result,
                        error=result.error,
                    )

                elif tool.tool_type == ToolType.HANDOFF:
                    # Handoff tools yield events (like passthrough) AND set a handoff target.
                    # After this, all subsequent process() calls go to the handoff target.
                    for evt in result.events:
                        yield evt

                    # Set the handoff target for subsequent calls
                    if result.handoff_target:
                        self._handoff_target = result.handoff_target
                        # Emit AgentHandoff event for the bus (target_agent is a string identifier)
                        target_name = (
                            getattr(result.handoff_target, "__name__", None)
                            or type(result.handoff_target).__name__
                        )
                        yield AgentHandoff(
                            target_agent=target_name,
                            reason=f"Handoff from tool {tc.name}",
                        )

                    yield ToolResultEvent(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=result.result,
                        error=result.error,
                    )

            if not should_continue:
                break

    async def _execute_tool(
        self, tool: FunctionTool, tool_call: ToolCall, history: List[SpecificInputEvent]
    ) -> ToolResult:
        """
        Execute a tool and return the result.

        Args:
            tool: The FunctionTool to execute.
            tool_call: The tool call request.
            history: The conversation history.

        Returns:
            ToolResult with the execution result.
        """
        try:
            # Parse arguments
            args = json.loads(tool_call.arguments) if tool_call.arguments else {}

            # Create tool context
            tool_ctx = ToolContext(
                conversation_history=list(history),  # Copy the history
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            )

            # Execute based on tool type
            if tool.tool_type == ToolType.PASSTHROUGH:
                # Passthrough tools are async generators that yield events
                events = []
                async for evt in tool.func(tool_ctx, **args):
                    events.append(evt)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    events=events,
                )

            elif tool.tool_type == ToolType.HANDOFF:
                # Handoff tools are async generators that yield events AND return an agent.
                # The last yielded value that is an Agent becomes the handoff target.
                events = []
                handoff_target = None
                async for evt in tool.func(tool_ctx, **args):
                    # Check if this is an agent (handoff target) or an event
                    if self._is_agent(evt):
                        handoff_target = evt
                    else:
                        events.append(evt)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    events=events,
                    handoff_target=handoff_target,
                )

            else:
                # Loopback tools can return:
                # 1. A bare value (use as-is)
                # 2. A Future/Coroutine (await it)
                # 3. An AsyncIterable (iterate and collect results)
                raw_result = tool.func(tool_ctx, **args)
                result = await self._resolve_loopback_result(raw_result)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                )

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                error=str(e),
            )

    def _is_agent(self, obj: Any) -> bool:
        """Check if an object is an Agent (has process method or is callable)."""
        if hasattr(obj, "process") and callable(obj.process):
            return True
        # Check if it looks like a function-based agent
        if callable(obj) and not isinstance(obj, type):
            # Heuristic: if it's a function with ctx and event params, it's likely an agent
            try:
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                return len(params) >= 2
            except (ValueError, TypeError):
                pass
        return False

    async def _resolve_loopback_result(self, raw_result: Any) -> Any:
        """
        Resolve a loopback tool result to a final value.

        Supports:
        - Bare values (returned as-is)
        - Coroutines/Futures (awaited)
        - AsyncIterables (collected into a list or single value)

        Args:
            raw_result: The raw return value from the tool function.

        Returns:
            The resolved result value.
        """
        # If it's a coroutine, await it
        if inspect.iscoroutine(raw_result):
            return await raw_result

        # If it's an awaitable (Future-like), await it
        if inspect.isawaitable(raw_result):
            return await raw_result

        # If it's an async iterable, collect the results
        if hasattr(raw_result, "__aiter__"):
            results = []
            async for item in raw_result:
                results.append(item)
            # If only one result, return it directly; otherwise return list
            return results[0] if len(results) == 1 else results

        # Otherwise, it's a bare value
        return raw_result

    def _build_messages(self, history: List[SpecificInputEvent], current_user_text: str) -> List[Message]:
        """
        Build LLM messages from conversation history.

        Args:
            history: The conversation history (list of SpecificInputEvent).
            current_user_text: The current user's text to append.

        Returns:
            List of Message objects for the LLM.
        """
        messages = []

        # Convert history to messages
        for event in history:
            if isinstance(event, SpecificUserTextSent):
                messages.append(Message(role="user", content=event.content))
            elif isinstance(event, SpecificAgentTextSent):
                messages.append(Message(role="assistant", content=event.content))
            # Note: Tool results in v02 don't have tool_call_id, so we skip them
            # The LLM wrapper manages tool call/result tracking internally

        # Add current user message
        messages.append(Message(role="user", content=current_user_text))

        return messages

    def reset_handoff(self) -> None:
        """
        Reset the handoff state, returning control to this LlmAgent.

        Call this to clear an active handoff and resume processing
        with the original LlmAgent.
        """
        self._handoff_target = None

    @property
    def handoff_target(self) -> Optional[Agent]:
        """Get the current handoff target, if any."""
        return self._handoff_target

    async def cleanup(self) -> None:
        """Clean up resources (close LLM client, reset state)."""
        self._handoff_target = None
        await self._llm.aclose()
