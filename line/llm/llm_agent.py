"""
LlmAgent - An Agent implementation that wraps LLM providers.

This module provides the LlmAgent class that abstracts over different LLM
providers (OpenAI, Anthropic, Google) and provides a unified interface
for tool calling with three paradigms: loopback, passthrough, and handoff.

Example:
    ```python
    from line.llm import LlmAgent, LlmConfig, function_tool, Field, passthrough_tool
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
        yield AgentOutput(text=message)
        yield AgentEndCall()

    agent = LlmAgent(
        model="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        tools=[get_weather, end_call],
        config=LlmConfig(
            system_prompt="You are a helpful weather assistant.",
            introduction="Hello! I can help you check the weather.",
            temperature=0.7,
        ),
    )
    ```
"""

import json
from typing import Any, AsyncIterable, Dict, List, Optional

from loguru import logger

from line.llm.agent import (
    AgentEndCall,
    AgentHandoff,
    AgentOutput,
    CallStarted,
    InputEvent,
    OutputEvent,
    ToolCallOutput,
    ToolResultOutput,
    TurnContext,
    UserTranscript,
)
from line.llm.config import LlmConfig, detect_provider, resolve_model_alias
from line.llm.function_tool import FunctionTool, ToolType
from line.llm.providers.base import LLM, Message, ToolCall
from line.llm.tool_context import ToolContext, ToolResult


class LlmAgent:
    """
    An Agent that wraps LLM providers for voice agents.

    LlmAgent provides:
    1. Abstraction over different LLM providers (OpenAI, Anthropic, Google)
    2. Automatic context management (conversation history, tool calls)
    3. Three tool calling paradigms (loopback, passthrough, handoff)
    4. Seamless integration with the Cartesia Agent Harness

    Example:
        ```python
        agent = LlmAgent(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            tools=[get_weather, end_call],
            config=LlmConfig(
                system_prompt="You are a helpful assistant.",
                introduction="Hello!",
                temperature=0.7,
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
            model: Model identifier (e.g., "gemini-2.0-flash", "gpt-4o", "claude-3.5-sonnet").
            api_key: API key for the provider.
            tools: List of FunctionTool instances for the agent to use.
            config: LLM configuration (system prompt, temperature, etc.).
            provider: Provider name ("openai", "anthropic", "google"). Auto-detected if not provided.
            max_tool_iterations: Safety limit for loopback tool cycles per turn. Loopback tools
                feed results back to the LLM, which may trigger more tool calls. This prevents
                infinite loops and runaway API costs. Default 10.
        """
        self._model = resolve_model_alias(model)
        self._api_key = api_key
        self._tools = tools or []
        self._config = config or LlmConfig()
        self._provider = detect_provider(self._model)
        self._max_tool_iterations = max_tool_iterations

        # Build tool lookup
        self._tool_map: Dict[str, FunctionTool] = {t.name: t for t in self._tools}

        # Create the LLM instance
        self._llm = self._create_llm()

        # Track if first message has been sent
        self._introduction_sent = False

        logger.info(
            f"LlmAgent initialized with model={self._model}, provider={self._provider}, "
            f"tools={[t.name for t in self._tools]}"
        )

    def _create_llm(self) -> LLM:
        """Create the appropriate LLM provider instance."""
        if self._provider == "openai":
            from line.llm.providers.openai import OpenAI

            return OpenAI(model=self._model, api_key=self._api_key, config=self._config)
        elif self._provider == "anthropic":
            from line.llm.providers.anthropic import Anthropic

            return Anthropic(model=self._model, api_key=self._api_key, config=self._config)
        elif self._provider == "google":
            from line.llm.providers.google import Google

            return Google(model=self._model, api_key=self._api_key, config=self._config)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._provider

    @property
    def tools(self) -> List[FunctionTool]:
        """Get the list of tools."""
        return self._tools

    @property
    def config(self) -> LlmConfig:
        """Get the LLM configuration."""
        return self._config

    async def process(
        self, ctx: TurnContext, event: InputEvent
    ) -> AsyncIterable[OutputEvent]:
        """
        Process an input event and yield output events.

        This is the main entry point called by the Cartesia Agent Harness.
        It handles:
        - First message on CallStarted (if configured)
        - LLM generation on UserTranscript
        - Tool execution with loopback/passthrough/handoff handling

        Args:
            ctx: The turn context with conversation state.
            event: The input event to process.

        Yields:
            Output events (AgentOutput, AgentEndCall, ToolCallOutput, etc.)
        """
        # Handle CallStarted - send first message if configured
        if isinstance(event, CallStarted):
            if self._config.introduction and not self._introduction_sent:
                self._introduction_sent = True
                yield AgentOutput(text=self._config.introduction)
            return

        # Handle UserTranscript - generate LLM response
        if isinstance(event, UserTranscript):
            async for output in self._generate_response(ctx, event.text):
                yield output
            return

        # Other events can be handled here as needed
        logger.debug(f"LlmAgent received unhandled event: {type(event).__name__}")

    async def _generate_response(
        self, ctx: TurnContext, user_text: str
    ) -> AsyncIterable[OutputEvent]:
        """
        Generate a response to user input using the LLM.

        Args:
            ctx: The turn context.
            user_text: The user's transcribed text.

        Yields:
            Output events from LLM generation and tool execution.
        """
        # Build messages from conversation history
        messages = self._build_messages(ctx, user_text)

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
                        yield AgentOutput(text=chunk.text)

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
                yield ToolCallOutput(
                    tool_name=tc.name,
                    tool_args=json.loads(tc.arguments) if tc.arguments else {},
                    tool_call_id=tc.id,
                )

                # Execute the tool
                result = await self._execute_tool(tool, tc, ctx)

                # Handle based on tool type
                if tool.tool_type == ToolType.LOOPBACK:
                    # Yield result for observability
                    yield ToolResultOutput(
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
                    yield ToolResultOutput(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=result.result,
                        error=result.error,
                    )

                elif tool.tool_type == ToolType.HANDOFF:
                    # Yield handoff event
                    if result.handoff_target:
                        yield AgentHandoff(
                            target_agent=result.handoff_target,
                            reason=f"Handoff from tool {tc.name}",
                        )

                    yield ToolResultOutput(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result=result.result,
                        error=result.error,
                    )

            if not should_continue:
                break

    async def _execute_tool(
        self, tool: FunctionTool, tool_call: ToolCall, ctx: TurnContext
    ) -> ToolResult:
        """
        Execute a tool and return the result.

        Args:
            tool: The FunctionTool to execute.
            tool_call: The tool call request.
            ctx: The turn context.

        Returns:
            ToolResult with the execution result.
        """
        try:
            # Parse arguments
            args = json.loads(tool_call.arguments) if tool_call.arguments else {}

            # Create tool context
            tool_ctx = ToolContext(
                conversation_history=ctx.conversation_history.copy(),
                metadata=ctx.metadata,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
            )

            # Execute based on tool type
            if tool.tool_type == ToolType.PASSTHROUGH:
                # Passthrough tools are async generators
                events = []
                async for evt in tool.func(tool_ctx, **args):
                    events.append(evt)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    events=events,
                )
            elif tool.tool_type == ToolType.HANDOFF:
                # Handoff tools return an agent/handler
                result = await tool(tool_ctx, **args)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    handoff_target=result,
                )
            else:
                # Loopback tools return a value
                result = await tool(tool_ctx, **args)
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

    def _build_messages(self, ctx: TurnContext, current_user_text: str) -> List[Message]:
        """
        Build LLM messages from conversation history.

        Args:
            ctx: The turn context with conversation history.
            current_user_text: The current user's text to append.

        Returns:
            List of Message objects for the LLM.
        """
        messages = []

        # Convert history to messages
        for event in ctx.conversation_history:
            if isinstance(event, UserTranscript):
                messages.append(Message(role="user", content=event.text))
            elif isinstance(event, AgentOutput):
                messages.append(Message(role="assistant", content=event.text))
            elif isinstance(event, ToolResultOutput):
                if event.tool_call_id:
                    content = str(event.result) if event.result else str(event.error)
                    messages.append(
                        Message(
                            role="tool",
                            content=content,
                            tool_call_id=event.tool_call_id,
                            name=event.tool_name,
                        )
                    )

        # Add current user message
        messages.append(Message(role="user", content=current_user_text))

        return messages

    async def cleanup(self) -> None:
        """Clean up resources (close LLM client)."""
        await self._llm.aclose()
