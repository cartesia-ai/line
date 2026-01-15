"""
Base LLM provider interface.

This module defines the abstract base classes for LLM providers and streams.
Concrete implementations for OpenAI, Anthropic, and Google extend these classes.

Streaming and Tool Calls
------------------------

LLM responses are streamed for low-latency voice applications. This means tool calls
arrive incrementally as the model generates them:

    ```
    # Example: Model calling get_weather(city="San Francisco")

    Chunk 1: ToolCall(id="call_1", name="get_weather", arguments="", is_complete=False)
    Chunk 2: ToolCall(id="call_1", name="get_weather", arguments='{"ci', is_complete=False)
    Chunk 3: ToolCall(id="call_1", name="get_weather", arguments='{"city":', is_complete=False)
    Chunk 4: ToolCall(id="call_1", name="get_weather", arguments='{"city": "San', is_complete=False)
    Chunk 5: ToolCall(id="call_1", name="get_weather", arguments='{"city": "San Francisco"}', is_complete=True)
    ```

The `is_complete` flag indicates when the tool call arguments are fully received and
can be parsed as valid JSON. Consumers should:

1. Accumulate tool calls by `id` as chunks arrive
2. Only execute tools once `is_complete=True`
3. Handle multiple concurrent tool calls (each with unique `id`)

Example consumption:
    ```python
    tool_calls = {}  # id -> ToolCall

    async for chunk in stream:
        for tc in chunk.tool_calls:
            if tc.id in tool_calls:
                # Update existing - append arguments
                tool_calls[tc.id].arguments += tc.arguments
                tool_calls[tc.id].is_complete = tc.is_complete
            else:
                # New tool call
                tool_calls[tc.id] = tc

        # Execute completed tool calls
        for tc in tool_calls.values():
            if tc.is_complete:
                args = json.loads(tc.arguments)
                result = await execute_tool(tc.name, args)
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from line.llm.config import LlmConfig
from line.llm.function_tool import FunctionTool


@dataclass
class StreamChunk:
    """
    A chunk from an LLM stream.

    Attributes:
        text: Text content (may be partial).
        tool_calls: List of tool call requests (may be partial, check is_complete).
        is_final: Whether this is the final chunk.
        raw: Raw response from the provider.
        usage: Token usage information (usually only in final chunk).
    """

    text: Optional[str] = None
    tool_calls: List["ToolCall"] = field(default_factory=list)
    is_final: bool = False
    raw: Any = None
    usage: Optional["UsageInfo"] = None


@dataclass
class ToolCall:
    """
    A tool/function call from the LLM.

    During streaming, tool calls arrive incrementally. The same tool call (identified
    by `id`) may appear in multiple chunks with progressively more complete `arguments`.
    Only parse and execute the tool when `is_complete=True`.

    Attributes:
        id: Unique identifier for this tool call. Use this to track the same
            tool call across multiple streaming chunks.
        name: Name of the tool/function to call.
        arguments: JSON string of arguments. During streaming, this builds up
            incrementally and may be invalid JSON until is_complete=True.
        is_complete: Whether the arguments are fully received. Only parse
            `arguments` as JSON when this is True.

    Example:
        ```python
        # Streaming chunks for get_weather(city="NYC")
        ToolCall(id="call_1", name="get_weather", arguments="", is_complete=False)
        ToolCall(id="call_1", name="get_weather", arguments='{"city":', is_complete=False)
        ToolCall(id="call_1", name="get_weather", arguments='{"city": "NYC"}', is_complete=True)

        # Safe to parse now
        if tool_call.is_complete:
            args = json.loads(tool_call.arguments)
        ```
    """

    id: str
    name: str
    arguments: str = ""
    is_complete: bool = False


@dataclass
class UsageInfo:
    """
    Token usage information.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    """
    A message in the conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant", "system", "tool").
        content: The message content.
        tool_calls: Tool calls made in this message (for assistant messages).
        tool_call_id: ID of the tool call this message responds to (for tool messages).
        name: Name of the tool (for tool messages).
    """

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class LLMStream(ABC):
    """
    Abstract base class for LLM streaming responses.

    Provides async iteration over response chunks and utilities for
    collecting the full response. Tool calls arrive incrementally during
    streaming - see module docstring for details on handling them.

    Example:
        ```python
        async with llm.chat(messages, tools) as stream:
            async for chunk in stream:
                # Handle text
                if chunk.text:
                    print(chunk.text, end="")

                # Handle tool calls (accumulate by id, execute when complete)
                for tc in chunk.tool_calls:
                    if tc.is_complete:
                        result = await execute_tool(tc.name, json.loads(tc.arguments))
        ```
    """

    def __init__(self, llm: "LLM", messages: List[Message], tools: List[FunctionTool]):
        """
        Initialize the stream.

        Args:
            llm: The LLM instance.
            messages: The messages sent to the LLM.
            tools: The tools available to the LLM.
        """
        self._llm = llm
        self._messages = messages
        self._tools = tools
        self._collected_text = ""
        self._collected_tool_calls: List[ToolCall] = []
        self._usage: Optional[UsageInfo] = None

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over stream chunks."""
        pass

    async def collect(self) -> StreamChunk:
        """
        Collect all chunks and return the final result.

        This is useful when you don't need streaming and want to wait
        for the complete response.

        Returns:
            A StreamChunk with the complete response.
        """
        async for chunk in self:
            if chunk.text:
                self._collected_text += chunk.text
            if chunk.tool_calls:
                self._update_tool_calls(chunk.tool_calls)
            if chunk.usage:
                self._usage = chunk.usage

        return StreamChunk(
            text=self._collected_text if self._collected_text else None,
            tool_calls=self._collected_tool_calls,
            is_final=True,
            usage=self._usage,
        )

    def _update_tool_calls(self, new_calls: List[ToolCall]) -> None:
        """Update collected tool calls with new chunks."""
        for new_call in new_calls:
            # Find existing call with same ID
            existing = next((c for c in self._collected_tool_calls if c.id == new_call.id), None)
            if existing:
                existing.arguments += new_call.arguments
                existing.is_complete = new_call.is_complete
            else:
                self._collected_tool_calls.append(new_call)

    async def to_text(self) -> str:
        """
        Collect all text content from the stream.

        Returns:
            The complete text response.
        """
        result = await self.collect()
        return result.text or ""

    async def __aenter__(self) -> "LLMStream":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass


class LLM(ABC):
    """
    Abstract base class for LLM providers.

    Provides a unified interface for interacting with different LLM providers.
    Subclasses implement provider-specific logic.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
    ):
        """
        Initialize the LLM.

        Args:
            model: The model identifier.
            api_key: API key for authentication.
            config: LLM configuration.
        """
        self._model = model
        self._api_key = api_key
        self._config = config or LlmConfig()

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def config(self) -> LlmConfig:
        """Get the LLM configuration."""
        return self._config

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> LLMStream:
        """
        Start a chat completion.

        Args:
            messages: The conversation messages.
            tools: Optional tools available to the LLM.
            **kwargs: Additional provider-specific arguments.

        Returns:
            An LLMStream for the response.
        """
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """Close the LLM client and release resources."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model!r})"
