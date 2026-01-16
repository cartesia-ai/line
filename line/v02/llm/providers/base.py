"""
Base LLM provider interface.

Abstract base classes for LLM providers and streams.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional

from line.v02.llm.config import LlmConfig
from line.v02.llm.function_tool import FunctionTool


@dataclass
class StreamChunk:
    """A chunk from an LLM stream."""

    text: Optional[str] = None
    tool_calls: List["ToolCall"] = field(default_factory=list)
    is_final: bool = False
    raw: Any = None
    usage: Optional["UsageInfo"] = None


@dataclass
class ToolCall:
    """
    A tool/function call from the LLM.

    During streaming, tool calls arrive incrementally. Only parse arguments
    when is_complete=True.
    """

    id: str
    name: str
    arguments: str = ""
    is_complete: bool = False


@dataclass
class UsageInfo:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    """A message in the conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class LLMStream(ABC):
    """Abstract base class for LLM streaming responses."""

    def __init__(self, llm: "LLM", messages: List[Message], tools: List[FunctionTool]):
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
        """Collect all chunks and return the final result."""
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
            existing = next((c for c in self._collected_tool_calls if c.id == new_call.id), None)
            if existing:
                existing.arguments += new_call.arguments
                existing.is_complete = new_call.is_complete
            else:
                self._collected_tool_calls.append(new_call)

    async def to_text(self) -> str:
        """Collect all text content from the stream."""
        result = await self.collect()
        return result.text or ""

    async def __aenter__(self) -> "LLMStream":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: B027
        pass


class LLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._config = config or LlmConfig()

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> LlmConfig:
        return self._config

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> LLMStream:
        """Start a streaming chat completion."""
        pass

    @abstractmethod
    async def aclose(self) -> None:
        """Close the LLM client."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model!r})"
