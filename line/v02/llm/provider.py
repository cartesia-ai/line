"""
LLM Provider using LiteLLM.

Provides a unified interface to 100+ LLM providers via LiteLLM.
See https://docs.litellm.ai/docs/providers for supported providers.

Model naming:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-3-5-sonnet-20241022"
- Google: "gemini/gemini-2.0-flash"
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from line.v02.llm.config import LlmConfig
from line.v02.llm.function_tool import FunctionTool
from line.v02.llm.schema_converter import function_tools_to_openai


@dataclass
class ToolCall:
    """A tool/function call from the LLM."""

    id: str
    name: str
    arguments: str = ""
    is_complete: bool = False


@dataclass
class StreamChunk:
    """A chunk from an LLM stream."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_final: bool = False


@dataclass
class Message:
    """A message in the conversation."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class LLMProvider:
    """
    LLM provider using LiteLLM for unified multi-provider access.

    Handles streaming responses and tool calls for all LiteLLM-supported models.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        num_retries: int = 2,
        fallbacks: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._config = config or LlmConfig()
        self._num_retries = num_retries
        self._fallbacks = fallbacks
        self._timeout = timeout

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> "_ChatStream":
        """Start a streaming chat completion."""
        litellm_messages = self._build_messages(messages)

        litellm_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": litellm_messages,
            "stream": True,
            "num_retries": self._num_retries,
        }

        if self._api_key:
            litellm_kwargs["api_key"] = self._api_key
        if self._fallbacks:
            litellm_kwargs["fallbacks"] = self._fallbacks
        if self._timeout:
            litellm_kwargs["timeout"] = self._timeout

        # Add config parameters
        if self._config.temperature is not None:
            litellm_kwargs["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            litellm_kwargs["max_tokens"] = self._config.max_tokens
        if self._config.top_p is not None:
            litellm_kwargs["top_p"] = self._config.top_p
        if self._config.stop:
            litellm_kwargs["stop"] = self._config.stop
        if self._config.seed is not None:
            litellm_kwargs["seed"] = self._config.seed
        if self._config.presence_penalty is not None:
            litellm_kwargs["presence_penalty"] = self._config.presence_penalty
        if self._config.frequency_penalty is not None:
            litellm_kwargs["frequency_penalty"] = self._config.frequency_penalty
        if self._config.extra:
            litellm_kwargs.update(self._config.extra)

        if tools:
            litellm_kwargs["tools"] = function_tools_to_openai(tools, strict=False)

        litellm_kwargs.update(kwargs)

        return _ChatStream(litellm_kwargs)

    def _build_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to LiteLLM format."""
        result = []

        if self._config.system_prompt:
            result.append({"role": "system", "content": self._config.system_prompt})

        for msg in messages:
            litellm_msg: Dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                litellm_msg["content"] = msg.content

            if msg.tool_calls:
                litellm_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]

            if msg.role == "tool":
                litellm_msg["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    litellm_msg["name"] = msg.name

            result.append(litellm_msg)

        return result

    async def aclose(self) -> None:
        """Close the provider (no-op for LiteLLM)."""
        pass


class _ChatStream:
    """Async context manager for streaming chat responses."""

    def __init__(self, litellm_kwargs: Dict[str, Any]):
        self._kwargs = litellm_kwargs
        self._response = None

    async def __aenter__(self) -> "_ChatStream":
        try:
            from litellm import acompletion
        except ImportError as e:
            raise ImportError("litellm required. Install: pip install 'cartesia-line[llm]'") from e

        self._response = await acompletion(**self._kwargs)
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        if self._response is None:
            raise RuntimeError("Stream not started. Use 'async with' context manager.")

        tool_calls: Dict[int, ToolCall] = {}

        async for chunk in self._response:
            text = None
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)

                # Handle incremental tool calls
                tc_delta = getattr(delta, "tool_calls", None)
                if tc_delta:
                    for tc in tc_delta:
                        idx = tc.index
                        if idx not in tool_calls:
                            tool_calls[idx] = ToolCall(
                                id=tc.id or "",
                                name=tc.function.name if tc.function else "",
                            )
                        else:
                            if tc.id:
                                tool_calls[idx].id = tc.id
                            if tc.function and tc.function.name:
                                tool_calls[idx].name = tc.function.name

                        if tc.function and tc.function.arguments:
                            # Providers stream tool args differently:
                            # - OpenAI: incremental chunks ("{\"ci", "ty\":", ...") - must concat
                            # - Anthropic: incremental chunks like OpenAI - must concat
                            # - Gemini: complete args repeated each chunk - must dedupe
                            # Detect by checking if existing args look complete (ends with "}")
                            existing = tool_calls[idx].arguments
                            new_args = tc.function.arguments
                            if not existing:
                                tool_calls[idx].arguments = new_args
                            elif not existing.endswith("}"):
                                tool_calls[idx].arguments += new_args  # Incremental
                            # else: complete args, skip duplicate

            # Check finish reason
            finish_reason = None
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason in ("tool_calls", "stop"):
                    for tc in tool_calls.values():
                        tc.is_complete = True

            yield StreamChunk(
                text=text,
                tool_calls=list(tool_calls.values()) if tool_calls else [],
                is_final=finish_reason is not None,
            )
