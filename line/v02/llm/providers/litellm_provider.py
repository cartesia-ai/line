"""
LiteLLM-based LLM provider.

This module provides a unified LLM implementation using LiteLLM, which abstracts
over multiple providers (OpenAI, Anthropic, Google, etc.) with a single interface.

LiteLLM handles:
- Provider-specific API differences
- Retry logic and fallbacks
- Streaming responses
- Tool/function calling

Model naming convention:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-opus-20240229"
- Google: "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"

Example:
    ```python
    from line.v02.llm.providers.litellm_provider import LiteLLMProvider

    llm = LiteLLMProvider(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        num_retries=3,
        fallbacks=["anthropic/claude-3-5-sonnet-20241022"],
    )

    async with llm.chat(messages, tools) as stream:
        async for chunk in stream:
            print(chunk.text or "", end="")
    ```
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from line.v02.llm.config import LlmConfig
from line.v02.llm.function_tool import FunctionTool
from line.v02.llm.providers.base import LLM, LLMStream, Message, StreamChunk, ToolCall, UsageInfo
from line.v02.llm.schema_converter import function_tools_to_openai


@dataclass
class LiteLLMConfig:
    """
    LiteLLM-specific configuration.

    Attributes:
        num_retries: Number of retries for failed requests.
        fallbacks: List of fallback models to try if primary fails.
        timeout: Request timeout in seconds.
        api_base: Custom API base URL (for proxies or self-hosted).
    """

    num_retries: int = 2
    fallbacks: Optional[List[str]] = None
    timeout: Optional[float] = None
    api_base: Optional[str] = None


class LiteLLMStream(LLMStream):
    """
    LLMStream implementation using LiteLLM's async streaming.
    """

    def __init__(
        self,
        llm: "LiteLLMProvider",
        messages: List[Message],
        tools: List[FunctionTool],
        litellm_kwargs: Dict[str, Any],
    ):
        super().__init__(llm, messages, tools)
        self._litellm_kwargs = litellm_kwargs
        self._response = None

    async def __aenter__(self) -> "LiteLLMStream":
        """Start the streaming request."""
        try:
            from litellm import acompletion
        except ImportError as e:
            raise ImportError("litellm is required. Install with: pip install 'cartesia-line[llm]'") from e

        self._response = await acompletion(**self._litellm_kwargs)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up the stream."""
        # LiteLLM handles cleanup automatically
        pass

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over streaming chunks."""
        if self._response is None:
            raise RuntimeError("Stream not started. Use 'async with' context manager.")

        # Track tool calls being built up
        current_tool_calls: Dict[int, ToolCall] = {}

        async for chunk in self._response:
            # Extract text content
            text = None
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)

                # Handle tool calls
                tool_calls_delta = getattr(delta, "tool_calls", None)
                if tool_calls_delta:
                    for tc_delta in tool_calls_delta:
                        idx = tc_delta.index
                        if idx not in current_tool_calls:
                            # New tool call
                            current_tool_calls[idx] = ToolCall(
                                id=tc_delta.id or "",
                                name=tc_delta.function.name if tc_delta.function else "",
                                arguments="",
                                is_complete=False,
                            )
                        else:
                            # Update existing - accumulate arguments
                            if tc_delta.id:
                                current_tool_calls[idx].id = tc_delta.id
                            if tc_delta.function and tc_delta.function.name:
                                current_tool_calls[idx].name = tc_delta.function.name

                        # Append arguments
                        if tc_delta.function and tc_delta.function.arguments:
                            current_tool_calls[idx].arguments += tc_delta.function.arguments

            # Check for finish reason to mark tool calls complete
            finish_reason = None
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason in ("tool_calls", "stop"):
                    for tc in current_tool_calls.values():
                        tc.is_complete = True

            # Extract usage info (usually in final chunk)
            usage = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage = UsageInfo(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )

            # Yield chunk
            yield StreamChunk(
                text=text,
                tool_calls=list(current_tool_calls.values()) if current_tool_calls else [],
                is_final=finish_reason is not None,
                raw=chunk,
                usage=usage,
            )


class LiteLLMProvider(LLM):
    """
    LLM provider using LiteLLM for unified multi-provider access.

    Supports all providers that LiteLLM supports (100+), including:
    - OpenAI (gpt-4o, gpt-4o-mini, o1, etc.)
    - Anthropic (claude-3-5-sonnet, claude-3-opus, etc.)
    - Google (gemini-2.0-flash, gemini-1.5-pro, etc.)
    - And many more...

    Example:
        ```python
        llm = LiteLLMProvider(
            model="gpt-4o",
            config=LlmConfig(system_prompt="You are helpful."),
            num_retries=3,
            fallbacks=["anthropic/claude-3-5-sonnet-20241022"],
        )
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        num_retries: int = 2,
        fallbacks: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize the LiteLLM provider.

        Args:
            model: Model identifier (e.g., "gpt-4o", "anthropic/claude-3-5-sonnet-20241022").
            api_key: API key for the provider. Can also be set via environment variables.
            config: LLM configuration (system prompt, temperature, etc.).
            num_retries: Number of retries for failed requests (default 2).
            fallbacks: List of fallback models to try if primary fails.
            timeout: Request timeout in seconds.
            api_base: Custom API base URL.
        """
        super().__init__(model, api_key, config)
        self._num_retries = num_retries
        self._fallbacks = fallbacks
        self._timeout = timeout
        self._api_base = api_base

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> LiteLLMStream:
        """
        Start a streaming chat completion.

        Args:
            messages: The conversation messages.
            tools: Optional tools available to the LLM.
            **kwargs: Additional arguments passed to LiteLLM.

        Returns:
            A LiteLLMStream for the response.
        """
        # Convert messages to LiteLLM format (OpenAI-compatible)
        litellm_messages = self._convert_messages(messages)

        # Build kwargs for LiteLLM
        litellm_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": litellm_messages,
            "stream": True,
            "num_retries": self._num_retries,
        }

        # Add API key if provided
        if self._api_key:
            litellm_kwargs["api_key"] = self._api_key

        # Add optional parameters
        if self._fallbacks:
            litellm_kwargs["fallbacks"] = self._fallbacks
        if self._timeout:
            litellm_kwargs["timeout"] = self._timeout
        if self._api_base:
            litellm_kwargs["api_base"] = self._api_base

        # Add config parameters
        if self._config:
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
            # Pass through extra parameters
            if self._config.extra:
                litellm_kwargs.update(self._config.extra)

        # Add tools if provided
        if tools:
            litellm_kwargs["tools"] = function_tools_to_openai(tools, strict=False)

        # Merge any additional kwargs
        litellm_kwargs.update(kwargs)

        return LiteLLMStream(self, messages, tools or [], litellm_kwargs)

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal Message objects to LiteLLM format."""
        litellm_messages = []

        # Add system prompt if configured
        if self._config and self._config.system_prompt:
            litellm_messages.append(
                {
                    "role": "system",
                    "content": self._config.system_prompt,
                }
            )

        for msg in messages:
            litellm_msg: Dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                litellm_msg["content"] = msg.content

            # Handle tool calls in assistant messages
            if msg.tool_calls:
                litellm_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Handle tool response messages
            if msg.role == "tool":
                litellm_msg["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    litellm_msg["name"] = msg.name

            litellm_messages.append(litellm_msg)

        return litellm_messages

    async def aclose(self) -> None:
        """Close the provider (no-op for LiteLLM)."""
        # LiteLLM manages its own connection pooling
        pass
