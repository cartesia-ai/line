"""
OpenAI LLM provider implementation.

This module provides the OpenAI implementation of the LLM interface,
supporting both the Chat Completions API and Responses API.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from line.llm.config import LlmConfig
from line.llm.function_tool import FunctionTool
from line.llm.providers.base import LLM, LLMStream, Message, StreamChunk, ToolCall, UsageInfo
from line.llm.schema_converter import function_tools_to_openai


class OpenAIStream(LLMStream):
    """OpenAI streaming response handler."""

    def __init__(
        self,
        llm: "OpenAI",
        messages: List[Message],
        tools: List[FunctionTool],
        stream: Any,
    ):
        super().__init__(llm, messages, tools)
        self._stream = stream
        self._tool_call_buffers: Dict[int, ToolCall] = {}

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over OpenAI stream chunks."""
        try:
            async for chunk in self._stream:
                yield self._process_chunk(chunk)
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            raise

    def _process_chunk(self, chunk: Any) -> StreamChunk:
        """Process a single OpenAI chunk."""
        text = None
        tool_calls = []
        is_final = False
        usage = None

        # Handle different response formats
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)

            if delta:
                # Extract text
                if hasattr(delta, "content") and delta.content:
                    text = delta.content

                # Extract tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index if hasattr(tc, "index") else 0

                        if idx not in self._tool_call_buffers:
                            self._tool_call_buffers[idx] = ToolCall(
                                id=tc.id if tc.id else f"call_{idx}",
                                name=tc.function.name if tc.function and tc.function.name else "",
                                arguments="",
                            )

                        if tc.function and tc.function.arguments:
                            self._tool_call_buffers[idx].arguments += tc.function.arguments

                        tool_calls.append(self._tool_call_buffers[idx])

            # Check for finish
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                is_final = True
                # Mark tool calls as complete
                for tc in self._tool_call_buffers.values():
                    tc.is_complete = True

        # Extract usage info
        if hasattr(chunk, "usage") and chunk.usage:
            usage = UsageInfo(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
                total_tokens=chunk.usage.total_tokens or 0,
            )

        return StreamChunk(
            text=text,
            tool_calls=tool_calls if tool_calls else [],
            is_final=is_final,
            raw=chunk,
            usage=usage,
        )


class OpenAI(LLM):
    """
    OpenAI LLM provider.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.

    Example:
        ```python
        from line.llm.providers.openai import OpenAI
        from line.llm import LlmConfig

        llm = OpenAI(
            model="gpt-4o",
            api_key="sk-...",
            config=LlmConfig(temperature=0.7),
        )

        messages = [Message(role="user", content="Hello!")]
        async with llm.chat(messages) as stream:
            async for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="")
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model: The model to use (e.g., "gpt-4o", "gpt-3.5-turbo").
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            config: LLM configuration.
            base_url: Optional base URL for API requests (for proxies/alternatives).
        """
        super().__init__(model, api_key, config)
        self._base_url = base_url
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "openai is required for OpenAI integration. "
                    "Install with: pip install openai"
                ) from e

            kwargs: Dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**kwargs)

        return self._client

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> OpenAIStream:
        """
        Start a chat completion with OpenAI.

        Args:
            messages: The conversation messages.
            tools: Optional tools available to the LLM.
            **kwargs: Additional arguments passed to the API.

        Returns:
            An OpenAIStream for the response.
        """
        client = self._get_client()

        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role}

            if msg.content:
                openai_msg["content"] = msg.content

            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]

            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            if msg.name:
                openai_msg["name"] = msg.name

            openai_messages.append(openai_msg)

        # Build API kwargs
        api_kwargs = self._config.to_openai_kwargs()
        api_kwargs.update(kwargs)

        # Add system instructions if configured
        if self._config.system_instructions and not any(
            m["role"] == "system" for m in openai_messages
        ):
            openai_messages.insert(0, {"role": "system", "content": self._config.system_instructions})

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = function_tools_to_openai(tools, strict=True, responses_api=False)

        # Create the stream
        stream = client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            tools=openai_tools,
            stream=True,
            **api_kwargs,
        )

        return OpenAIStream(self, messages, tools or [], stream)

    async def aclose(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
