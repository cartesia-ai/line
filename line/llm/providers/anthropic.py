"""
Anthropic LLM provider implementation.

This module provides the Anthropic (Claude) implementation of the LLM interface.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from line.llm.config import LlmConfig
from line.llm.function_tool import FunctionTool
from line.llm.providers.base import LLM, LLMStream, Message, StreamChunk, ToolCall, UsageInfo
from line.llm.schema_converter import function_tools_to_anthropic


class AnthropicStream(LLMStream):
    """Anthropic streaming response handler."""

    def __init__(
        self,
        llm: "Anthropic",
        messages: List[Message],
        tools: List[FunctionTool],
        stream: Any,
    ):
        super().__init__(llm, messages, tools)
        self._stream = stream
        self._current_tool_call: Optional[ToolCall] = None
        self._tool_call_counter = 0

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over Anthropic stream chunks."""
        try:
            async for event in self._stream:
                chunk = self._process_event(event)
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            raise

    def _process_event(self, event: Any) -> Optional[StreamChunk]:
        """Process a single Anthropic event."""
        event_type = getattr(event, "type", None)

        if event_type == "content_block_start":
            content_block = event.content_block
            if content_block.type == "tool_use":
                self._current_tool_call = ToolCall(
                    id=content_block.id,
                    name=content_block.name,
                    arguments="",
                )
            return None

        elif event_type == "content_block_delta":
            delta = event.delta

            if hasattr(delta, "text"):
                return StreamChunk(text=delta.text)

            elif hasattr(delta, "partial_json"):
                if self._current_tool_call:
                    self._current_tool_call.arguments += delta.partial_json
                return None

        elif event_type == "content_block_stop":
            if self._current_tool_call:
                self._current_tool_call.is_complete = True
                tool_call = self._current_tool_call
                self._current_tool_call = None
                return StreamChunk(tool_calls=[tool_call])
            return None

        elif event_type == "message_delta":
            # End of message
            usage = None
            if hasattr(event, "usage"):
                usage = UsageInfo(
                    prompt_tokens=0,  # Anthropic reports this differently
                    completion_tokens=event.usage.output_tokens or 0,
                    total_tokens=event.usage.output_tokens or 0,
                )
            return StreamChunk(is_final=True, usage=usage)

        elif event_type == "message_start":
            # Beginning of message, extract input tokens
            if hasattr(event, "message") and hasattr(event.message, "usage"):
                # Store for later
                pass
            return None

        return None


class Anthropic(LLM):
    """
    Anthropic (Claude) LLM provider.

    Supports Claude 3, Claude 3.5, and Claude Opus 4 models.

    Example:
        ```python
        from line.llm.providers.anthropic import Anthropic
        from line.llm import LlmConfig

        llm = Anthropic(
            model="claude-3-5-sonnet-20241022",
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
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model: The model to use (e.g., "claude-3-5-sonnet-20241022").
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            config: LLM configuration.
            base_url: Optional base URL for API requests.
        """
        super().__init__(model, api_key, config)
        self._base_url = base_url
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic is required for Anthropic integration. "
                    "Install with: pip install anthropic"
                ) from e

            kwargs: Dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncAnthropic(**kwargs)

        return self._client

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> AnthropicStream:
        """
        Start a chat completion with Anthropic.

        Args:
            messages: The conversation messages.
            tools: Optional tools available to the LLM.
            **kwargs: Additional arguments passed to the API.

        Returns:
            An AnthropicStream for the response.
        """
        client = self._get_client()

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_content = None

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                continue

            anthropic_msg: Dict[str, Any] = {"role": msg.role}

            # Build content
            content: List[Dict[str, Any]] = []

            if msg.content:
                content.append({"type": "text", "text": msg.content})

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": json.loads(tc.arguments) if tc.arguments else {},
                        }
                    )

            if msg.tool_call_id:
                # This is a tool result message
                anthropic_msg["role"] = "user"
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }
                ]

            anthropic_msg["content"] = content if len(content) > 1 else (content[0] if content else "")
            anthropic_messages.append(anthropic_msg)

        # Build API kwargs
        api_kwargs = self._config.to_anthropic_kwargs()
        api_kwargs.update(kwargs)

        # Set default max_tokens if not provided (required by Anthropic)
        if "max_tokens" not in api_kwargs:
            api_kwargs["max_tokens"] = 4096

        # Add system instructions
        system = system_content or self._config.system_instructions
        if system:
            api_kwargs["system"] = system

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = function_tools_to_anthropic(tools)

        # Create the stream
        stream = client.messages.stream(
            model=self._model,
            messages=anthropic_messages,
            tools=anthropic_tools,
            **api_kwargs,
        )

        return AnthropicStream(self, messages, tools or [], stream)

    async def aclose(self) -> None:
        """Close the Anthropic client."""
        if self._client:
            await self._client.close()
            self._client = None
