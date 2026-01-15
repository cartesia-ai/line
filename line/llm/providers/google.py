"""
Google (Gemini) LLM provider implementation.

This module provides the Google Gemini implementation of the LLM interface.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from line.llm.config import LlmConfig
from line.llm.function_tool import FunctionTool
from line.llm.providers.base import LLM, LLMStream, Message, StreamChunk, ToolCall, UsageInfo
from line.llm.schema_converter import function_tools_to_gemini, merge_gemini_tools


class GoogleStream(LLMStream):
    """Google/Gemini streaming response handler."""

    def __init__(
        self,
        llm: "Google",
        messages: List[Message],
        tools: List[FunctionTool],
        stream: Any,
    ):
        super().__init__(llm, messages, tools)
        self._stream = stream
        self._tool_call_counter = 0

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over Gemini stream chunks."""
        try:
            async for chunk in self._stream:
                yield self._process_chunk(chunk)
        except Exception as e:
            logger.error(f"Gemini stream error: {e}")
            raise

    def _process_chunk(self, chunk: Any) -> StreamChunk:
        """Process a single Gemini chunk."""
        text = None
        tool_calls = []
        is_final = False
        usage = None

        # Extract text content
        if hasattr(chunk, "text") and chunk.text:
            text = chunk.text

        # Extract function calls
        if hasattr(chunk, "function_calls") and chunk.function_calls:
            for fc in chunk.function_calls:
                self._tool_call_counter += 1
                tool_calls.append(
                    ToolCall(
                        id=f"call_{self._tool_call_counter}",
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                        is_complete=True,  # Gemini provides complete function calls
                    )
                )

        # Check for candidates to determine if final
        if hasattr(chunk, "candidates") and chunk.candidates:
            for candidate in chunk.candidates:
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    is_final = True

        # Extract usage metadata
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            um = chunk.usage_metadata
            usage = UsageInfo(
                prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                total_tokens=getattr(um, "total_token_count", 0) or 0,
            )

        return StreamChunk(
            text=text,
            tool_calls=tool_calls,
            is_final=is_final,
            raw=chunk,
            usage=usage,
        )


class Google(LLM):
    """
    Google (Gemini) LLM provider.

    Supports Gemini 1.5, Gemini 2.0, and other Google models.

    Example:
        ```python
        from line.llm.providers.google import Google
        from line.llm import LlmConfig

        llm = Google(
            model="gemini-2.0-flash",
            api_key="...",
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
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
    ):
        """
        Initialize the Google provider.

        Args:
            model: The model to use (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            config: LLM configuration.
        """
        super().__init__(model, api_key, config)
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "google-genai is required for Gemini integration. "
                    "Install with: pip install google-genai"
                ) from e

            kwargs: Dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key

            self._client = genai.Client(**kwargs)

        return self._client

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> GoogleStream:
        """
        Start a chat completion with Gemini.

        Args:
            messages: The conversation messages.
            tools: Optional tools available to the LLM.
            **kwargs: Additional arguments passed to the API.

        Returns:
            A GoogleStream for the response.
        """
        try:
            from google.genai import types as gemini_types
        except ImportError as e:
            raise ImportError(
                "google-genai is required for Gemini integration. "
                "Install with: pip install google-genai"
            ) from e

        client = self._get_client()

        # Convert messages to Gemini format
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            role = "model" if msg.role == "assistant" else "user"
            parts = []

            if msg.content:
                parts.append(gemini_types.Part.from_text(text=msg.content))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.arguments) if tc.arguments else {}
                    parts.append(
                        gemini_types.Part.from_function_call(name=tc.name, args=args)
                    )

            if msg.tool_call_id and msg.name:
                # This is a tool result
                parts.append(
                    gemini_types.Part.from_function_response(
                        name=msg.name, response={"output": msg.content}
                    )
                )

            if parts:
                if role == "user":
                    gemini_contents.append(gemini_types.UserContent(parts=parts))
                else:
                    gemini_contents.append(gemini_types.ModelContent(parts=parts))

        # Build generation config
        config_kwargs = self._config.to_gemini_kwargs()
        config_kwargs.update(kwargs)

        # Add system instruction
        system = system_instruction or self._config.system_instructions
        if system:
            config_kwargs["system_instruction"] = system

        # Convert and merge tools
        gemini_tools = None
        if tools:
            tool_list = function_tools_to_gemini(tools)
            gemini_tools = [merge_gemini_tools(tool_list)]

        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        # Handle thinking budget
        if self._config.thinking_budget is not None:
            config_kwargs["thinking_config"] = gemini_types.ThinkingConfig(
                thinking_budget=self._config.thinking_budget
            )

        generation_config = gemini_types.GenerateContentConfig(**config_kwargs)

        # Create the stream
        stream = client.aio.models.generate_content_stream(
            model=self._model,
            contents=gemini_contents,
            config=generation_config,
        )

        return GoogleStream(self, messages, tools or [], stream)

    async def aclose(self) -> None:
        """Close the Google client."""
        # The google-genai client doesn't need explicit closing
        self._client = None
