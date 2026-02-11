"""
LLM Provider using LiteLLM.

Provides a unified interface to 100+ LLM providers via LiteLLM.
See https://docs.litellm.ai/docs/providers for supported providers.

Model naming:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-haiku-4-5-20251001"
- Google: "gemini/gemini-2.5-flash-preview-09-2025"
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from litellm import acompletion, get_llm_provider, get_supported_openai_params
from litellm.utils import get_optional_params

from line.llm_agent.config import LlmConfig
from line.llm_agent.schema_converter import function_tools_to_openai
from line.llm_agent.tools.utils import FunctionTool


@dataclass
class ToolCall:
    """A tool/function call from the LLM."""

    id: str
    name: str
    arguments: str = ""
    is_complete: bool = False
    thought_signature: Optional[str] = None  # For Gemini 3+ models


@dataclass
class StreamChunk:
    """An output chunk from an LLM stream."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_final: bool = False


@dataclass
class Message:
    """A input message in the conversation."""

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

        supported = get_supported_openai_params(model=model) or []
        self._supports_reasoning_effort = "reasoning_effort" in supported

        # Determine the right default when no explicit reasoning_effort is configured.
        # "none" is ideal (disables reasoning entirely) but not all providers support it.
        # Probe litellm's own parameter mapping to find out: if mapping "none" through the
        # provider's config raises, fall back to "low" (the lowest universally-supported level).
        self._default_reasoning_effort = "low"
        if self._supports_reasoning_effort:
            try:
                _, provider, _, _ = get_llm_provider(model=model)
                get_optional_params(model=model, custom_llm_provider=provider, reasoning_effort="none")
                self._default_reasoning_effort = "none"
            except Exception:
                pass

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        **kwargs,
    ) -> "_ChatStream":
        """Start a streaming chat completion."""
        llm_messages = self._build_messages(messages)

        llm_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": llm_messages,
            "stream": True,
            "num_retries": self._num_retries,
        }

        if self._api_key:
            llm_kwargs["api_key"] = self._api_key
        if self._fallbacks:
            llm_kwargs["fallbacks"] = self._fallbacks
        if self._timeout:
            llm_kwargs["timeout"] = self._timeout

        # Add config parameters
        if self._config.temperature is not None:
            llm_kwargs["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            llm_kwargs["max_tokens"] = self._config.max_tokens
        if self._config.top_p is not None:
            llm_kwargs["top_p"] = self._config.top_p
        if self._config.stop:
            llm_kwargs["stop"] = self._config.stop
        if self._config.seed is not None:
            llm_kwargs["seed"] = self._config.seed
        if self._config.presence_penalty is not None:
            llm_kwargs["presence_penalty"] = self._config.presence_penalty
        if self._config.frequency_penalty is not None:
            llm_kwargs["frequency_penalty"] = self._config.frequency_penalty
        if self._supports_reasoning_effort:
            llm_kwargs["reasoning_effort"] = self._config.reasoning_effort or self._default_reasoning_effort

        if self._config.extra:
            llm_kwargs.update(self._config.extra)

        if tools:
            llm_kwargs["tools"] = function_tools_to_openai(tools, strict=False)

        llm_kwargs.update(kwargs)

        return _ChatStream(llm_kwargs)

    def _build_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to LiteLLM format."""
        result = []

        if self._config.system_prompt:
            result.append({"role": "system", "content": self._config.system_prompt})

        for msg in messages:
            llm_msg: Dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                llm_msg["content"] = msg.content

            if msg.tool_calls:
                # ToolCallRequest
                llm_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                        # Include thought_signature for Gemini 3+ models
                        # LiteLLM expects this in provider_specific_fields
                        **(
                            {"provider_specific_fields": {"thought_signature": tc.thought_signature}}
                            if tc.thought_signature
                            else {}
                        ),
                    }
                    for tc in msg.tool_calls
                ]

            if msg.role == "tool":
                # ToolCallResponse
                llm_msg["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    llm_msg["name"] = msg.name

            result.append(llm_msg)
        return result

    async def aclose(self) -> None:
        """Close the provider (no-op for LiteLLM)."""
        pass


class _ChatStream:
    """Async context manager for streaming chat responses."""

    def __init__(self, llm_kwargs: Dict[str, Any]):
        self._kwargs = llm_kwargs
        self._response = None

    async def __aenter__(self) -> "_ChatStream":
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

                        # Capture thought_signature for Gemini 3+ models
                        # LiteLLM stores it in provider_specific_fields
                        provider_fields = getattr(tc, "provider_specific_fields", None)
                        if provider_fields:
                            thought_sig = provider_fields.get("thought_signature")
                            if thought_sig:
                                tool_calls[idx].thought_signature = thought_sig

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
