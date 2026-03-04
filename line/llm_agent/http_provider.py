"""
HTTP-based LLM Provider using LiteLLM.

Provides a unified interface to 100+ LLM providers via LiteLLM.
See https://docs.litellm.ai/docs/providers for supported providers.

Model naming:
- OpenAI: "gpt-4o", "gpt-4o-mini"
- Anthropic: "anthropic/claude-haiku-4-5-20251001"
- Google: "gemini/gemini-2.5-flash-preview-09-2025"
"""

from typing import Any, AsyncIterator, Dict, List, Optional

from litellm import acompletion

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, StreamChunk, ToolCall
from line.llm_agent.schema_converter import function_tools_to_openai
from line.llm_agent.tools.utils import FunctionTool


class _HttpProvider:
    """
    LLM provider using LiteLLM for unified multi-provider access.

    Handles streaming responses and tool calls for all LiteLLM-supported models.

    Config normalization and reasoning-effort detection are handled by the
    ``LlmProvider`` facade — this class receives fully-resolved configs and
    tools on every call.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        supports_reasoning_effort: bool = False,
        default_reasoning_effort: str = "low",
    ):
        self._model = model
        self._api_key = api_key
        self._supports_reasoning_effort = supports_reasoning_effort
        self._default_reasoning_effort = default_reasoning_effort

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        config: Optional[LlmConfig] = None,
        **kwargs,
    ) -> "_ChatStream":
        """Start a streaming chat completion.

        Returns a ``_ChatStream`` async context manager.  The actual HTTP
        request is issued in ``__aenter__``.

        Args:
            messages: Conversation messages.
            tools: Optional function tools available for this call.
            config: Optional per-call config override (pre-normalized by LlmProvider).
        """
        llm_messages = self._build_messages(messages, config)

        llm_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": llm_messages,
            "stream": True,
            "num_retries": config.num_retries,
        }

        if self._api_key:
            llm_kwargs["api_key"] = self._api_key
        if config.fallbacks:
            llm_kwargs["fallbacks"] = config.fallbacks
        if config.timeout:
            llm_kwargs["timeout"] = config.timeout

        # Add config parameters
        if config.temperature is not None:
            llm_kwargs["temperature"] = config.temperature
        if config.max_tokens is not None:
            llm_kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            llm_kwargs["top_p"] = config.top_p
        if config.stop:
            llm_kwargs["stop"] = config.stop
        if config.seed is not None:
            llm_kwargs["seed"] = config.seed
        if config.presence_penalty is not None:
            llm_kwargs["presence_penalty"] = config.presence_penalty
        if config.frequency_penalty is not None:
            llm_kwargs["frequency_penalty"] = config.frequency_penalty
        if self._supports_reasoning_effort:
            llm_kwargs["reasoning_effort"] = config.reasoning_effort or self._default_reasoning_effort

        if config.extra:
            llm_kwargs.update(config.extra)

        if tools:
            llm_kwargs["tools"] = function_tools_to_openai(tools, strict=False)

        llm_kwargs.update(kwargs)

        return _ChatStream(llm_kwargs)

    def _build_messages(
        self, messages: List[Message], config: Optional[LlmConfig] = None
    ) -> List[Dict[str, Any]]:
        """Convert Message objects to LiteLLM format."""
        result = []

        if config.system_prompt:
            result.append({"role": "system", "content": config.system_prompt})

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

    async def warmup(self, config=None, tools=None):
        """No-op for stateless HTTP provider."""
        pass

    async def aclose(self) -> None:
        """Close the provider (no-op for LiteLLM)."""
        pass


class _ChatStream:
    """Async-iterable stream for HTTP chat responses.

    All setup, iteration, and cleanup happen inside ``__aiter__`` — no
    ``async with`` is needed.  Breaking out of ``async for`` triggers
    ``GeneratorExit`` which closes the underlying response via ``finally``.
    """

    def __init__(self, llm_kwargs: Dict[str, Any]):
        self._kwargs = llm_kwargs

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        response = await acompletion(**self._kwargs)
        try:
            tool_calls: Dict[int, ToolCall] = {}

            async for chunk in response:
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
        finally:
            aclose = getattr(response, "aclose", None)
            if callable(aclose):
                await aclose()
