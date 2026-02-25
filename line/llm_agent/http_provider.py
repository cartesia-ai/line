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


class HttpProvider:
    """
    LLM provider using LiteLLM for unified multi-provider access.

    Handles streaming responses and tool calls for all LiteLLM-supported models.

    Config normalization and reasoning-effort detection are handled by the
    ``LlmProvider`` facade — this class receives pre-normalized configs and
    pre-computed reasoning flags.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[LlmConfig] = None,
        supports_reasoning_effort: bool = False,
        default_reasoning_effort: str = "low",
    ):
        self._model = model
        self._api_key = api_key
        self._config = config or LlmConfig()
        self._supports_reasoning_effort = supports_reasoning_effort
        self._default_reasoning_effort = default_reasoning_effort

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[FunctionTool]] = None,
        config: Optional[LlmConfig] = None,
        **kwargs,
    ) -> "_ChatStream":
        """Start a streaming chat completion.

        Args:
            messages: Conversation messages.
            tools: Optional function tools available for this call.
            config: Optional per-call config override (pre-normalized by LlmProvider).
        """
        cfg = config if config else self._config
        llm_messages = self._build_messages(messages, cfg)

        llm_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": llm_messages,
            "stream": True,
            "num_retries": cfg.num_retries,
        }

        if self._api_key:
            llm_kwargs["api_key"] = self._api_key
        if cfg.fallbacks:
            llm_kwargs["fallbacks"] = cfg.fallbacks
        if cfg.timeout:
            llm_kwargs["timeout"] = cfg.timeout

        # Add config parameters
        if cfg.temperature is not None:
            llm_kwargs["temperature"] = cfg.temperature
        if cfg.max_tokens is not None:
            llm_kwargs["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            llm_kwargs["top_p"] = cfg.top_p
        if cfg.stop:
            llm_kwargs["stop"] = cfg.stop
        if cfg.seed is not None:
            llm_kwargs["seed"] = cfg.seed
        if cfg.presence_penalty is not None:
            llm_kwargs["presence_penalty"] = cfg.presence_penalty
        if cfg.frequency_penalty is not None:
            llm_kwargs["frequency_penalty"] = cfg.frequency_penalty
        if self._supports_reasoning_effort:
            llm_kwargs["reasoning_effort"] = cfg.reasoning_effort or self._default_reasoning_effort

        if cfg.extra:
            llm_kwargs.update(cfg.extra)

        if tools:
            llm_kwargs["tools"] = function_tools_to_openai(tools, strict=False)

        llm_kwargs.update(kwargs)

        return _ChatStream(llm_kwargs)

    def _build_messages(
        self, messages: List[Message], config: Optional[LlmConfig] = None
    ) -> List[Dict[str, Any]]:
        """Convert Message objects to LiteLLM format."""
        cfg = config if config else self._config
        result = []

        if cfg.system_prompt:
            result.append({"role": "system", "content": cfg.system_prompt})

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

    async def warmup(self, config=None):
        """No-op for stateless HTTP provider."""
        pass

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
