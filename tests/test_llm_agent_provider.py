"""Tests for the LlmProvider facade and HTTP backend."""

import asyncio
from typing import Annotated, Any, List, Optional, get_type_hints

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.http_provider import _feed_tool_args, _HttpProvider
from line.llm_agent.provider import (
    ChatStream,
    LlmProvider,
    Message,
    ToolCall,
    _extract_instructions_and_messages,
    _get_model_config,
    _is_realtime_model,
    _is_websocket_model,
    _normalize_messages,
)
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.tools.system import web_search


class _DummyBackend:
    def __init__(self):
        self.calls = []
        self.warmup_calls = []

    def chat(self, messages, tools=None, *, config, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "config": config,
                "kwargs": kwargs,
            }
        )
        return "ok"

    async def warmup(self, config, tools=None, **kwargs):
        self.warmup_calls.append(
            {
                "config": config,
                "tools": tools,
                "kwargs": kwargs,
            }
        )

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# LlmProvider facade
# ---------------------------------------------------------------------------


def test_llm_provider_preserves_native_web_search_defaults(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "supports_web_search", lambda model: True)

    provider = LlmProvider(
        model="gpt-4o",
        api_key="test-key",
        tools=[web_search(search_context_size="high")],
    )
    backend = _DummyBackend()
    provider._backend = backend

    result = provider.chat([Message(role="user", content="hi")])

    assert result == "ok"
    assert len(backend.calls) == 1
    assert backend.calls[0]["tools"] == []
    assert backend.calls[0]["kwargs"]["web_search_options"] == {"search_context_size": "high"}


def test_llm_provider_web_search_override_replaces_default(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "supports_web_search", lambda model: False)

    async def web_search_override(ctx, query: Annotated[str, "Search query"]) -> str:
        return f"override: {query}"

    web_search_override.__name__ = "web_search"

    provider = LlmProvider(
        model="gpt-4o",
        api_key="test-key",
        tools=[web_search(search_context_size="high")],
    )
    backend = _DummyBackend()
    provider._backend = backend

    result = provider.chat(
        [Message(role="user", content="hi")],
        tools=[web_search_override],
    )

    assert result == "ok"
    assert len(backend.calls) == 1
    assert [tool.name for tool in backend.calls[0]["tools"]] == ["web_search"]
    assert backend.calls[0]["tools"][0].description == ""
    assert "web_search_options" not in backend.calls[0]["kwargs"]


def test_llm_provider__set_tools_replaces_default_tools(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "supports_web_search", lambda model: False)

    async def original_tool(ctx, query: Annotated[str, "Search query"]) -> str:
        return f"original: {query}"

    async def replacement_tool(ctx, query: Annotated[str, "Search query"]) -> str:
        return f"replacement: {query}"

    provider = LlmProvider(
        model="gpt-4o",
        api_key="test-key",
        tools=[original_tool],
    )
    backend = _DummyBackend()
    provider._backend = backend

    provider._set_tools([replacement_tool])
    result = provider.chat([Message(role="user", content="hi")])

    assert result == "ok"
    assert len(backend.calls) == 1
    assert [tool.name for tool in backend.calls[0]["tools"]] == ["replacement_tool"]


def test_llm_provider_warmup_normalizes_current_default_tools(monkeypatch):
    import asyncio

    import litellm

    monkeypatch.setattr(litellm, "supports_web_search", lambda model: False)

    async def original_tool(ctx, query: Annotated[str, "Search query"]) -> str:
        return f"original: {query}"

    async def replacement_tool(ctx, query: Annotated[str, "Search query"]) -> str:
        return f"replacement: {query}"

    provider = LlmProvider(
        model="gpt-4o",
        api_key="test-key",
        tools=[original_tool],
    )
    backend = _DummyBackend()
    provider._backend = backend

    provider._set_tools([replacement_tool])
    asyncio.run(provider.warmup())

    assert len(backend.warmup_calls) == 1
    assert [tool.name for tool in backend.warmup_calls[0]["tools"]] == ["replacement_tool"]


def test_llm_provider_warmup_preserves_native_web_search_defaults(monkeypatch):
    import asyncio

    import litellm

    monkeypatch.setattr(litellm, "supports_web_search", lambda model: True)

    provider = LlmProvider(
        model="openai/gpt-5.2",
        api_key="test-key",
        tools=[web_search(search_context_size="high")],
    )
    backend = _DummyBackend()
    provider._backend = backend

    asyncio.run(provider.warmup())

    assert len(backend.warmup_calls) == 1
    assert backend.warmup_calls[0]["tools"] == []
    assert backend.warmup_calls[0]["kwargs"]["web_search_options"] == {"search_context_size": "high"}


def test_llm_provider_routes_websocket_models_with_unsupported_config_to_http_backend():
    provider = LlmProvider(
        model="openai/gpt-5.2",
        api_key="test-key",
    )
    websocket_backend = _DummyBackend()
    http_backend = _DummyBackend()
    provider._backend = websocket_backend
    provider._http_fallback_backend = http_backend

    result = provider.chat(
        [Message(role="user", content="hi")],
        config=LlmConfig(stop=["DONE"]),
    )

    assert result == "ok"
    assert len(websocket_backend.calls) == 0
    assert len(http_backend.calls) == 1


def test_llm_provider_keeps_websocket_backend_for_supported_config():
    provider = LlmProvider(
        model="openai/gpt-5.2",
        api_key="test-key",
    )
    websocket_backend = _DummyBackend()
    http_backend = _DummyBackend()
    provider._backend = websocket_backend
    provider._http_fallback_backend = http_backend

    result = provider.chat(
        [Message(role="user", content="hi")],
        config=LlmConfig(temperature=0.2, top_p=0.9),
    )

    assert result == "ok"
    assert len(websocket_backend.calls) == 1
    assert len(http_backend.calls) == 0


def test_llm_provider_warmup_routes_unsupported_websocket_config_to_http_backend():
    provider = LlmProvider(
        model="openai/gpt-5.2",
        api_key="test-key",
    )
    websocket_backend = _DummyBackend()
    http_backend = _DummyBackend()
    provider._backend = websocket_backend
    provider._http_fallback_backend = http_backend

    asyncio.run(provider.warmup(config=LlmConfig(extra={"service_tier": "flex"})))

    assert len(websocket_backend.warmup_calls) == 0
    assert len(http_backend.warmup_calls) == 1


def test_extract_instructions_includes_config_system_prompt():
    instructions, non_system = _extract_instructions_and_messages(
        [Message(role="system", content="call-specific"), Message(role="user", content="hi")],
        LlmConfig(system_prompt="base"),
    )

    assert instructions == "base\n\ncall-specific"
    assert len(non_system) == 1
    assert non_system[0].role == "user"


def test_build_openai_tool_defs_adds_native_web_search():
    assert build_openai_tool_defs(
        web_search_options={"search_context_size": "high"},
        responses_api=True,
    ) == [{"type": "web_search", "search_context_size": "high"}]


def test_llm_provider_requires_api_key():
    try:
        LlmProvider(model="gpt-4o", api_key="")
    except ValueError as exc:
        assert "Missing API key" in str(exc)
    else:
        raise AssertionError("Expected LlmProvider to reject missing api_key")


def test_llm_provider_rejects_unsupported_model(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)

    try:
        LlmProvider(model="fake-provider/definitely-not-a-real-model", api_key="test-key")
    except ValueError as exc:
        assert "is not supported" in str(exc)
    else:
        raise AssertionError("Expected LlmProvider to reject unsupported model")


def test_is_supported_model_accepts_direct_openai_websocket_model(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)
    assert _get_model_config("openai/gpt-5.2") is not None


def test_backend_override_http_for_websocket_model():
    provider = LlmProvider(model="openai/gpt-5.2", api_key="test-key", backend="http")
    from line.llm_agent.http_provider import _HttpProvider

    assert isinstance(provider._backend, _HttpProvider)


def test_backend_override_realtime_rejects_non_realtime_model():
    try:
        LlmProvider(model="gpt-4o", api_key="test-key", backend="realtime")
    except ValueError as exc:
        assert "realtime" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for non-realtime model with realtime backend")


def test_backend_override_websocket_rejects_non_websocket_model():
    try:
        LlmProvider(model="gpt-4o", api_key="test-key", backend="websocket")
    except ValueError as exc:
        assert "websocket" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for non-websocket model with websocket backend")


def test_backend_override_invalid_value():
    try:
        LlmProvider(model="gpt-4o", api_key="test-key", backend="banana")
    except ValueError as exc:
        assert "Invalid backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid backend value")


# ---------------------------------------------------------------------------
# _get_model_config
# ---------------------------------------------------------------------------


class TestGetModelConfig:
    """Tests for _get_model_config backend routing and reasoning-effort detection."""

    # -- Backend routing -------------------------------------------------------

    def test_realtime_model_selects_realtime_backend(self):
        cfg = _get_model_config("openai/gpt-4o-realtime-preview")
        assert cfg.backend == "realtime"
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_websocket_model_selects_websocket_backend(self):
        cfg = _get_model_config("openai/gpt-5.2")
        assert cfg.backend == "websocket"

    def test_websocket_model_with_http_override_selects_http(self):
        cfg = _get_model_config("openai/gpt-5.2", backend="http")
        assert cfg.backend == "http"

    def test_websocket_model_with_explicit_websocket_backend(self):
        cfg = _get_model_config("openai/gpt-5.2", backend="websocket")
        assert cfg.backend == "websocket"

    def test_plain_http_model_selects_http_backend(self):
        cfg = _get_model_config("openai/gpt-4o")
        assert cfg.backend == "http"

    # -- Error cases -----------------------------------------------------------

    def test_invalid_backend_raises(self):
        try:
            _get_model_config("openai/gpt-4o", backend="banana")
        except ValueError as exc:
            assert "Invalid backend" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    def test_realtime_backend_with_non_realtime_model_raises(self):
        try:
            _get_model_config("openai/gpt-4o", backend="realtime")
        except ValueError as exc:
            assert "realtime" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError")

    def test_non_realtime_backend_with_realtime_model_raises(self):
        try:
            _get_model_config("openai/gpt-4o-realtime-preview", backend="http")
        except ValueError as exc:
            assert "incompatible" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError")

    def test_websocket_backend_with_non_websocket_model_raises(self):
        try:
            _get_model_config("openai/gpt-4o", backend="websocket")
        except ValueError as exc:
            assert "websocket" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError")

    def test_unsupported_model_raises(self, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)
        try:
            _get_model_config("fake-provider/not-a-model")
        except ValueError as exc:
            assert "is not supported" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    # -- Reasoning effort for websocket models ---------------------------------

    def test_websocket_model_with_reasoning_support(self):
        """Reasoning models (e.g. o3) routed via websocket get reasoning enabled."""
        cfg = _get_model_config("openai/o3")
        assert cfg.backend == "websocket"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort == "low"

    def test_websocket_model_without_reasoning_support(self):
        """Non-reasoning websocket models (e.g. gpt-4.1) must not get reasoning params."""
        cfg = _get_model_config("openai/gpt-4.1")
        assert cfg.backend == "websocket"
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    # -- Reasoning effort for HTTP models --------------------------------------

    def test_http_model_without_reasoning_support(self):
        cfg = _get_model_config("openai/gpt-4o")
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_http_model_with_reasoning_support(self):
        """o4-mini goes through the HTTP path and supports reasoning."""
        cfg = _get_model_config("openai/o4-mini")
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort is not None

    def test_websocket_model_forced_to_http_gets_correct_reasoning(self):
        """gpt-4.1 forced to HTTP should still have reasoning disabled."""
        cfg = _get_model_config("openai/gpt-4.1", backend="http")
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_anthropic_model_reasoning_default_is_none(self):
        """Anthropic models use None (not 'low') to skip the thinking block."""
        cfg = _get_model_config("anthropic/claude-sonnet-4-20250514")
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort is None


def test_is_websocket_model_matches_gpt52_variants():
    assert _is_websocket_model("gpt-5.2")
    assert _is_websocket_model("gpt-5.2-pro")
    assert _is_websocket_model("gpt-5.4")
    assert _is_websocket_model("gpt-5.4-mini")
    assert _is_websocket_model("chatgpt/gpt-5.4-pro")
    assert not _is_websocket_model("azure/gpt-5.2")
    assert not _is_websocket_model("openrouter/gpt-5.2")
    assert not _is_websocket_model("openai/gpt-4o")


def test_is_realtime_model_matches_only_direct_openai_models():
    assert _is_realtime_model("openai/gpt-4o-realtime-preview")
    assert _is_realtime_model("gpt-4o-realtime-preview")  # detected via LiteLLM registry
    assert not _is_realtime_model("azure/gpt-4o-realtime-preview")
    assert not _is_realtime_model("openrouter/gpt-4o-realtime-preview")


def test_llm_provider_public_methods_have_type_hints():
    chat_hints = get_type_hints(LlmProvider.chat)
    assert chat_hints["messages"] == List[Message]
    assert chat_hints["tools"] == Optional[List[Any]]
    assert chat_hints["config"] == Optional[LlmConfig]
    assert chat_hints["kwargs"] is Any
    assert chat_hints["return"] == ChatStream

    warmup_hints = get_type_hints(LlmProvider.warmup)
    assert warmup_hints["config"] == Optional[LlmConfig]
    assert warmup_hints["tools"] == Optional[List[Any]]
    assert warmup_hints["return"] is type(None)

    aclose_hints = get_type_hints(LlmProvider.aclose)
    assert aclose_hints["return"] is type(None)


# ---------------------------------------------------------------------------
# HTTP provider
# ---------------------------------------------------------------------------


def test_http_chat_stream_awaits_response_aclose(monkeypatch):
    closed = False

    class _FakeResponse:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            nonlocal closed
            closed = True

    async def _fake_acompletion(**kwargs):
        return _FakeResponse()

    monkeypatch.setattr("line.llm_agent.http_provider.acompletion", _fake_acompletion)

    provider = _HttpProvider(model="gpt-4o")

    async def _consume():
        async for _ in provider.chat(
            [Message(role="user", content="hi")], config=_normalize_config(LlmConfig())
        ):
            pass

    asyncio.run(_consume())

    assert closed


# ---------------------------------------------------------------------------
# _feed_tool_args
# ---------------------------------------------------------------------------


class TestIncrementalConcatenation:
    """OpenAI and Anthropic stream arguments as small JSON fragments."""

    def test_single_fragment_complete_object(self):
        state = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert state.args == '{"city": "Tokyo"}'
        assert state.depth == 0

    def test_two_fragments(self):
        s1 = _feed_tool_args(None, '{"ci')
        assert s1.args == '{"ci'
        assert s1.depth == 1

        s2 = _feed_tool_args(s1, 'ty": "Tokyo"}')
        assert s2.args == '{"city": "Tokyo"}'
        assert s2.depth == 0

    def test_many_small_fragments(self):
        fragments = ["{", '"name"', ": ", '"', "Alice", '"', ", ", '"age"', ": ", "30", "}"]
        state = None
        for frag in fragments:
            state = _feed_tool_args(state, frag)

        assert state.args == '{"name": "Alice", "age": 30}'
        assert state.depth == 0
        assert state.in_string is False
        assert state.escape_next is False

    def test_empty_object(self):
        s1 = _feed_tool_args(None, "{")
        assert s1.depth == 1

        s2 = _feed_tool_args(s1, "}")
        assert s2.args == "{}"
        assert s2.depth == 0


class TestGeminiReplace:
    """Gemini sends complete JSON objects, possibly repeated or growing."""

    def test_identical_resend(self):
        s1 = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert s1.depth == 0

        s2 = _feed_tool_args(s1, '{"city": "Tokyo"}')
        assert s2.args == '{"city": "Tokyo"}'
        assert s2.depth == 0

    def test_progressive_update(self):
        s1 = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert s1.depth == 0

        s2 = _feed_tool_args(s1, '{"city": "Tokyo", "date": "2025-01-01"}')
        assert s2.args == '{"city": "Tokyo", "date": "2025-01-01"}'
        assert s2.depth == 0

    def test_three_resends(self):
        s = _feed_tool_args(None, '{"a": 1}')
        s = _feed_tool_args(s, '{"a": 1, "b": 2}')
        s = _feed_tool_args(s, '{"a": 1, "b": 2, "c": 3}')
        assert s.args == '{"a": 1, "b": 2, "c": 3}'
        assert s.depth == 0


class TestNestedObjects:
    def test_nested_braces_incremental(self):
        s = _feed_tool_args(None, '{"a": {')
        assert s.depth == 2

        s = _feed_tool_args(s, '"b": 1}')
        assert s.depth == 1

        s = _feed_tool_args(s, "}")
        assert s.args == '{"a": {"b": 1}}'
        assert s.depth == 0

    def test_deeply_nested(self):
        obj = '{"a": {"b": {"c": 1}}}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0


class TestBracesInStrings:
    def test_braces_inside_string_value(self):
        obj = '{"template": "{hello}"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0

    def test_braces_in_string_across_fragments(self):
        s = _feed_tool_args(None, '{"t": "{')
        assert s.depth == 1
        assert s.in_string is True

        s = _feed_tool_args(s, 'x}"}')
        assert s.args == '{"t": "{x}"}'
        assert s.depth == 0
        assert s.in_string is False


class TestEscapeSequences:
    def test_escaped_quote_in_value(self):
        r"""Value containing \" — must not close the string."""
        obj = r'{"msg": "say \"hi\""}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_escaped_backslash_before_quote(self):
        r"""Value ending with \\ followed by closing quote."""
        obj = '{"p": "\\\\"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_escaped_backslash_then_escaped_quote(self):
        r"""\\\" inside a string: literal backslash + literal quote."""
        obj = '{"p": "\\\\\\""}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_unicode_escape(self):
        r"""\uXXXX escapes must not confuse the parser."""
        obj = '{"ch": "\\u0022"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_unicode_escape_brace_codepoint(self):
        r"""\u007B is { — must not count as a real brace."""
        obj = '{"ch": "\\u007B"}'
        s = _feed_tool_args(None, obj)
        assert s.depth == 0
        assert s.in_string is False

    def test_escape_spanning_fragments(self):
        r"""Backslash at end of one fragment, escaped char in next."""
        s = _feed_tool_args(None, '{"m": "a\\')
        assert s.in_string is True
        assert s.escape_next is True

        s = _feed_tool_args(s, '"b"}')
        assert s.args == '{"m": "a\\"b"}'
        assert s.depth == 0
        assert s.in_string is False
        assert s.escape_next is False


class TestEdgeCases:
    def test_first_fragment_is_empty_string(self):
        s = _feed_tool_args(None, "")
        assert s.args == ""
        assert s.depth == 0

        s = _feed_tool_args(s, '{"a": 1}')
        assert s.args == '{"a": 1}'
        assert s.depth == 0

    def test_state_none_always_starts_fresh(self):
        s = _feed_tool_args(None, '{"x": 1}')
        assert s.args == '{"x": 1}'
        assert s.depth == 0

    def test_negative_depth_does_not_trigger_replace(self):
        s = _feed_tool_args(None, '{"a": 1}}')
        assert s.depth == -1
        s = _feed_tool_args(s, "extra")
        assert s.args == '{"a": 1}}extra'
        assert s.depth == -1

    def test_boolean_and_null_values(self):
        obj = '{"flag": true, "empty": null, "off": false}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0

    def test_array_values(self):
        obj = '{"items": [1, 2, {"nested": true}]}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0


# ---------------------------------------------------------------------------
# _normalize_messages
# ---------------------------------------------------------------------------


class TestNormalizeMessages:
    """Tests for _normalize_messages which filters empty/invalid messages."""

    def test_empty_user_message_returns_none(self):
        """Empty user message should result in no LLM call."""
        assert _normalize_messages([Message(role="user", content="")]) is None

    def test_whitespace_only_user_message_returns_none(self):
        """Whitespace-only user message should result in no LLM call."""
        assert _normalize_messages([Message(role="user", content="   \n\t  ")]) is None

    def test_empty_user_message_after_valid_conversation_returns_none(self):
        """Empty message following valid conversation should result in no LLM call."""
        result = _normalize_messages(
            [
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello!"),
                Message(role="user", content=""),
            ]
        )
        # Empty user message filtered → ends with assistant → None
        assert result is None

    def test_empty_user_messages_filtered_from_mixed_history(self):
        """Empty user messages in history should be filtered before sending to LLM."""
        result = _normalize_messages(
            [
                Message(role="user", content="First message"),
                Message(role="assistant", content="Response to first"),
                Message(role="user", content=""),  # Empty - should be filtered
                Message(role="user", content="Second message"),
            ]
        )
        assert result is not None
        user_messages = [msg for msg in result if msg.role == "user"]
        for msg in user_messages:
            assert msg.content.strip() != "", f"Found empty user message: {msg}"

    def test_empty_messages_list_returns_none(self):
        """Empty messages list should result in no LLM call."""
        assert _normalize_messages([]) is None

    def test_tool_result_with_empty_content_preserved(self):
        """Tool-result messages with empty content must NOT be filtered."""

        result = _normalize_messages(
            [
                Message(role="user", content="Call the tool"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="my_tool", arguments="{}")],
                ),
                Message(role="tool", content="", tool_call_id="call_1", name="my_tool"),
            ]
        )
        assert result is not None
        tool_msgs = [msg for msg in result if msg.role == "tool"]
        assert len(tool_msgs) == 1

    def test_tool_responses_placed_after_their_tool_call(self):
        """Tool responses must appear immediately after the assistant message that invoked them."""

        result = _normalize_messages(
            [
                Message(role="user", content="Do stuff"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="tool_a", arguments="{}")],
                ),
                Message(role="user", content="Meanwhile"),
                Message(role="tool", content="result_a", tool_call_id="c1", name="tool_a"),
            ]
        )
        assert result is not None
        roles = [m.role for m in result]
        assert roles == ["user", "assistant", "tool", "user"]
        assert result[2].tool_call_id == "c1"

    def test_multiple_tool_responses_reordered_to_follow_assistant(self):
        """Multiple tool responses scattered in the history are grouped after their assistant."""

        result = _normalize_messages(
            [
                Message(role="user", content="Go"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(id="c1", name="t1", arguments="{}"),
                        ToolCall(id="c2", name="t2", arguments="{}"),
                    ],
                ),
                Message(role="user", content="Extra context"),
                Message(role="tool", content="r2", tool_call_id="c2", name="t2"),
                Message(role="tool", content="r1", tool_call_id="c1", name="t1"),
            ]
        )
        assert result is not None
        roles = [m.role for m in result]
        assert roles == ["user", "assistant", "tool", "tool", "user"]
        # Order follows tool_calls order in the assistant message, not input order
        assert result[2].tool_call_id == "c1"
        assert result[3].tool_call_id == "c2"

    def test_tool_responses_already_adjacent_stay_in_place(self):
        """When tool responses are already correctly placed, order is preserved."""

        result = _normalize_messages(
            [
                Message(role="user", content="Hi"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="t1", arguments="{}")],
                ),
                Message(role="tool", content="done", tool_call_id="c1", name="t1"),
                Message(role="user", content="Thanks"),
            ]
        )
        assert result is not None
        roles = [m.role for m in result]
        assert roles == ["user", "assistant", "tool", "user"]

    def test_interleaved_tool_calls_across_multiple_assistants(self):
        """Tool responses from different assistant turns are each placed after their own turn."""

        result = _normalize_messages(
            [
                Message(role="user", content="Start"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="a1", name="ta", arguments="{}")],
                ),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="b1", name="tb", arguments="{}")],
                ),
                Message(role="tool", content="rb", tool_call_id="b1", name="tb"),
                Message(role="tool", content="ra", tool_call_id="a1", name="ta"),
            ]
        )
        assert result is not None
        roles = [m.role for m in result]
        assert roles == ["user", "assistant", "tool", "assistant", "tool"]
        assert result[2].tool_call_id == "a1"
        assert result[4].tool_call_id == "b1"

    def test_duplicate_tool_call_id_preserves_all_responses(self):
        """Multiple tool responses sharing the same tool_call_id are all kept."""

        result = _normalize_messages(
            [
                Message(role="user", content="Go"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="t1", arguments="{}")],
                ),
                Message(role="tool", content="first", tool_call_id="c1", name="t1"),
                Message(role="tool", content="second", tool_call_id="c1", name="t1"),
            ]
        )
        assert result is not None
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == "first"
        assert tool_msgs[1].content == "second"


# ---------------------------------------------------------------------------
# Gemini thought_signature / provider_specific_fields (http_provider.py)
# ---------------------------------------------------------------------------


class TestThoughtSignatureStreaming:
    """Verify that thought_signature is captured from provider_specific_fields
    during streaming and round-tripped through message building."""

    def _make_chunk(self, tool_calls_delta=None, content=None, finish_reason=None):
        """Build a minimal litellm-style streaming chunk."""

        class _Delta:
            pass

        class _Choice:
            pass

        class _Chunk:
            pass

        delta = _Delta()
        delta.content = content
        delta.tool_calls = tool_calls_delta

        choice = _Choice()
        choice.delta = delta
        choice.finish_reason = finish_reason

        chunk = _Chunk()
        chunk.choices = [choice]
        return chunk

    def _make_tc_delta(self, index, id=None, name=None, arguments=None, thought_signature=None):
        """Build a minimal tool_call delta object with optional provider_specific_fields."""

        class _Func:
            pass

        class _TC:
            pass

        tc = _TC()
        tc.index = index
        tc.id = id
        tc.function = _Func() if (name or arguments) else None
        if tc.function:
            tc.function.name = name
            tc.function.arguments = arguments
        if thought_signature:
            tc.provider_specific_fields = {"thought_signature": thought_signature}
        else:
            tc.provider_specific_fields = None
        return tc

    def test_thought_signature_captured_from_stream_chunk(self, monkeypatch):
        """provider_specific_fields.thought_signature on a tool_call delta
        should end up on the yielded ToolCall."""

        chunks = [
            self._make_chunk(tool_calls_delta=[
                self._make_tc_delta(0, id="call_1", name="my_tool", arguments='{"a": 1}',
                                    thought_signature="sig_abc"),
            ]),
            self._make_chunk(finish_reason="tool_calls"),
        ]

        async def _fake_acompletion(**kwargs):
            class _Response:
                async def __aiter__(self_inner):
                    for c in chunks:
                        yield c

                async def aclose(self_inner):
                    pass

            return _Response()

        monkeypatch.setattr("line.llm_agent.http_provider.acompletion", _fake_acompletion)

        provider = _HttpProvider(model="gemini/gemini-2.5-flash")

        collected = []

        async def _consume():
            async for chunk in provider.chat(
                [Message(role="user", content="hi")],
                config=_normalize_config(LlmConfig()),
            ):
                collected.append(chunk)

        asyncio.run(_consume())

        all_tcs = [tc for chunk in collected for tc in chunk.tool_calls]
        assert any(tc.thought_signature == "sig_abc" for tc in all_tcs)

    def test_thought_signature_none_when_no_provider_fields(self, monkeypatch):
        """When provider_specific_fields is absent, thought_signature stays None."""

        chunks = [
            self._make_chunk(tool_calls_delta=[
                self._make_tc_delta(0, id="call_1", name="my_tool", arguments='{"a": 1}'),
            ]),
            self._make_chunk(finish_reason="tool_calls"),
        ]

        async def _fake_acompletion(**kwargs):
            class _Response:
                async def __aiter__(self_inner):
                    for c in chunks:
                        yield c

                async def aclose(self_inner):
                    pass

            return _Response()

        monkeypatch.setattr("line.llm_agent.http_provider.acompletion", _fake_acompletion)

        provider = _HttpProvider(model="openai/gpt-4o")

        collected = []

        async def _consume():
            async for chunk in provider.chat(
                [Message(role="user", content="hi")],
                config=_normalize_config(LlmConfig()),
            ):
                collected.append(chunk)

        asyncio.run(_consume())

        all_tcs = [tc for chunk in collected for tc in chunk.tool_calls]
        assert all(tc.thought_signature is None for tc in all_tcs)

    def test_thought_signature_round_trips_through_message_building(self):
        """ToolCall.thought_signature should appear in provider_specific_fields
        when _build_messages serializes tool calls."""

        provider = _HttpProvider(model="gemini/gemini-2.5-flash")
        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="my_tool", arguments='{"x": 1}',
                             thought_signature="sig_xyz"),
                ],
            ),
            Message(role="tool", content="result", tool_call_id="call_1", name="my_tool"),
        ]

        built = provider._build_messages(messages, _normalize_config(LlmConfig()))
        assistant_msg = built[0]
        tc_dict = assistant_msg["tool_calls"][0]
        assert tc_dict["provider_specific_fields"] == {"thought_signature": "sig_xyz"}

    def test_no_provider_specific_fields_when_no_thought_signature(self):
        """Without thought_signature, provider_specific_fields key should be absent."""

        provider = _HttpProvider(model="openai/gpt-4o")
        messages = [
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="my_tool", arguments='{"x": 1}'),
                ],
            ),
            Message(role="tool", content="result", tool_call_id="call_1", name="my_tool"),
        ]

        built = provider._build_messages(messages, _normalize_config(LlmConfig()))
        tc_dict = built[0]["tool_calls"][0]
        assert "provider_specific_fields" not in tc_dict


# ---------------------------------------------------------------------------
# get_supported_openai_params usage (provider.py:399-416)
# ---------------------------------------------------------------------------


class TestGetSupportedOpenaiParamsUsage:
    """Verify that get_supported_openai_params drives backend selection and
    reasoning_effort detection correctly."""

    def test_model_with_reasoning_in_supported_params(self, monkeypatch):
        """When get_supported_openai_params includes 'reasoning_effort',
        the config should report supports_reasoning_effort=True."""
        import litellm
        import litellm.utils

        monkeypatch.setattr(
            litellm, "get_supported_openai_params",
            lambda model: ["temperature", "reasoning_effort", "max_tokens"],
        )
        # Also need to patch get_optional_params to not raise
        monkeypatch.setattr(
            litellm, "get_llm_provider",
            lambda model: (model, "test_provider", None, None),
        )
        monkeypatch.setattr(
            litellm.utils, "get_optional_params",
            lambda model, custom_llm_provider, reasoning_effort: {},
        )

        cfg = _get_model_config("test-provider/test-reasoning-model")
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort == "none"

    def test_model_without_reasoning_in_supported_params(self, monkeypatch):
        """When get_supported_openai_params lacks 'reasoning_effort',
        the config should report supports_reasoning_effort=False."""
        import litellm

        monkeypatch.setattr(
            litellm, "get_supported_openai_params",
            lambda model: ["temperature", "max_tokens"],
        )

        cfg = _get_model_config("test-provider/test-no-reasoning-model")
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_get_optional_params_raises_falls_back_for_anthropic(self, monkeypatch):
        """When get_optional_params raises for an Anthropic model,
        default_reasoning_effort should be None (the Anthropic hack)."""
        import litellm
        import litellm.utils

        monkeypatch.setattr(
            litellm, "get_supported_openai_params",
            lambda model: ["reasoning_effort", "temperature"],
        )

        def _raise(*a, **kw):
            raise Exception("unsupported value")

        monkeypatch.setattr(litellm, "get_llm_provider", lambda model: (model, "anthropic", None, None))
        monkeypatch.setattr(litellm.utils, "get_optional_params", _raise)

        cfg = _get_model_config("anthropic/claude-test-model")
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort is None

    def test_get_optional_params_raises_non_anthropic_gets_low(self, monkeypatch):
        """When get_optional_params raises for a non-Anthropic model,
        default_reasoning_effort should be 'low'."""
        import litellm
        import litellm.utils

        monkeypatch.setattr(
            litellm, "get_supported_openai_params",
            lambda model: ["reasoning_effort", "temperature"],
        )

        def _raise(*a, **kw):
            raise Exception("unsupported value")

        monkeypatch.setattr(litellm, "get_llm_provider", lambda model: (model, "openai", None, None))
        monkeypatch.setattr(litellm.utils, "get_optional_params", _raise)

        cfg = _get_model_config("openai/some-reasoning-model")
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort == "low"

    def test_get_supported_params_returns_none_raises(self, monkeypatch):
        """When get_supported_openai_params returns None, model is rejected."""
        import litellm

        monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)

        try:
            _get_model_config("fake/unsupported-model")
        except ValueError as exc:
            assert "is not supported" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


# ---------------------------------------------------------------------------
# Schema converter (schema_converter.py)
# ---------------------------------------------------------------------------


class TestSchemaConverter:
    """Verify tool schema conversion to litellm format produces valid output."""

    def _make_tool(self, name, description, params):
        """Build a FunctionTool with the given parameters."""
        from line.llm_agent.tools.utils import FunctionTool, ParameterInfo, ToolType

        return FunctionTool(
            name=name,
            description=description,
            func=lambda ctx: None,
            parameters=params,
            tool_type=ToolType.LOOPBACK,
        )

    def test_basic_string_param_tool(self):
        from line.llm_agent.schema_converter import function_tool_to_litellm
        from line.llm_agent.tools.utils import ParameterInfo

        tool = self._make_tool("greet", "Say hello", {
            "name": ParameterInfo(name="name", type_annotation=str, description="Name", required=True),
        })
        result = function_tool_to_litellm(tool)
        assert result["type"] == "function"
        fn = result["function"]
        assert fn["name"] == "greet"
        assert fn["description"] == "Say hello"
        assert fn["parameters"]["properties"]["name"]["type"] == "string"
        assert "name" in fn["parameters"]["required"]
        assert fn["strict"] is True
        assert fn["parameters"]["additionalProperties"] is False

    def test_multiple_param_types(self):
        from line.llm_agent.schema_converter import function_tool_to_litellm
        from line.llm_agent.tools.utils import ParameterInfo

        tool = self._make_tool("search", "Search items", {
            "query": ParameterInfo(name="query", type_annotation=str, description="Query", required=True),
            "limit": ParameterInfo(name="limit", type_annotation=int, description="Max results", required=True),
            "fuzzy": ParameterInfo(name="fuzzy", type_annotation=bool, description="Fuzzy match", required=True),
        })
        result = function_tool_to_litellm(tool)
        props = result["function"]["parameters"]["properties"]
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        assert props["fuzzy"]["type"] == "boolean"

    def test_optional_param_disables_strict(self):
        from line.llm_agent.schema_converter import function_tool_to_litellm
        from line.llm_agent.tools.utils import ParameterInfo

        tool = self._make_tool("opt_tool", "Tool with optional", {
            "name": ParameterInfo(name="name", type_annotation=str, description="Name", required=True),
            "tag": ParameterInfo(name="tag", type_annotation=str, description="Tag",
                                 required=False, default="default"),
        })
        result = function_tool_to_litellm(tool)
        assert "strict" not in result["function"]

    def test_list_param(self):
        from line.llm_agent.schema_converter import function_tool_to_litellm
        from line.llm_agent.tools.utils import ParameterInfo

        tool = self._make_tool("batch", "Batch op", {
            "ids": ParameterInfo(name="ids", type_annotation=List[int], description="IDs", required=True),
        })
        result = function_tool_to_litellm(tool)
        prop = result["function"]["parameters"]["properties"]["ids"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "integer"

    def test_tools_to_litellm_converts_list(self):
        from line.llm_agent.schema_converter import tools_to_litellm
        from line.llm_agent.tools.utils import ParameterInfo

        tools = [
            self._make_tool("a", "Tool A", {
                "x": ParameterInfo(name="x", type_annotation=str, description="X", required=True),
            }),
            self._make_tool("b", "Tool B", {
                "y": ParameterInfo(name="y", type_annotation=int, description="Y", required=True),
            }),
        ]
        result = tools_to_litellm(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_bare_dict_raises_in_strict_mode(self):
        from line.llm_agent.schema_converter import python_type_to_json_schema

        try:
            python_type_to_json_schema(dict, strict=True)
        except ValueError as exc:
            assert "TypedDict" in str(exc)
        else:
            raise AssertionError("Expected ValueError for bare dict in strict mode")

    def test_bare_dict_allowed_in_non_strict_mode(self):
        from line.llm_agent.schema_converter import python_type_to_json_schema

        result = python_type_to_json_schema(dict, strict=False)
        assert result == {"type": "object"}

    def test_literal_enum_values(self):
        from typing import Literal

        from line.llm_agent.schema_converter import python_type_to_json_schema

        result = python_type_to_json_schema(Literal["red", "green", "blue"])
        assert result == {"type": "string", "enum": ["red", "green", "blue"]}

    def test_typeddict_generates_object_schema(self):
        from typing import TypedDict

        from line.llm_agent.schema_converter import python_type_to_json_schema

        class Item(TypedDict):
            name: str
            quantity: int

        result = python_type_to_json_schema(Item, strict=True)
        assert result["type"] == "object"
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["quantity"]["type"] == "integer"
        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"name", "quantity"}

    def test_function_tool_to_litellm_output_format(self):
        """Verify the top-level shape matches what litellm expects:
        {"type": "function", "function": {"name": ..., "parameters": ...}}"""
        from line.llm_agent.schema_converter import function_tool_to_litellm
        from line.llm_agent.tools.utils import ParameterInfo

        tool = self._make_tool("test", "Test tool", {
            "arg": ParameterInfo(name="arg", type_annotation=str, description="Arg", required=True),
        })
        result = function_tool_to_litellm(tool)
        assert set(result.keys()) == {"type", "function"}
        assert result["type"] == "function"
        assert "name" in result["function"]
        assert "description" in result["function"]
        assert "parameters" in result["function"]
