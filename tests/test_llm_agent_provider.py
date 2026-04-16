"""Tests for the LlmProvider facade and HTTP backend."""

import asyncio
from typing import Annotated, Any, List, Optional, get_type_hints

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.http_provider import _feed_tool_args, _HttpProvider
from line.llm_agent.provider import (
    ChatStream,
    LlmProvider,
    Message,
    ParsedModelId,
    ToolCall,
    _extract_instructions_and_messages,
    _get_model_config,
    _is_realtime_model,
    _is_websocket_model,
    _normalize_messages,
    parse_model_id,
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


def test_llm_provider_routes_temperature_to_http_backend():
    """temperature/top_p cause the OpenAI WebSocket endpoint to close silently,
    so they must route to the HTTP fallback."""
    provider = LlmProvider(
        model="openai/gpt-5.2",
        api_key="test-key",
    )
    websocket_backend = _DummyBackend()
    http_backend = _DummyBackend()
    provider._backend = websocket_backend
    provider._http_fallback_backend = http_backend

    provider.chat(
        [Message(role="user", content="hi")],
        config=LlmConfig(temperature=0.2, top_p=0.9),
    )

    assert len(websocket_backend.calls) == 0
    assert len(http_backend.calls) == 1


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
    assert _get_model_config(parse_model_id("openai/gpt-5.2")) is not None


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
        cfg = _get_model_config(parse_model_id("openai/gpt-4o-realtime-preview"))
        assert cfg.backend == "realtime"
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_websocket_model_selects_websocket_backend(self):
        cfg = _get_model_config(parse_model_id("openai/gpt-5.2"))
        assert cfg.backend == "websocket"

    def test_websocket_model_with_http_override_selects_http(self):
        cfg = _get_model_config(parse_model_id("openai/gpt-5.2"), backend="http")
        assert cfg.backend == "http"

    def test_websocket_model_with_explicit_websocket_backend(self):
        cfg = _get_model_config(parse_model_id("openai/gpt-5.2"), backend="websocket")
        assert cfg.backend == "websocket"

    def test_plain_http_model_selects_http_backend(self):
        cfg = _get_model_config(parse_model_id("openai/gpt-4o"))
        assert cfg.backend == "http"

    # -- Error cases -----------------------------------------------------------

    def test_invalid_backend_raises(self):
        try:
            _get_model_config(parse_model_id("openai/gpt-4o"), backend="banana")
        except ValueError as exc:
            assert "Invalid backend" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    def test_realtime_backend_with_non_realtime_model_raises(self):
        try:
            _get_model_config(parse_model_id("openai/gpt-4o"), backend="realtime")
        except ValueError as exc:
            assert "realtime" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError")

    def test_non_realtime_backend_with_realtime_model_raises(self):
        try:
            _get_model_config(parse_model_id("openai/gpt-4o-realtime-preview"), backend="http")
        except ValueError as exc:
            assert "incompatible" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError")

    def test_websocket_backend_with_non_websocket_model_raises(self):
        try:
            _get_model_config(parse_model_id("openai/gpt-4o"), backend="websocket")
        except ValueError as exc:
            assert "websocket" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError")

    def test_unsupported_model_raises(self, monkeypatch):
        import litellm

        monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)
        try:
            _get_model_config(parse_model_id("fake-provider/not-a-model"))
        except ValueError as exc:
            assert "is not supported" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    # -- Reasoning effort for websocket models ---------------------------------

    def test_websocket_model_with_reasoning_support(self):
        """Reasoning models (e.g. o3) routed via websocket get reasoning enabled."""
        cfg = _get_model_config(parse_model_id("openai/gpt-5.2"))
        assert cfg.backend == "websocket"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort == "low"

    # -- Reasoning effort for HTTP models --------------------------------------

    def test_http_model_without_reasoning_support(self):
        cfg = _get_model_config(parse_model_id("openai/gpt-4o"))
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_http_model_with_reasoning_support(self):
        """o4-mini goes through the HTTP path and supports reasoning."""
        cfg = _get_model_config(parse_model_id("openai/o4-mini"))
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort is not None

    def test_websocket_model_forced_to_http_gets_correct_reasoning(self):
        """gpt-4.1 forced to HTTP should still have reasoning disabled."""
        cfg = _get_model_config(parse_model_id("openai/gpt-4.1"), backend="http")
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is False
        assert cfg.default_reasoning_effort is None

    def test_anthropic_model_reasoning_default_is_none(self):
        """Anthropic models use None (not 'low') to skip the thinking block."""
        cfg = _get_model_config(parse_model_id("anthropic/claude-sonnet-4-20250514"))
        assert cfg.backend == "http"
        assert cfg.supports_reasoning_effort is True
        assert cfg.default_reasoning_effort is None


def test_is_websocket_model_matches_gpt52_variants():
    assert _is_websocket_model(parse_model_id("gpt-5.2"))
    assert _is_websocket_model(parse_model_id("gpt-5.2-pro"))
    assert _is_websocket_model(parse_model_id("gpt-5.4"))
    assert _is_websocket_model(parse_model_id("gpt-5.4-mini"))
    assert _is_websocket_model(parse_model_id("chatgpt/gpt-5.4-pro"))
    assert not _is_websocket_model(parse_model_id("azure/gpt-5.2"))
    assert not _is_websocket_model(parse_model_id("openrouter/gpt-5.2"))
    assert not _is_websocket_model(parse_model_id("openai/gpt-4o"))


def test_is_realtime_model_matches_only_direct_openai_models():
    assert _is_realtime_model(parse_model_id("openai/gpt-4o-realtime-preview"))
    assert _is_realtime_model(parse_model_id("gpt-4o-realtime-preview"))
    assert not _is_realtime_model(parse_model_id("azure/gpt-4o-realtime-preview"))
    assert not _is_realtime_model(parse_model_id("openrouter/gpt-4o-realtime-preview"))


class TestParseModelId:
    """Tests for parse_model_id normalization."""

    def test_bare_gpt_string_defaults_to_openai(self):
        assert parse_model_id("gpt-4o") == ParsedModelId("openai", "gpt-4o")

    def test_openai_prefix_preserved(self):
        assert parse_model_id("openai/gpt-4o") == ParsedModelId("openai", "gpt-4o")

    def test_chatgpt_prefix_normalized_to_openai(self):
        assert parse_model_id("chatgpt/gpt-5.4-pro") == ParsedModelId("openai", "gpt-5.4-pro")

    def test_anthropic_prefix_preserved(self):
        assert parse_model_id("anthropic/claude-sonnet-4-5") == ParsedModelId(
            "anthropic", "claude-sonnet-4-5"
        )

    def test_gemini_prefix_preserved(self):
        assert parse_model_id("gemini/gemini-2.5-flash") == ParsedModelId(
            "gemini", "gemini-2.5-flash"
        )

    def test_provider_lowercased(self):
        assert parse_model_id("OpenAI/gpt-4o") == ParsedModelId("openai", "gpt-4o")

    def test_str_round_trips_to_canonical_form(self):
        assert str(parse_model_id("chatgpt/gpt-5.2")) == "openai/gpt-5.2"
        assert str(parse_model_id("gpt-4o")) == "openai/gpt-4o"


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

    provider = _HttpProvider(model_id=parse_model_id("gpt-4o"))

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
