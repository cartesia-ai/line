"""Tests for provider internals and facade behavior."""

import asyncio
from typing import Annotated, Any, AsyncIterable, List, Optional, get_type_hints

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.http_provider import _HttpProvider, _feed_tool_args
from line.llm_agent.provider import LlmProvider, Message, StreamChunk
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
    import litellm

    monkeypatch.setattr(litellm, "supports_web_search", lambda model: True)

    provider = LlmProvider(
        model="gpt-4o",
        api_key="test-key",
        tools=[web_search(search_context_size="high")],
    )
    backend = _DummyBackend()
    provider._backend = backend

    asyncio.run(provider.warmup())

    assert len(backend.warmup_calls) == 1
    assert backend.warmup_calls[0]["tools"] == []
    assert backend.warmup_calls[0]["kwargs"]["web_search_options"] == {"search_context_size": "high"}


def test_llm_provider_requires_api_key():
    try:
        LlmProvider(model="gpt-4o", api_key="")
    except ValueError as exc:
        assert "Missing API key" in str(exc)
    else:
        raise AssertionError("Expected LlmProvider to reject missing api_key")


def test_llm_provider_public_methods_have_type_hints():
    chat_hints = get_type_hints(LlmProvider.chat)
    assert chat_hints["messages"] == List[Message]
    assert chat_hints["tools"] == Optional[List[Any]]
    assert chat_hints["config"] == Optional[LlmConfig]
    assert chat_hints["kwargs"] is Any
    assert chat_hints["return"] == AsyncIterable[StreamChunk]

    warmup_hints = get_type_hints(LlmProvider.warmup)
    assert warmup_hints["config"] == Optional[LlmConfig]
    assert warmup_hints["tools"] == Optional[List[Any]]
    assert warmup_hints["return"] is type(None)

    aclose_hints = get_type_hints(LlmProvider.aclose)
    assert aclose_hints["return"] is type(None)


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
# OpenAI / Anthropic: incremental fragments → concatenate
# ---------------------------------------------------------------------------


class TestIncrementalConcatenation:
    """OpenAI and Anthropic stream arguments as small JSON fragments."""

    def test_single_fragment_complete_object(self):
        """A single fragment that is already a complete JSON object."""
        state = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert state.args == '{"city": "Tokyo"}'
        assert state.depth == 0

    def test_two_fragments(self):
        """Two incremental fragments that together form a complete object."""
        s1 = _feed_tool_args(None, '{"ci')
        assert s1.args == '{"ci'
        assert s1.depth == 1

        s2 = _feed_tool_args(s1, 'ty": "Tokyo"}')
        assert s2.args == '{"city": "Tokyo"}'
        assert s2.depth == 0

    def test_many_small_fragments(self):
        """Simulates typical OpenAI streaming with many tiny fragments."""
        fragments = ["{", '"name"', ": ", '"', "Alice", '"', ", ", '"age"', ": ", "30", "}"]
        state = None
        for frag in fragments:
            state = _feed_tool_args(state, frag)

        assert state.args == '{"name": "Alice", "age": 30}'
        assert state.depth == 0
        assert state.in_string is False
        assert state.escape_next is False

    def test_empty_object(self):
        """Tool with no parameters: {}."""
        s1 = _feed_tool_args(None, "{")
        assert s1.depth == 1

        s2 = _feed_tool_args(s1, "}")
        assert s2.args == "{}"
        assert s2.depth == 0


# ---------------------------------------------------------------------------
# Gemini: complete objects repeated → replace
# ---------------------------------------------------------------------------


class TestGeminiReplace:
    """Gemini sends complete JSON objects, possibly repeated or growing."""

    def test_identical_resend(self):
        """Same complete object sent twice — should replace, not double."""
        s1 = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert s1.depth == 0

        s2 = _feed_tool_args(s1, '{"city": "Tokyo"}')
        assert s2.args == '{"city": "Tokyo"}'
        assert s2.depth == 0

    def test_progressive_update(self):
        """Gemini sends progressively larger complete objects."""
        s1 = _feed_tool_args(None, '{"city": "Tokyo"}')
        assert s1.depth == 0

        s2 = _feed_tool_args(s1, '{"city": "Tokyo", "date": "2025-01-01"}')
        assert s2.args == '{"city": "Tokyo", "date": "2025-01-01"}'
        assert s2.depth == 0

    def test_three_resends(self):
        """Multiple consecutive replacements."""
        s = _feed_tool_args(None, '{"a": 1}')
        s = _feed_tool_args(s, '{"a": 1, "b": 2}')
        s = _feed_tool_args(s, '{"a": 1, "b": 2, "c": 3}')
        assert s.args == '{"a": 1, "b": 2, "c": 3}'
        assert s.depth == 0


# ---------------------------------------------------------------------------
# Nested objects
# ---------------------------------------------------------------------------


class TestNestedObjects:
    def test_nested_braces_incremental(self):
        """Nested objects tracked correctly across incremental fragments."""
        s = _feed_tool_args(None, '{"a": {')
        assert s.depth == 2

        s = _feed_tool_args(s, '"b": 1}')
        assert s.depth == 1

        s = _feed_tool_args(s, "}")
        assert s.args == '{"a": {"b": 1}}'
        assert s.depth == 0

    def test_deeply_nested(self):
        """Three levels of nesting in one fragment."""
        obj = '{"a": {"b": {"c": 1}}}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0


# ---------------------------------------------------------------------------
# Braces and special chars inside strings
# ---------------------------------------------------------------------------


class TestBracesInStrings:
    def test_braces_inside_string_value(self):
        """Braces inside a JSON string must not affect depth tracking."""
        obj = '{"template": "{hello}"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0

    def test_braces_in_string_across_fragments(self):
        s = _feed_tool_args(None, '{"t": "{')
        assert s.depth == 1  # only the outer { counts
        assert s.in_string is True

        s = _feed_tool_args(s, 'x}"}')
        assert s.args == '{"t": "{x}"}'
        assert s.depth == 0
        assert s.in_string is False


# ---------------------------------------------------------------------------
# Escape sequences
# ---------------------------------------------------------------------------


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
        # JSON: {"p": "\\"} — the value is a single backslash
        obj = '{"p": "\\\\"}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_escaped_backslash_then_escaped_quote(self):
        r"""\\\" inside a string: literal backslash + literal quote."""
        # JSON: {"p": "\\\""} — the value is \"
        obj = '{"p": "\\\\\\""}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
        assert s.in_string is False

    def test_unicode_escape(self):
        r"""\uXXXX escapes must not confuse the parser."""
        obj = '{"ch": "\\u0022"}'  # \u0022 is "
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


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_first_fragment_is_empty_string(self):
        """Empty first fragment should be a no-op, next fragment concatenates."""
        s = _feed_tool_args(None, "")
        assert s.args == ""
        assert s.depth == 0

        # Next real fragment should concatenate (args is falsy → else branch)
        s = _feed_tool_args(s, '{"a": 1}')
        assert s.args == '{"a": 1}'
        assert s.depth == 0

    def test_state_none_always_starts_fresh(self):
        s = _feed_tool_args(None, '{"x": 1}')
        assert s.args == '{"x": 1}'
        assert s.depth == 0

    def test_negative_depth_does_not_trigger_replace(self):
        """Malformed JSON with extra } — depth goes negative, no false replace."""
        s = _feed_tool_args(None, '{"a": 1}}')
        assert s.depth == -1
        # Next fragment should concatenate (depth != 0)
        s = _feed_tool_args(s, "extra")
        assert s.args == '{"a": 1}}extra'
        assert s.depth == -1

    def test_boolean_and_null_values(self):
        obj = '{"flag": true, "empty": null, "off": false}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0

    def test_array_values(self):
        """Arrays inside object values — [ ] should not affect brace depth."""
        obj = '{"items": [1, 2, {"nested": true}]}'
        s = _feed_tool_args(None, obj)
        assert s.args == obj
        assert s.depth == 0
