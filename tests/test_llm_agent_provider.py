"""Tests for provider internals and facade behavior."""

import asyncio
from typing import Annotated, Any, AsyncIterable, List, Optional, get_type_hints

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.http_provider import _feed_tool_args, _HttpProvider
from line.llm_agent.provider import (
    LlmProvider,
    Message,
    StreamChunk,
    ToolCall,
    _extract_instructions_and_messages,
    _get_model_config,
    _is_realtime_model,
    _is_websocket_model,
)
from line.llm_agent.realtime_provider import (
    _diff_messages,
    _SessionUpdateOp,
    _track_output_item,
)
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.stream import _compute_divergence, _context_identity, _expand_messages
from line.llm_agent.tools.system import web_search
from line.llm_agent.websocket_provider import (
    _build_request,
    _extract_model_output_identities,
    _WebSocketProvider,
)


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
        model="gpt-5-mini",
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
        model="gpt-5-mini",
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
        model="gpt-5-mini",
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
        model="gpt-5-mini",
        api_key="test-key",
    )
    websocket_backend = _DummyBackend()
    http_backend = _DummyBackend()
    provider._backend = websocket_backend
    provider._http_fallback_backend = http_backend

    asyncio.run(provider.warmup(config=LlmConfig(extra={"service_tier": "flex"})))

    assert len(websocket_backend.warmup_calls) == 0
    assert len(http_backend.warmup_calls) == 1


def test_realtime_diff_updates_session_for_sampling_changes():
    state = [(_context_identity("stay concise", None, temperature=0.2, max_tokens=100), None)]
    messages = [Message(role="system", content="stay concise"), Message(role="user", content="hi")]

    ops = _diff_messages(
        state,
        messages,
        tools=None,
        config=LlmConfig(temperature=0.9, max_tokens=200),
    )

    assert isinstance(ops[0], _SessionUpdateOp)
    assert ops[0].instructions == "stay concise"
    assert ops[0].tools is None
    assert ops[0].temperature == 0.9
    assert ops[0].max_tokens == 200


def test_realtime_diff_clears_session_fields():
    state = [
        (
            _context_identity(
                "old instructions",
                [{"type": "function", "name": "old_tool", "parameters": {"type": "object"}}],
                temperature=0.7,
                max_tokens=400,
            ),
            None,
        )
    ]
    messages = [Message(role="user", content="hi")]

    ops = _diff_messages(
        state,
        messages,
        tools=None,
        config=LlmConfig(temperature=None, max_tokens=None),
    )

    assert isinstance(ops[0], _SessionUpdateOp)
    assert ops[0].instructions is None
    assert ops[0].tools is None
    assert ops[0].temperature is None
    assert ops[0].max_tokens is None


def test_extract_instructions_includes_config_system_prompt():
    instructions, non_system = _extract_instructions_and_messages(
        [Message(role="system", content="call-specific"), Message(role="user", content="hi")],
        LlmConfig(system_prompt="base"),
    )

    assert instructions == "base\n\ncall-specific"
    assert len(non_system) == 1
    assert non_system[0].role == "user"


def test_realtime_diff_uses_config_system_prompt_without_system_message():
    ops = _diff_messages(
        [],
        [Message(role="user", content="hi")],
        tools=None,
        config=LlmConfig(system_prompt="stay concise"),
    )

    assert isinstance(ops[0], _SessionUpdateOp)
    assert ops[0].instructions == "stay concise"


def test_realtime_diff_includes_native_web_search_tool():
    ops = _diff_messages(
        [],
        [Message(role="user", content="hi")],
        tools=None,
        config=LlmConfig(),
        web_search_options={"search_context_size": "high"},
    )

    assert isinstance(ops[0], _SessionUpdateOp)
    assert ops[0].tools == [{"type": "web_search", "name": "web_search", "search_context_size": "high"}]


def test_build_openai_tool_defs_adds_native_web_search():
    assert build_openai_tool_defs(
        web_search_options={"search_context_size": "high"},
        responses_api=True,
    ) == [{"type": "web_search", "name": "web_search", "search_context_size": "high"}]


def test_llm_provider_requires_api_key():
    try:
        LlmProvider(model="gpt-4o", api_key="")
    except ValueError as exc:
        assert "Missing API key" in str(exc)
    else:
        raise AssertionError("Expected LlmProvider to reject missing api_key")


def test_llm_provider_rejects_unsupported_model(monkeypatch):
    monkeypatch.setattr(
        "line.llm_agent.provider._get_model_config",
        lambda model: None,
    )

    try:
        LlmProvider(model="definitely-not-a-real-model", api_key="test-key")
    except ValueError as exc:
        assert "is not supported" in str(exc)
    else:
        raise AssertionError("Expected LlmProvider to reject unsupported model")


def test_is_supported_model_accepts_direct_openai_websocket_model(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)
    assert _get_model_config("gpt-5-mini") is not None


def test_websocket_extract_model_output_identities_reads_output_text():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "output_text", "text": ", world!"},
                ],
            }
        ]
    }

    assert _extract_model_output_identities(response) == [("assistant", "Hello, world!", "", "")]


def test_expand_messages_preserves_assistant_text_before_tool_call_for_responses_api():
    message = Message(
        role="assistant",
        content="Let me check. ",
        tool_calls=[
            ToolCall(id="call_1", name="get_weather", arguments='{"city":"NYC"}'),
        ],
    )

    assert _expand_messages([message], assistant_text_type="output_text") == [
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check. "}],
            },
            ("assistant", "Let me check. ", "", ""),
        ),
        (
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
            },
            ("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)),
        ),
    ]


def test_compute_divergence_preserves_assistant_text_before_tool_call():
    message = Message(
        role="assistant",
        content="Let me check. ",
        tool_calls=[
            ToolCall(id="call_1", name="get_weather", arguments='{"city":"NYC"}'),
        ],
    )
    desired_pairs = _expand_messages([message], assistant_text_type="output_text")
    prefix_len, after = _compute_divergence(
        current_identities=[("assistant", "Let me check. ", "", "")],
        desired_pairs=desired_pairs,
    )

    assert after == [
        (
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
            },
            ("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)),
        )
    ]
    assert prefix_len == 1


def test_websocket_finalize_response_preserves_text_then_tool_call_outputs():
    provider = _WebSocketProvider(model="gpt-5-mini")
    context_id = ("__context__", "", ())
    provider._history = [(context_id, "warmup")]

    response = {
        "id": "resp_1",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check. "}],
            },
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
                "call_id": "call_1",
            },
        ],
    }

    provider._finalize_response(
        response=response,
        continuation_idx=1,
        desired_ids=[context_id, ("user", "Weather?", "", "")],
    )

    assert provider._history == [
        (context_id, "warmup"),
        (("user", "Weather?", "", ""), None),
        (("assistant", "Let me check. ", "", ""), "resp_1"),
        (("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)), "resp_1"),
    ]


def test_realtime_track_output_item_preserves_text_then_tool_call_outputs():
    history = []

    _track_output_item(
        history,
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me check. "}],
        },
    )
    _track_output_item(
        history,
        {
            "id": "fc_1",
            "type": "function_call",
            "name": "get_weather",
            "arguments": '{"city":"NYC"}',
            "call_id": "call_1",
        },
    )

    assert [identity for identity, _ in history] == [
        ("assistant", "Let me check. ", "", ""),
        ("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)),
    ]


def test_expand_messages_preserves_assistant_text_before_tool_call_for_realtime():
    message = Message(
        role="assistant",
        content="Let me check. ",
        tool_calls=[
            ToolCall(id="call_1", name="get_weather", arguments='{"city":"NYC"}'),
        ],
    )

    assert _expand_messages([message], assistant_text_type="text") == [
        (
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Let me check. "}],
            },
            ("assistant", "Let me check. ", "", ""),
        ),
        (
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
            },
            ("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)),
        ),
    ]


def test_websocket_build_request_strips_openai_prefix_from_model():
    request = _build_request(
        model="openai/gpt-5-mini",
        default_reasoning_effort="none",
        instructions=None,
        tool_defs=None,
        cfg=LlmConfig(),
    )

    assert request["model"] == "gpt-5-mini"


def test_websocket_context_identity_accepts_native_web_search_tool():
    tool_defs = build_openai_tool_defs(
        web_search_options={"search_context_size": "high"},
        responses_api=True,
    )

    assert _context_identity(None, tool_defs, temperature=None, max_tokens=None) == (
        "__context__",
        "",
        ('{"name": "web_search", "search_context_size": "high", "type": "web_search"}',),
        None,
        None,
    )


def test_websocket_context_identity_changes_when_native_web_search_options_change():
    high = build_openai_tool_defs(
        web_search_options={"search_context_size": "high"},
        responses_api=True,
    )
    low = build_openai_tool_defs(
        web_search_options={"search_context_size": "low"},
        responses_api=True,
    )

    assert _context_identity(None, high, temperature=None, max_tokens=None) != _context_identity(
        None, low, temperature=None, max_tokens=None
    )


def test_websocket_chat_retries_previous_response_not_found():
    provider = _WebSocketProvider(model="gpt-5-mini")
    attempts = 0

    class _FakeStream:
        def __init__(self, *, fail: bool):
            self._fail = fail
            self.done = False

        async def __aiter__(self):
            self.done = True
            if self._fail:
                provider._reset_response_state()
                raise RuntimeError(
                    "OpenAI API error (previous_response_not_found): previous response was not found"
                )
            yield StreamChunk(text="hello")
            yield StreamChunk(is_final=True)

    async def _fake_setup():
        nonlocal attempts
        attempts += 1
        lock = asyncio.Lock()
        await lock.acquire()
        return _FakeStream(fail=attempts == 1), object(), lock

    provider._setup_chat = lambda *args, **kwargs: _fake_setup()
    chunks: List[StreamChunk] = []

    async def _consume():
        async for chunk in provider.chat(
            [Message(role="user", content="hi")],
            config=LlmConfig(),
        ):
            chunks.append(chunk)

    asyncio.run(_consume())

    assert attempts == 2
    assert [chunk.text for chunk in chunks if chunk.text] == ["hello"]


def test_is_websocket_model_matches_all_gpt5_variants():
    assert _is_websocket_model("gpt-5-nano")
    assert _is_websocket_model("openai/gpt-5-mini")
    assert _is_websocket_model("gpt-5.2")
    assert _is_websocket_model("gpt5.2")
    assert not _is_websocket_model("azure/gpt-5-mini")
    assert not _is_websocket_model("openrouter/gpt-5-mini")
    assert not _is_websocket_model("gpt-4o")


def test_is_realtime_model_matches_only_direct_openai_models():
    assert _is_realtime_model("gpt-4o-realtime-preview")
    assert _is_realtime_model("openai/gpt-4o-realtime-preview")
    assert not _is_realtime_model("azure/gpt-4o-realtime-preview")
    assert not _is_realtime_model("openrouter/gpt-4o-realtime-preview")


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
