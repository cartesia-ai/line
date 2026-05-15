"""Tests for the WebSocket (Responses API) provider."""

import asyncio
from typing import List

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.provider import Message, StreamChunk, ToolCall, parse_model_id
from line.llm_agent.provider_utils import (
    _compute_divergence,
    _context_identity,
    _expand_messages,
    _extract_model_output_identities,
)
from line.llm_agent.schema_converter import build_openai_tool_defs
from line.llm_agent.websocket_provider import (
    _build_request,
    _plan_chat,
    _WebSocketProvider,
)

# ---------------------------------------------------------------------------
# _plan_chat
# ---------------------------------------------------------------------------


def test_plan_chat_update_preserves_text_then_tool_call_outputs():
    config = _normalize_config(LlmConfig())
    context_id = _context_identity(None, None, temperature=config.temperature, max_tokens=config.max_tokens)
    history = [(context_id, "warmup")]

    _, update = _plan_chat(
        history=history,
        model_id=parse_model_id("openai/gpt-5.2"),
        default_reasoning_effort="none",
        messages=[Message(role="user", content="Weather?")],
        tools=None,
        config=config,
    )

    response = {
        "id": "resp_1",
        "status": "completed",
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

    new_history = update(history, response)

    assert new_history[0] == (context_id, "warmup")
    assert new_history[1][0] == ("user", "Weather?", "", "")
    assert new_history[2] == (("assistant", "Let me check. ", "", ""), "resp_1")
    assert new_history[3] == (
        ("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)),
        "resp_1",
    )


def test_plan_chat_continuation_from_checkpoint():
    """Plan chains from the latest response_id checkpoint."""
    config = _normalize_config(LlmConfig())
    context_id = _context_identity(None, None, temperature=config.temperature, max_tokens=config.max_tokens)
    history = [
        (context_id, "warmup"),
        (("user", "hello", "", ""), None),
        (("assistant", "hi", "", ""), "resp_1"),
    ]

    request, _ = _plan_chat(
        history=history,
        model_id=parse_model_id("openai/gpt-5.2"),
        default_reasoning_effort="none",
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="what's up?"),
        ],
        tools=None,
        config=config,
    )

    assert request["previous_response_id"] == "resp_1"
    # Only the new message should be in the input
    input_items = request["input"]
    assert len(input_items) == 1
    assert input_items[0]["content"][0]["text"] == "what's up?"


def test_plan_chat_divergence_rolls_back_to_checkpoint():
    """When history diverges, rolls back to the latest response_id checkpoint."""
    config = _normalize_config(LlmConfig())
    context_id = _context_identity(None, None, temperature=config.temperature, max_tokens=config.max_tokens)
    history = [
        (context_id, "warmup"),
        (("user", "hello", "", ""), None),
        (("assistant", "original response", "", ""), "resp_1"),
    ]

    request, _ = _plan_chat(
        history=history,
        model_id=parse_model_id("openai/gpt-5.2"),
        default_reasoning_effort="none",
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="truncated"),  # diverges
            Message(role="user", content="continue"),
        ],
        tools=None,
        config=config,
    )

    # Diverges at assistant message, but warmup checkpoint exists
    assert request["previous_response_id"] == "warmup"
    # Resends from after the warmup checkpoint
    assert len(request["input"]) == 3


def test_plan_chat_update_builds_correct_history():
    """Update function builds history from a completed response."""
    config = _normalize_config(LlmConfig())

    _, update = _plan_chat(
        history=[],
        model_id=parse_model_id("openai/gpt-5.2"),
        default_reasoning_effort="none",
        messages=[Message(role="user", content="hi")],
        tools=None,
        config=config,
    )

    completed_response = {
        "status": "completed",
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
    }
    result = update([], completed_response)
    # Should contain context entry + user message entry + model output entry
    assert len(result) >= 2
    assert result[-1][1] == "resp_123"


# ---------------------------------------------------------------------------
# _build_request
# ---------------------------------------------------------------------------


def test_build_request_uses_bare_model_name():
    request = _build_request(
        model_id=parse_model_id("openai/gpt-5.2"),
        default_reasoning_effort="none",
        instructions=None,
        tool_defs=None,
        cfg=LlmConfig(),
    )
    assert request["model"] == "gpt-5.2"


# ---------------------------------------------------------------------------
# _extract_model_output_identities
# ---------------------------------------------------------------------------


def test_extract_model_output_identities_reads_output_text():
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


# ---------------------------------------------------------------------------
# chat retry
# ---------------------------------------------------------------------------


def test_chat_retries_previous_response_not_found(monkeypatch):
    provider = _WebSocketProvider(model_id=parse_model_id("openai/gpt-5.2"))
    attempts = 0

    class _FakeStream:
        def __init__(self, *, fail: bool):
            self._fail = fail
            self.done = False

        async def __aiter__(self):
            self.done = True
            if self._fail:
                provider._history = []
                raise RuntimeError(
                    "OpenAI API error (previous_response_not_found): previous response was not found"
                )
            yield StreamChunk(text="hello")
            yield StreamChunk(is_final=True)

    async def _fake_setup(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        await provider._get_lock().acquire()
        return _FakeStream(fail=attempts == 1)

    provider._setup_chat = lambda *a, **kw: _fake_setup()
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


# ---------------------------------------------------------------------------
# _expand_messages (responses API text type)
# ---------------------------------------------------------------------------


def test_expand_messages_preserves_assistant_text_before_tool_call():
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


# ---------------------------------------------------------------------------
# _context_identity
# ---------------------------------------------------------------------------


def test_context_identity_accepts_native_web_search_tool():
    tool_defs = build_openai_tool_defs(
        web_search_options={"search_context_size": "high"},
        responses_api=True,
    )

    assert _context_identity(None, tool_defs, temperature=None, max_tokens=None) == (
        "__context__",
        "",
        ('{"search_context_size": "high", "type": "web_search"}',),
        None,
        None,
    )


def test_context_identity_changes_when_web_search_options_change():
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


# ---------------------------------------------------------------------------
# _compute_divergence
# ---------------------------------------------------------------------------


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
