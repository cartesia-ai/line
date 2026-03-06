"""Tests for the Realtime WebSocket provider."""

from line.llm_agent.config import LlmConfig
from line.llm_agent.provider import Message, ToolCall
from line.llm_agent.realtime_provider import RECONNECT_THRESHOLD, _plan_chat, _track_output_items
from line.llm_agent.stream import _context_identity, _expand_messages


def _ctx_id(instructions=None, tool_defs=None, config=None):
    """Helper to build a context identity matching what _plan_chat computes.

    Uses raw config values (including _UNSET sentinels) to match the identity
    that _plan_chat generates internally.
    """
    cfg = config or LlmConfig()
    return _context_identity(instructions, tool_defs, temperature=cfg.temperature, max_tokens=cfg.max_tokens)


# ---------------------------------------------------------------------------
# _plan_chat
# ---------------------------------------------------------------------------


def test_plan_chat_session_update_on_sampling_change():
    state = [(_context_identity("stay concise", None, temperature=0.2, max_tokens=100), None)]
    messages = [Message(role="system", content="stay concise"), Message(role="user", content="hi")]

    diff = _plan_chat(
        state,
        messages,
        tools=None,
        config=LlmConfig(temperature=0.9, max_tokens=200),
    )

    assert not diff.reconnect
    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    session = event["session"]
    assert session["instructions"] == "stay concise"
    assert session["tools"] is None
    assert session["temperature"] == 0.9
    assert session["max_response_output_tokens"] == 200


def test_plan_chat_clears_session_fields():
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

    diff = _plan_chat(
        state,
        messages,
        tools=None,
        config=LlmConfig(temperature=None, max_tokens=None),
    )

    assert not diff.reconnect
    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    session = event["session"]
    assert session["instructions"] is None
    assert session["tools"] is None
    assert session["temperature"] is None
    assert session["max_response_output_tokens"] is None


def test_plan_chat_uses_config_system_prompt():
    diff = _plan_chat(
        [],
        [Message(role="user", content="hi")],
        tools=None,
        config=LlmConfig(system_prompt="stay concise"),
    )

    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    assert event["session"]["instructions"] == "stay concise"


def test_plan_chat_includes_native_web_search_tool():
    diff = _plan_chat(
        [],
        [Message(role="user", content="hi")],
        tools=None,
        config=LlmConfig(),
        web_search_options={"search_context_size": "high"},
    )

    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    assert event["session"]["tools"] == [
        {"type": "web_search", "name": "web_search", "search_context_size": "high"}
    ]


def test_plan_chat_append_only():
    """Common case: history has context + user msg, new user msg appended."""
    ctx_id = _ctx_id("hi")
    history = [
        (ctx_id, None),
        (("user", "hello", "", ""), "item_abc"),
        (("assistant", "hi there", "", ""), "item_def"),
    ]

    diff = _plan_chat(
        history,
        [
            Message(role="system", content="hi"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
            Message(role="user", content="what's up?"),
        ],
        tools=None,
        config=LlmConfig(),
    )

    assert not diff.reconnect
    # No session update (context unchanged), just one create
    assert len(diff.steps) == 1
    event, _ = diff.steps[0]
    assert event["type"] == "conversation.item.create"
    assert event["item"]["content"][0]["text"] == "what's up?"


def test_plan_chat_deletes_divergent_suffix():
    """When history diverges, old items are deleted and new ones created."""
    ctx_id = _ctx_id("sys")
    history = [
        (ctx_id, None),
        (("user", "hello", "", ""), "item_1"),
        (("assistant", "old response", "", ""), "item_2"),
    ]

    diff = _plan_chat(
        history,
        [
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="new response"),
        ],
        tools=None,
        config=LlmConfig(),
    )

    assert not diff.reconnect
    # Delete item_2 (old response), create new response
    types = [event["type"] for event, _ in diff.steps]
    assert "conversation.item.delete" in types
    assert "conversation.item.create" in types


def test_plan_chat_reconnect_on_large_diff():
    """When too many items need deleting, reconnect is requested."""
    ctx_id = _ctx_id("sys")
    history = [(ctx_id, None)]
    for i in range(RECONNECT_THRESHOLD + 5):
        history.append((("user", f"msg_{i}", "", ""), f"item_{i}"))

    diff = _plan_chat(
        history,
        [Message(role="system", content="sys"), Message(role="user", content="fresh start")],
        tools=None,
        config=LlmConfig(),
    )

    assert diff.reconnect
    # Steps contain full rebuild (session update + all creates)
    assert len(diff.steps) > 0
    assert diff.steps[0][0]["type"] == "session.update"


def test_plan_chat_no_ops_when_unchanged():
    """No steps when desired state matches current history exactly."""
    ctx_id = _ctx_id("sys")
    history = [
        (ctx_id, None),
        (("user", "hello", "", ""), "item_1"),
    ]

    diff = _plan_chat(
        history,
        [Message(role="system", content="sys"), Message(role="user", content="hello")],
        tools=None,
        config=LlmConfig(),
    )

    assert not diff.reconnect
    assert len(diff.steps) == 0


def test_plan_chat_history_update_applies_server_id():
    """The create step's update function picks up the server-assigned item ID."""
    diff = _plan_chat(
        [],
        [Message(role="user", content="hi")],
        tools=None,
        config=LlmConfig(),
    )

    # Find the create step (after session update)
    create_steps = [(e, u) for e, u in diff.steps if e["type"] == "conversation.item.create"]
    assert len(create_steps) == 1

    _, update = create_steps[0]
    new_history = update([], {"item": {"id": "server_assigned_123"}})
    assert new_history[-1][1] == "server_assigned_123"


# ---------------------------------------------------------------------------
# _track_output_items
# ---------------------------------------------------------------------------


def test_track_output_items_preserves_text_then_tool_call():
    history = _track_output_items(
        [],
        [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Let me check. "}],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"city":"NYC"}',
                "call_id": "call_1",
            },
        ],
    )

    assert [identity for identity, _ in history] == [
        ("assistant", "Let me check. ", "", ""),
        ("assistant_tool_call", (("get_weather", '{"city":"NYC"}', "call_1"),)),
    ]


def test_track_output_items_ignores_unknown_types():
    history = _track_output_items([], [{"type": "unknown", "id": "x"}])
    assert history == []


# ---------------------------------------------------------------------------
# _expand_messages (realtime text type)
# ---------------------------------------------------------------------------


def test_expand_messages_preserves_assistant_text_before_tool_call():
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
