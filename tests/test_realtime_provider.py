"""Tests for the Realtime WebSocket provider."""

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.provider import Message, ToolCall
from line.llm_agent.realtime_provider import RECONNECT_THRESHOLD, _plan_chat, _track_output_items
from line.llm_agent.provider_utils import _context_identity, _expand_messages


def _cfg(**overrides):
    """Return a normalized LlmConfig with optional overrides."""
    return _normalize_config(LlmConfig(**overrides))


def _ctx_id(instructions=None, tool_defs=None, config=None):
    """Helper to build a context identity matching what _plan_chat computes."""
    cfg = config or _cfg()
    return _context_identity(instructions, tool_defs, temperature=cfg.temperature, max_tokens=cfg.max_tokens)


# ---------------------------------------------------------------------------
# _plan_chat
# ---------------------------------------------------------------------------


def test_plan_chat_session_update_on_sampling_change():
    state = [(_context_identity("stay concise", None, temperature=0.2, max_tokens=100), None)]
    messages = [Message(role="system", content="stay concise"), Message(role="user", content="hi")]

    diff = _plan_chat(
        history=state,
        messages=messages,
        tools=None,
        config=_cfg(temperature=0.9, max_tokens=200),
    )

    assert not diff.reconnect
    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    session = event["session"]
    assert session["instructions"] == "stay concise"
    assert session["tools"] == []
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
        history=state,
        messages=messages,
        tools=None,
        config=_cfg(temperature=None, max_tokens=None),
    )

    assert not diff.reconnect
    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    session = event["session"]
    assert session["instructions"] == ""
    assert session["tools"] == []
    assert "temperature" not in session
    assert "max_response_output_tokens" not in session


def test_plan_chat_uses_config_system_prompt():
    diff = _plan_chat(
        history=[],
        messages=[Message(role="user", content="hi")],
        tools=None,
        config=_cfg(system_prompt="stay concise"),
    )

    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    assert event["session"]["instructions"] == "stay concise"


def test_plan_chat_includes_native_web_search_tool():
    diff = _plan_chat(
        history=[],
        messages=[Message(role="user", content="hi")],
        tools=None,
        config=_cfg(),
        web_search_options={"search_context_size": "high"},
    )

    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    assert event["session"]["tools"] == [{"type": "web_search", "search_context_size": "high"}]


def test_plan_chat_strips_strict_flag_from_function_tools():
    """Realtime API rejects ``session.tools[*].strict`` — the plan must omit
    the flag even though the schema is built in strict mode."""
    from typing import Annotated

    from line.llm_agent.tools.decorators import loopback_tool
    from line.llm_agent.tools.utils import ToolEnv

    @loopback_tool
    async def echo(ctx: ToolEnv, text: Annotated[str, "text to echo"]):
        """Echo the text."""
        return text

    diff = _plan_chat(
        history=[],
        messages=[Message(role="user", content="hi")],
        tools=[echo],
        config=_cfg(),
    )

    event, _ = diff.steps[0]
    assert event["type"] == "session.update"
    tool_defs = event["session"]["tools"]
    assert len(tool_defs) == 1
    assert tool_defs[0]["type"] == "function"
    assert tool_defs[0]["name"] == "echo"
    assert "strict" not in tool_defs[0]
    # Strict schema construction is preserved.
    assert tool_defs[0]["parameters"]["additionalProperties"] is False


def test_plan_chat_append_only():
    """Common case: history has context + user msg, new user msg appended."""
    ctx_id = _ctx_id("hi")
    history = [
        (ctx_id, None),
        (("user", "hello", "", ""), "item_abc"),
        (("assistant", "hi there", "", ""), "item_def"),
    ]

    diff = _plan_chat(
        history=history,
        messages=[
            Message(role="system", content="hi"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
            Message(role="user", content="what's up?"),
        ],
        tools=None,
        config=_cfg(),
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
        history=history,
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="new response"),
        ],
        tools=None,
        config=_cfg(),
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
        history=history,
        messages=[Message(role="system", content="sys"), Message(role="user", content="fresh start")],
        tools=None,
        config=_cfg(),
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
        history=history,
        messages=[Message(role="system", content="sys"), Message(role="user", content="hello")],
        tools=None,
        config=_cfg(),
    )

    assert not diff.reconnect
    assert len(diff.steps) == 0


def test_plan_chat_history_update_applies_server_id():
    """The create step's update function picks up the server-assigned item ID."""
    diff = _plan_chat(
        history=[],
        messages=[Message(role="user", content="hi")],
        tools=None,
        config=_cfg(),
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
