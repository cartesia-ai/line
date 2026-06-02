"""
Tests for built-in tools.

uv run pytest tests/test_llm_agent_tools_system.py -v
"""

import json
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from line.events import AgentEndCall, AgentSendDtmf, AgentSendText, AgentTransferCall
from line.knowledge_base import KnowledgeBaseError
from line.llm_agent.provider import parse_model_id
from line.llm_agent.tools.system import (
    EndCallTool,
    KnowledgeBaseTool,
    TransferCallTool,
    end_call,
    knowledge_base,
    send_dtmf,
    transfer_call,
    webhook_tool,
)
from line.llm_agent.tools.utils import FunctionTool, ToolType

# Use anyio for async test support with asyncio backend only
pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


async def collect_events(gen) -> List:
    """Helper to collect all events from an async generator."""
    events = []
    async for event in gen:
        events.append(event)
    return events


@pytest.fixture
def mock_ctx():
    """Create a mock ToolEnv context."""
    return MagicMock()


class FakeKnowledgeBase:
    def __init__(self, results=None, error: Optional[Exception] = None):
        self.results = results if results is not None else []
        self.error = error
        self.last_query = None
        self.last_filters = None
        self.last_top_k = None
        self.last_timeout_s = None

    async def query(self, query, filters=None, top_k=None, timeout_s=None):
        self.last_query = query
        self.last_filters = filters
        self.last_top_k = top_k
        self.last_timeout_s = timeout_s
        if self.error is not None:
            raise self.error
        return self.results


def tool_ctx_with_kb(kb: FakeKnowledgeBase):
    ctx = MagicMock()
    ctx.knowledge_base.return_value = kb
    return ctx


# =============================================================================
# Tests: knowledge_base
# =============================================================================


def test_knowledge_base_tool_default_metadata(anyio_backend):
    ft = knowledge_base.as_function_tool()
    assert ft.name == "knowledge_base"
    assert ft.tool_type == ToolType.GENERAL
    assert "knowledge base" in ft.description.lower()
    assert "query" in ft.parameters


def test_knowledge_base_tool_configured_filters(anyio_backend):
    configured = knowledge_base(filters={"k": "v"}, top_k=2, timeout_s=1.5)
    assert isinstance(configured, KnowledgeBaseTool)
    assert configured._filters == {"k": "v"}
    assert configured._top_k == 2
    assert configured._timeout_s == 1.5


def test_knowledge_base_tool_call_inherits_existing_config(anyio_backend):
    # Calling an already-configured tool with no overrides preserves the
    # original config rather than silently resetting it to defaults.
    configured = knowledge_base(filters={"k": "v"}, top_k=2, timeout_s=1.5)
    rechained = configured()
    assert rechained._filters == {"k": "v"}
    assert rechained._top_k == 2
    assert rechained._timeout_s == 1.5

    # Per-arg overrides win, the rest are inherited.
    overridden = configured(top_k=7)
    assert overridden._filters == {"k": "v"}
    assert overridden._top_k == 7
    assert overridden._timeout_s == 1.5


async def test_knowledge_base_tool_invokes_kb_with_configured_filters(anyio_backend):
    kb = FakeKnowledgeBase(results=[{"content": "doc"}])
    ctx = tool_ctx_with_kb(kb)

    tool = knowledge_base(filters={"category": "billing"}, top_k=2, timeout_s=1.5).as_function_tool()
    result = await tool.func(ctx, "what is X?")

    assert result == "doc"
    assert kb.last_query == "what is X?"
    assert kb.last_filters == {"category": "billing"}
    assert kb.last_top_k == 2
    assert kb.last_timeout_s == 1.5


async def test_knowledge_base_tool_joins_multiple_chunks(anyio_backend):
    kb = FakeKnowledgeBase(results=[{"content": "first"}, {"content": "second"}])
    ctx = tool_ctx_with_kb(kb)

    result = await knowledge_base.as_function_tool().func(ctx, "q")

    assert result == "first\n\n---\n\nsecond"


async def test_knowledge_base_tool_skips_blank_content(anyio_backend):
    # Blank/missing content is dropped at the tool layer (presentation concern),
    # not at the client layer (transport concern).
    kb = FakeKnowledgeBase(results=[{"content": ""}, {"content": "real"}, {"foo": "bar"}])
    ctx = tool_ctx_with_kb(kb)

    result = await knowledge_base.as_function_tool().func(ctx, "q")

    assert result == "real"


async def test_knowledge_base_tool_returns_friendly_no_results(anyio_backend):
    kb = FakeKnowledgeBase(results=[])
    ctx = tool_ctx_with_kb(kb)

    result = await knowledge_base.as_function_tool().func(ctx, "anything")

    assert "no relevant" in result.lower()


async def test_knowledge_base_tool_handles_kb_error_gracefully(anyio_backend):
    kb = FakeKnowledgeBase(error=KnowledgeBaseError("missing credentials"))
    ctx = tool_ctx_with_kb(kb)

    result = await knowledge_base.as_function_tool().func(ctx, "anything")

    assert "currently unavailable" in result.lower()


def test_knowledge_base_tool_default_is_not_background(anyio_backend):
    assert knowledge_base.as_function_tool().is_background is False


def test_knowledge_base_tool_is_background_propagates_to_function_tool(anyio_backend):
    configured = knowledge_base(is_background=True)
    assert configured._is_background is True
    assert configured.as_function_tool().is_background is True


def test_knowledge_base_tool_call_inherits_is_background(anyio_backend):
    # is_background must round-trip through __call__ chaining like the
    # other config fields, otherwise re-configuring (e.g. tweaking top_k)
    # would silently flip it back to the default.
    configured = knowledge_base(is_background=True)
    rechained = configured(top_k=7)
    assert rechained._is_background is True
    assert rechained.as_function_tool().is_background is True


def test_knowledge_base_tool_warns_on_long_timeout(anyio_backend):
    from loguru import logger as loguru_logger

    messages: List[str] = []
    handler_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        knowledge_base(timeout_s=30.0)
    finally:
        loguru_logger.remove(handler_id)
    assert any("timeout_s=30.0" in m for m in messages)


def test_knowledge_base_tool_does_not_warn_on_short_timeout(anyio_backend):
    from loguru import logger as loguru_logger

    messages: List[str] = []
    handler_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    try:
        knowledge_base(timeout_s=2.0)
    finally:
        loguru_logger.remove(handler_id)
    assert not any("timeout_s" in m for m in messages)


# =============================================================================
# Tests: transfer_call
# =============================================================================


async def test_transfer_call_valid_number(mock_ctx, anyio_backend):
    """Test that a valid E.164 phone number triggers transfer."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_valid_number_with_message(mock_ctx, anyio_backend):
    """Test that a tool configured with message= sends it before transfer."""
    tool = transfer_call(message="Transferring you now")
    events = await collect_events(tool.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 2
    assert isinstance(events[0], AgentSendText)
    assert events[0].text == "Transferring you now"
    assert events[0].interruptible is True
    assert isinstance(events[1], AgentTransferCall)
    assert events[1].target_phone_number == "+14155551234"
    assert events[1].interruptible is True


async def test_transfer_call_invalid_number(mock_ctx, anyio_backend):
    """Test that an invalid phone number returns error message."""
    # +1415555123 is too short to be valid
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+1415555123"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendText)
    assert "invalid" in events[0].text.lower()


async def test_transfer_call_unparseable_number(mock_ctx, anyio_backend):
    """Test that an unparseable phone number returns error message."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "not-a-phone-number"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendText)
    assert "couldn't understand" in events[0].text.lower()


async def test_transfer_call_invalid_number_no_transfer(mock_ctx, anyio_backend):
    """Test that invalid number does not yield AgentTransferCall."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "123"))

    # Should only have error message, no transfer
    for event in events:
        assert not isinstance(event, AgentTransferCall)


async def test_transfer_call_international_number(mock_ctx, anyio_backend):
    """Test that international numbers are validated correctly."""
    # Valid UK number
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+442071234567"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].target_phone_number == "+442071234567"


async def test_transfer_call_normalizes_spaces(mock_ctx, anyio_backend):
    """Test that phone numbers with spaces are normalized to E.164 format."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+1 415 555 1234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    # Should be normalized to E.164 without spaces
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_normalizes_dashes(mock_ctx, anyio_backend):
    """Test that phone numbers with dashes are normalized to E.164 format."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+1-415-555-1234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    # Should be normalized to E.164 without dashes
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_normalizes_mixed_formatting(mock_ctx, anyio_backend):
    """Test that phone numbers with mixed formatting are normalized to E.164."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+1 (415) 555-1234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    # Should be normalized to E.164 without any formatting
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_normalizes_international_with_spaces(mock_ctx, anyio_backend):
    """Test that international numbers with spaces are normalized."""
    # UK number with spaces
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+44 20 7123 4567"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].target_phone_number == "+442071234567"


# =============================================================================
# Tests: send_dtmf
# =============================================================================


async def test_send_dtmf_digit(mock_ctx, anyio_backend):
    """Test that digit buttons send DTMF."""
    events = await collect_events(send_dtmf.func(mock_ctx, "5"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendDtmf)
    assert events[0].button == "5"


async def test_send_dtmf_star(mock_ctx, anyio_backend):
    """Test that star button sends DTMF."""
    events = await collect_events(send_dtmf.func(mock_ctx, "*"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendDtmf)
    assert events[0].button == "*"


async def test_send_dtmf_hash(mock_ctx, anyio_backend):
    """Test that hash button sends DTMF."""
    events = await collect_events(send_dtmf.func(mock_ctx, "#"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendDtmf)
    assert events[0].button == "#"


# =============================================================================
# Tests: end_call
# =============================================================================


async def test_end_call_default_description(mock_ctx, anyio_backend):
    """Test that default end_call has a conservative default description."""
    assert end_call.description == EndCallTool.DEFAULT_DESCRIPTION
    assert "Use when:" in end_call.description
    assert "Don't use when:" in end_call.description


async def test_end_call_yields_agent_end_call(mock_ctx, anyio_backend):
    """Test that end_call yields AgentEndCall event."""
    func_tool = end_call.as_function_tool()
    # LLM must provide a reason when calling end_call
    events = await collect_events(func_tool.func(mock_ctx, reason="user said goodbye"))

    assert len(events) == 1
    assert isinstance(events[0], AgentEndCall)


async def test_end_call_requires_reason_parameter(mock_ctx, anyio_backend):
    """Test that the end_call tool schema requires a reason parameter."""
    func_tool = end_call.as_function_tool()

    # Check that 'reason' is in the parameters and is required
    assert "reason" in func_tool.parameters
    assert func_tool.parameters["reason"].required is True


async def test_end_call_custom_description(mock_ctx, anyio_backend):
    """Test that custom description replaces the default."""
    custom_desc = "Only end when user says 'terminate'"
    custom_end_call = end_call(description=custom_desc)

    assert custom_end_call.description == custom_desc


async def test_end_call_has_function_tool_attributes(mock_ctx, anyio_backend):
    """Test that EndCallTool.as_function_tool() returns a proper FunctionTool."""
    func_tool = end_call.as_function_tool()

    # Check it's a real FunctionTool with all required attributes
    assert hasattr(func_tool, "name")
    assert hasattr(func_tool, "description")
    assert hasattr(func_tool, "parameters")
    assert hasattr(func_tool, "tool_type")
    assert hasattr(func_tool, "is_background")
    assert hasattr(func_tool, "func")

    assert func_tool.name == "end_call"
    assert func_tool.tool_type == ToolType.GENERAL
    assert func_tool.is_background is False

    # Verify it's actually a FunctionTool instance (not duck-typed)
    from line.llm_agent.tools.utils import FunctionTool

    assert isinstance(func_tool, FunctionTool)


async def test_end_call_callable_returns_new_instance(mock_ctx, anyio_backend):
    """Test that calling end_call() returns a new configured instance."""
    custom_desc = "Custom description for test"
    configured = end_call(description=custom_desc)

    # Should be a new instance
    assert configured is not end_call
    assert isinstance(configured, EndCallTool)

    # Original should be unchanged
    assert end_call.description == EndCallTool.DEFAULT_DESCRIPTION
    # Configured should have custom description
    assert configured.description == custom_desc


# =============================================================================
# Tests: EndCallTool interruptible
# =============================================================================


async def test_end_call_default_interruptible(mock_ctx, anyio_backend):
    """Test that default end_call has interruptible=True."""
    func_tool = end_call.as_function_tool()
    events = await collect_events(func_tool.func(mock_ctx, reason="goodbye"))

    assert len(events) == 1
    assert isinstance(events[0], AgentEndCall)
    assert events[0].interruptible is True


async def test_end_call_interruptible_false(mock_ctx, anyio_backend):
    """Test that EndCallTool(interruptible=False) propagates to AgentEndCall."""
    tool = EndCallTool(interruptible=False)
    events = await collect_events(tool.as_function_tool().func(mock_ctx, reason="done"))

    assert len(events) == 1
    assert isinstance(events[0], AgentEndCall)
    assert events[0].interruptible is False


async def test_end_call_callable_interruptible_false(mock_ctx, anyio_backend):
    """Test that end_call(interruptible=False) propagates to AgentEndCall."""
    tool = end_call(interruptible=False)
    events = await collect_events(tool.as_function_tool().func(mock_ctx, reason="done"))

    assert len(events) == 1
    assert isinstance(events[0], AgentEndCall)
    assert events[0].interruptible is False


async def test_end_call_custom_description_and_interruptible(mock_ctx, anyio_backend):
    """Test EndCallTool with both custom description and interruptible=False."""
    tool = EndCallTool(description="Custom end", interruptible=False)
    assert tool.description == "Custom end"
    assert tool.interruptible is False

    events = await collect_events(tool.as_function_tool().func(mock_ctx, reason="done"))
    assert events[0].interruptible is False


# =============================================================================
# Tests: TransferCallTool instantiation and interruptible
# =============================================================================


async def test_transfer_call_default_interruptible(mock_ctx, anyio_backend):
    """Test that default transfer_call has interruptible=True."""
    events = await collect_events(transfer_call.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].interruptible is True


async def test_transfer_call_interruptible_false(mock_ctx, anyio_backend):
    """Configured message + interruptible=False propagate to AgentSendText and AgentTransferCall."""
    tool = TransferCallTool(message="Hold on", interruptible=False)
    events = await collect_events(tool.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 2
    assert isinstance(events[0], AgentSendText)
    assert events[0].interruptible is False
    assert isinstance(events[1], AgentTransferCall)
    assert events[1].interruptible is False


async def test_transfer_call_callable_interruptible_false(mock_ctx, anyio_backend):
    """Test that transfer_call(interruptible=False) propagates to events."""
    tool = transfer_call(interruptible=False)
    events = await collect_events(tool.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].interruptible is False


async def test_transfer_call_with_default_message(mock_ctx, anyio_backend):
    """Test that TransferCallTool(message=...) speaks that message before transfer."""
    tool = TransferCallTool(message="Please hold")
    events = await collect_events(tool.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 2
    assert isinstance(events[0], AgentSendText)
    assert events[0].text == "Please hold"
    assert events[0].interruptible is True
    assert isinstance(events[1], AgentTransferCall)
    assert events[1].interruptible is True


async def test_transfer_call_callable_with_message_and_interruptible(mock_ctx, anyio_backend):
    """Test that transfer_call(message=..., interruptible=False) works."""
    tool = transfer_call(message="Transferring", interruptible=False)
    assert tool.message == "Transferring"
    assert tool.interruptible is False

    events = await collect_events(tool.as_function_tool().func(mock_ctx, "+14155551234"))

    assert len(events) == 2
    assert events[0].text == "Transferring"
    assert events[0].interruptible is False
    assert events[1].interruptible is False


async def test_transfer_call_has_function_tool_attributes(mock_ctx, anyio_backend):
    """Test that TransferCallTool.as_function_tool() returns a proper FunctionTool."""
    from line.llm_agent.tools.utils import FunctionTool

    func_tool = transfer_call.as_function_tool()
    assert isinstance(func_tool, FunctionTool)
    assert func_tool.name == "transfer_call"
    assert func_tool.tool_type == ToolType.GENERAL
    assert set(func_tool.parameters.keys()) == {"target_phone_number"}
    assert "message" not in func_tool.parameters
    assert func_tool.parameters["target_phone_number"].required is True


# =============================================================================
# Tests: _normalize_tools with TransferCallTool
# =============================================================================


async def test_normalize_tools_handles_transfer_call_tool(anyio_backend):
    """Test that _normalize_tools correctly handles TransferCallTool instances."""
    from line.llm_agent.tools.utils import FunctionTool, _normalize_tools

    tools, _ = _normalize_tools([TransferCallTool()], model_id=parse_model_id("gpt-4o"))
    assert len(tools) == 1
    assert isinstance(tools[0], FunctionTool)
    assert tools[0].name == "transfer_call"


async def test_normalize_tools_handles_end_call_tool(anyio_backend):
    """Test that _normalize_tools correctly handles EndCallTool instances."""
    from line.llm_agent.tools.utils import FunctionTool, _normalize_tools

    tools, _ = _normalize_tools([EndCallTool()], model_id=parse_model_id("gpt-4o"))
    assert len(tools) == 1
    assert isinstance(tools[0], FunctionTool)
    assert tools[0].name == "end_call"


async def test_normalize_tools_handles_knowledge_base_tool(anyio_backend):
    """Test that _normalize_tools correctly handles a single KnowledgeBaseTool."""
    from line.llm_agent.tools.utils import _normalize_tools

    function_tools, web_search_options = _normalize_tools(
        [knowledge_base(filters={"x": "y"})],
        parse_model_id("openai/gpt-4o"),
    )

    assert web_search_options is None
    assert [t.name for t in function_tools] == ["knowledge_base"]


async def test_normalize_tools_rejects_duplicate_names(anyio_backend):
    from line.llm_agent.tools.utils import _normalize_tools

    with pytest.raises(ValueError, match="Duplicate tool name"):
        _normalize_tools(
            [knowledge_base, knowledge_base(filters={"x": "y"})],
            parse_model_id("openai/gpt-4o"),
        )


# =============================================================================
# Tests: webhook_tool — build-time validation
# =============================================================================


def test_webhook_tool_empty_name_raises(anyio_backend):
    with pytest.raises(ValueError, match="name must be a non-empty"):
        webhook_tool(name="", description="d", url="https://example.com")


def test_webhook_tool_whitespace_name_raises(anyio_backend):
    with pytest.raises(ValueError, match="name must be a non-empty"):
        webhook_tool(name="   ", description="d", url="https://example.com")


def test_webhook_tool_empty_description_raises(anyio_backend):
    with pytest.raises(ValueError, match="description must be a non-empty"):
        webhook_tool(name="t", description="", url="https://example.com")


def test_webhook_tool_empty_url_raises(anyio_backend):
    with pytest.raises(ValueError, match="url must be a non-empty"):
        webhook_tool(name="t", description="d", url="")


def test_webhook_tool_invalid_method_raises(anyio_backend):
    with pytest.raises(ValueError, match="not a valid HTTP method"):
        webhook_tool(name="t", description="d", url="https://example.com", method="BANANA")


def test_webhook_tool_unmatched_opening_brace_raises(anyio_backend):
    with pytest.raises(ValueError, match="unmatched opening brace"):
        webhook_tool(name="t", description="d", url="https://example.com/{id")


def test_webhook_tool_unmatched_closing_brace_raises(anyio_backend):
    with pytest.raises(ValueError, match="unmatched closing brace"):
        webhook_tool(name="t", description="d", url="https://example.com/id}")


def test_webhook_tool_nested_braces_raises(anyio_backend):
    with pytest.raises(ValueError, match="nested braces"):
        webhook_tool(name="t", description="d", url="https://example.com/{{id}}")


def test_webhook_tool_empty_url_placeholder_raises(anyio_backend):
    with pytest.raises(ValueError, match="url template variable"):
        webhook_tool(name="t", description="d", url="https://example.com/{}/tickets")


def test_webhook_tool_duplicate_url_placeholder_raises(anyio_backend):
    with pytest.raises(ValueError, match="appears more than once"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com/{item_id}/related/{item_id}",
        )


@pytest.mark.parametrize("placeholder", ["tenant id", " tenant", "tenant:id", "x" * 65])
def test_webhook_tool_invalid_url_placeholder_name_raises(anyio_backend, placeholder):
    with pytest.raises(ValueError, match="url template variable"):
        webhook_tool(
            name="t",
            description="d",
            url=f"https://example.com/{{{placeholder}}}/tickets",
        )


def test_webhook_tool_invalid_url_placeholder_raises_before_auth_env(
    anyio_backend, monkeypatch
):
    monkeypatch.delenv("MISSING_AUTH_ENV", raising=False)

    with pytest.raises(ValueError, match="url template variable"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com/{tenant id}/tickets",
            auth={"Authorization": "Bearer ${MISSING_AUTH_ENV}"},
        )


def test_webhook_tool_body_schema_not_object_type_raises(anyio_backend):
    with pytest.raises(ValueError, match='"type": "object"'):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={"type": "array", "items": {"type": "string"}},
        )


def test_webhook_tool_body_schema_missing_properties_raises(anyio_backend):
    with pytest.raises(ValueError, match='"properties"'):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={"type": "object"},
        )


def test_webhook_tool_body_schema_required_not_list_raises(anyio_backend):
    with pytest.raises(ValueError, match="must be a list"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "required": "name",
                "properties": {"name": {"type": "string"}},
            },
        )


def test_webhook_tool_body_schema_required_unknown_field_raises(anyio_backend):
    with pytest.raises(ValueError, match="not in properties"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "required": ["name", "ghost"],
                "properties": {"name": {"type": "string"}},
            },
        )


def test_webhook_tool_property_unknown_type_raises(anyio_backend):
    with pytest.raises(ValueError, match="unknown type"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {"x": {"type": "timestamp"}},
            },
        )


def test_webhook_tool_constant_value_type_mismatch_raises(anyio_backend):
    with pytest.raises(ValueError, match="constant_value.*is str"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "constant_value": "not_a_number"},
                },
            },
        )


def test_webhook_tool_integer_constant_value_rejects_bool(anyio_backend):
    with pytest.raises(ValueError, match="type='integer'.*constant_value=True is bool"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "constant_value": True},
                },
            },
        )


def test_webhook_tool_number_constant_value_rejects_bool(anyio_backend):
    with pytest.raises(ValueError, match="type='number'.*constant_value=False is bool"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "constant_value": False},
                },
            },
        )


def test_webhook_tool_query_integer_constant_value_rejects_bool(anyio_backend):
    with pytest.raises(ValueError, match="type='integer'.*constant_value=True is bool"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            query_params_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "constant_value": True},
                },
            },
        )


@pytest.mark.parametrize("constant_value", [{"x": 1}, ["x"], None])
def test_webhook_tool_query_constant_value_rejects_non_scalar(
    constant_value, anyio_backend
):
    with pytest.raises(
        ValueError,
        match="query_params_schema.*constant_value must be a scalar",
    ):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            query_params_schema={
                "type": "object",
                "properties": {
                    "fixed": {"constant_value": constant_value},
                },
            },
        )


def test_webhook_tool_constant_value_bool_mismatch_raises(anyio_backend):
    with pytest.raises(ValueError, match="constant_value.*is int"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean", "constant_value": 1},
                },
            },
        )


def test_webhook_tool_nested_schema_validation(anyio_backend):
    """Validation recurses into nested object schemas."""
    with pytest.raises(ValueError, match="nested_prop.*unknown type"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {
                    "outer": {
                        "type": "object",
                        "properties": {
                            "nested_prop": {"type": "invalid_type"},
                        },
                    },
                },
            },
        )


def test_webhook_tool_query_params_schema_validation(anyio_backend):
    """query_params_schema gets the same validation as body_schema."""
    with pytest.raises(ValueError, match="query_params_schema.*\"type\": \"object\""):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            query_params_schema={"type": "string"},
        )


def test_webhook_tool_negative_timeout_raises(anyio_backend):
    with pytest.raises(ValueError, match="timeout must be positive"):
        webhook_tool(name="t", description="d", url="https://example.com", timeout=-1.0)



def test_webhook_tool_property_not_dict_raises(anyio_backend):
    with pytest.raises(ValueError, match="must be a dict"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            body_schema={
                "type": "object",
                "properties": {"x": "string"},
            },
        )


# =============================================================================
# Tests: webhook_tool — behavior
# =============================================================================

def _fake_aiohttp(monkeypatch, *, status=200, body="ok", capture=None):
    """Patch aiohttp.ClientSession to return a canned response.

    If *capture* is a dict, request kwargs are stored into it.
    """

    class _Resp:
        def __init__(self):
            self.status = status

        async def text(self):
            return body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class _Sess:
        def request(self, **kwargs):
            if capture is not None:
                capture.update(kwargs)
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    monkeypatch.setattr("aiohttp.ClientSession", lambda: _Sess())


_TICKET_BODY_SCHEMA = {
    "type": "object",
    "required": ["subject"],
    "properties": {
        "subject": {"type": "string", "description": "Short summary of the issue."},
        "source": {"type": "string", "constant_value": "voice_agent"},
    },
}


def test_webhook_tool_returns_function_tool(anyio_backend, monkeypatch):
    monkeypatch.setenv("SUPPORT_API_KEY", "test-key")
    tool = webhook_tool(
        name="create_ticket",
        description="Creates a ticket.",
        url="https://example.com/api/tickets",
        body_schema=_TICKET_BODY_SCHEMA,
        auth={"Authorization": "Bearer ${SUPPORT_API_KEY}"},
    )
    assert isinstance(tool, FunctionTool)
    assert tool.name == "create_ticket"
    assert tool.description == "Creates a ticket."
    assert tool.tool_type == ToolType.GENERAL
    # constant_value properties are hidden
    assert "subject" in tool.parameters
    assert "source" not in tool.parameters


def test_webhook_tool_parameter_types(anyio_backend):
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["s", "i", "n", "b", "a"],
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
            },
        },
    )
    assert tool.parameters["s"].type_annotation is str
    assert tool.parameters["i"].type_annotation is int
    assert tool.parameters["n"].type_annotation is float
    assert tool.parameters["b"].type_annotation is bool
    assert tool.parameters["a"].type_annotation is list


def test_webhook_tool_required_and_optional(anyio_backend):
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
        },
    )
    assert tool.parameters["a"].required is True
    assert tool.parameters["b"].required is False


def test_webhook_tool_is_background_default(anyio_backend):
    tool = webhook_tool(name="t", description="d", url="https://example.com")
    assert tool.is_background is True


def test_webhook_tool_is_background_configurable(anyio_backend):
    tool = webhook_tool(name="t", description="d", url="https://example.com", is_background=False)
    assert tool.is_background is False


def test_webhook_tool_auth_missing_env_raises(anyio_backend, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(ValueError, match="MISSING_VAR"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            auth={"Authorization": "Bearer ${MISSING_VAR}"},
        )


def test_webhook_tool_url_template_params(anyio_backend):
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com/orders/{order_id}/items/{item_id}",
    )
    assert "order_id" in tool.parameters
    assert "item_id" in tool.parameters
    assert tool.parameters["order_id"].required is True
    assert tool.parameters["item_id"].required is True
    assert tool.parameters["order_id"].type_annotation is str


def test_webhook_tool_query_params_schema(anyio_backend):
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com/search",
        method="GET",
        query_params_schema={
            "type": "object",
            "required": ["q"],
            "properties": {
                "q": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
        },
    )
    assert "q" in tool.parameters
    assert "limit" in tool.parameters
    assert tool.parameters["q"].required is True
    assert tool.parameters["limit"].required is False
    assert tool.parameters["q"].description == "Search query"


def test_webhook_tool_query_schema_does_not_leak_constant_value(anyio_backend):
    from line.llm_agent.schema_converter import function_tool_to_litellm

    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com/search",
        method="GET",
        query_params_schema={
            "type": "object",
            "required": ["source", "mode"],
            "properties": {
                "source": {"type": "string", "constant_value": "voice_agent"},
                "mode": {
                    "anyOf": [
                        {"type": "string", "constant_value": "standard"},
                        {"type": "integer"},
                    ],
                },
            },
        },
    )

    schema = function_tool_to_litellm(tool, strict=False)
    properties = schema["function"]["parameters"]["properties"]
    mode_schema = properties["mode"]
    required = schema["function"]["parameters"]["required"]
    assert "source" not in properties
    assert required == ["mode"]
    assert mode_schema == {"anyOf": [{"type": "string"}, {"type": "integer"}]}


def test_webhook_tool_combined_params(anyio_backend):
    """URL template + body + query params all appear in parameters."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com/{tenant_id}/tickets",
        body_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string"},
                "source": {"type": "string", "constant_value": "voice"},
            },
        },
        query_params_schema={
            "type": "object",
            "required": ["priority"],
            "properties": {
                "priority": {"type": "string"},
            },
        },
    )
    assert set(tool.parameters.keys()) == {"tenant_id", "subject", "priority"}


def test_webhook_tool_normalize_tools(anyio_backend, monkeypatch):
    """webhook_tool is already a FunctionTool and passes through _normalize_tools."""
    from line.llm_agent.tools.utils import _normalize_tools

    monkeypatch.setenv("K", "v")
    tool = webhook_tool(
        name="wh",
        description="d",
        url="https://example.com",
        auth={"X": "${K}"},
    )
    function_tools, _ = _normalize_tools([tool], model_id=parse_model_id("gpt-4o"))
    assert len(function_tools) == 1
    assert function_tools[0].name == "wh"


async def test_webhook_tool_http_call(mock_ctx, anyio_backend, monkeypatch):
    """Test that the impl function makes the correct aiohttp request."""
    monkeypatch.setenv("API_KEY", "sk-test")

    tool = webhook_tool(
        name="create_ticket",
        description="Creates a ticket.",
        url="https://example.com/api/{tenant}/tickets",
        method="POST",
        body_schema=_TICKET_BODY_SCHEMA,
        query_params_schema={
            "type": "object",
            "required": ["notify"],
            "properties": {"notify": {"type": "boolean"}},
        },
        auth={"Authorization": "Bearer ${API_KEY}"},
        headers={"X-Custom": "value"},
        timeout=5.0,
    )

    captured = {}
    _fake_aiohttp(monkeypatch, status=201, body='{"id": "t-1"}', capture=captured)

    result = await tool.func(
        mock_ctx,
        tenant="acme",
        subject="Help me",
        notify=True,
    )

    assert captured["method"] == "POST"
    assert captured["url"] == "https://example.com/api/acme/tickets"
    assert captured["json"] == {"source": "voice_agent", "subject": "Help me"}
    assert captured["params"] == {"notify": "true"}
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["X-Custom"] == "value"
    assert captured["timeout"].total == 5.0
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["status"] == 201
    assert '"id": "t-1"' in parsed["body"]


async def test_webhook_tool_http_error_handling(mock_ctx, anyio_backend, monkeypatch):
    """HTTP errors return a structured error string instead of raising."""
    import aiohttp

    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
    )

    class _ErrorSession:
        def request(self, **kwargs):
            raise aiohttp.ClientConnectionError("Connection refused")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    monkeypatch.setattr("aiohttp.ClientSession", lambda: _ErrorSession())

    result = await tool.func(mock_ctx)
    parsed = json.loads(result)
    assert parsed["ok"] is False
    assert parsed["status"] is None
    assert "ClientConnectionError" in parsed["error"]


async def test_webhook_tool_timeout_without_configured_value(
    mock_ctx, anyio_backend, monkeypatch
):
    """Timeout errors omit the duration when no timeout was configured."""
    import asyncio

    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
    )

    class _TimeoutSession:
        def request(self, **kwargs):
            raise asyncio.TimeoutError

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    monkeypatch.setattr("aiohttp.ClientSession", lambda: _TimeoutSession())

    result = await tool.func(mock_ctx)

    parsed = json.loads(result)
    assert parsed["ok"] is False
    assert parsed["status"] is None
    assert parsed["error"] == "Request timed out."


async def test_webhook_tool_response_truncation(mock_ctx, anyio_backend, monkeypatch):
    """Large response bodies are truncated."""
    tool = webhook_tool(name="t", description="d", url="https://example.com")
    _fake_aiohttp(monkeypatch, body="x" * 10_000)

    result = await tool.func(mock_ctx)
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["body"].endswith("... (truncated)")
    assert len(parsed["body"]) < 5000


async def test_webhook_tool_unknown_kwargs_ignored(mock_ctx, anyio_backend, monkeypatch):
    """Kwargs not in any param set are not sent in the request body."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
        },
    )
    captured = {}
    _fake_aiohttp(monkeypatch, capture=captured)

    await tool.func(mock_ctx, a="hello", hallucinated_param="bad")
    assert "hallucinated_param" not in captured.get("json", {})


def test_webhook_tool_param_name_collision_raises(anyio_backend):
    """Duplicate parameter names across URL/body/query raise ValueError."""
    with pytest.raises(ValueError, match="order_id"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com/{order_id}",
            body_schema={
                "type": "object",
                "required": ["order_id"],
                "properties": {"order_id": {"type": "string"}},
            },
        )


async def test_webhook_tool_get_request_no_body(mock_ctx, anyio_backend, monkeypatch):
    """GET requests with only query params should not send a json body."""
    tool = webhook_tool(
        name="lookup",
        description="Look up order.",
        url="https://example.com/orders",
        method="GET",
        query_params_schema={
            "type": "object",
            "required": ["order_id"],
            "properties": {"order_id": {"type": "string"}},
        },
    )
    captured = {}
    _fake_aiohttp(monkeypatch, body='{"status": "shipped"}', capture=captured)

    result = await tool.func(mock_ctx, order_id="ORD-123")

    assert captured["method"] == "GET"
    assert captured["params"] == {"order_id": "ORD-123"}
    assert "json" not in captured
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["status"] == 200
    assert "shipped" in parsed["body"]


async def test_webhook_tool_url_params_are_encoded(mock_ctx, anyio_backend, monkeypatch):
    """URL template params are percent-encoded to prevent broken URLs."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com/items/{item_id}",
    )
    captured = {}
    _fake_aiohttp(monkeypatch, capture=captured)

    await tool.func(mock_ctx, item_id="has spaces/and slashes")
    assert captured["url"] == "https://example.com/items/has%20spaces%2Fand%20slashes"


async def test_webhook_tool_url_params_allow_hyphens(mock_ctx, anyio_backend, monkeypatch):
    """URL template params can contain non-word characters like hyphens."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com/{tenant-id}/tickets",
    )
    captured = {}
    _fake_aiohttp(monkeypatch, capture=captured)

    assert "tenant-id" in tool.parameters

    await tool.func(mock_ctx, **{"tenant-id": "acme corp"})
    assert captured["url"] == "https://example.com/acme%20corp/tickets"


def test_webhook_tool_enum_passthrough(anyio_backend):
    """Enum constraints from JSON schema are preserved in ParameterInfo."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["priority"],
            "properties": {
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Ticket priority.",
                },
            },
        },
    )
    assert tool.parameters["priority"].enum == ["low", "medium", "high"]


def test_webhook_tool_nested_constant_value(anyio_backend):
    """constant_value inside nested objects is hidden from parameters."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["ticket"],
            "properties": {
                "ticket": {
                    "type": "object",
                    "required": ["subject", "channel"],
                    "properties": {
                        "subject": {"type": "string", "description": "Summary."},
                        "channel": {"type": "string", "constant_value": "voice"},
                    },
                },
            },
        },
    )
    # ticket is still visible (has non-constant children)
    assert "ticket" in tool.parameters
    assert tool.parameters["ticket"].required is True
    assert tool.parameters["ticket"].type_annotation is dict
    assert tool.parameters["ticket"].json_schema["type"] == "object"
    assert set(tool.parameters["ticket"].json_schema["properties"]) == {"subject"}
    assert tool.parameters["ticket"].json_schema["required"] == ["subject"]


async def test_webhook_tool_nested_constant_injected(mock_ctx, anyio_backend, monkeypatch):
    """Nested constant_value fields are deep-merged into the request body."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["ticket"],
            "properties": {
                "ticket": {
                    "type": "object",
                    "required": ["subject", "channel"],
                    "properties": {
                        "subject": {"type": "string"},
                        "channel": {"type": "string", "constant_value": "voice"},
                    },
                },
                "source": {"type": "string", "constant_value": "agent"},
            },
        },
    )
    captured = {}
    _fake_aiohttp(monkeypatch, capture=captured)

    await tool.func(mock_ctx, ticket={"subject": "Help"})

    body = captured["json"]
    assert body["source"] == "agent"
    assert body["ticket"]["channel"] == "voice"
    assert body["ticket"]["subject"] == "Help"


def test_webhook_tool_all_constant_nested_object_hidden(anyio_backend):
    """A nested object with only constant_value children is fully hidden."""
    tool = webhook_tool(
        name="t",
        description="d",
        url="https://example.com",
        body_schema={
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "constant_value": "voice"},
                        "version": {"type": "string", "constant_value": "1"},
                    },
                },
            },
        },
    )
    assert "name" in tool.parameters
    assert "metadata" not in tool.parameters


def test_webhook_tool_auth_headers_collision_raises(anyio_backend, monkeypatch):
    """Overlapping keys in auth and headers raise ValueError."""
    monkeypatch.setenv("KEY", "from-env")
    with pytest.raises(ValueError, match="X-Header"):
        webhook_tool(
            name="t",
            description="d",
            url="https://example.com",
            auth={"X-Header": "${KEY}"},
            headers={"X-Header": "static-override"},
        )
