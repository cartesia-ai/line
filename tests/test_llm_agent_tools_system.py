"""
Tests for built-in tools.

uv run pytest tests/test_llm_agent_tools_system.py -v
"""

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
)
from line.llm_agent.tools.utils import ToolType

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
    assert ft.tool_type == ToolType.LOOPBACK
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
