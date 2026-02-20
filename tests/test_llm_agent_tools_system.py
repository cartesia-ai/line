"""
Tests for built-in tools.

uv run pytest tests/test_tools.py -v
"""

from typing import List
from unittest.mock import MagicMock

import pytest

from line.events import AgentEndCall, AgentSendDtmf, AgentSendText, AgentTransferCall
from line.llm_agent.tools.system import EndCallTool, end_call, send_dtmf, transfer_call
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


# =============================================================================
# Tests: transfer_call
# =============================================================================


async def test_transfer_call_valid_number(mock_ctx, anyio_backend):
    """Test that a valid E.164 phone number triggers transfer."""
    events = await collect_events(transfer_call.func(mock_ctx, "+14155551234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_valid_number_with_message(mock_ctx, anyio_backend):
    """Test that a valid number with message sends message then transfers."""
    events = await collect_events(
        transfer_call.func(mock_ctx, "+14155551234", message="Transferring you now")
    )

    assert len(events) == 2
    assert isinstance(events[0], AgentSendText)
    assert events[0].text == "Transferring you now"
    assert isinstance(events[1], AgentTransferCall)
    assert events[1].target_phone_number == "+14155551234"


async def test_transfer_call_invalid_number(mock_ctx, anyio_backend):
    """Test that an invalid phone number returns error message."""
    # +1415555123 is too short to be valid
    events = await collect_events(transfer_call.func(mock_ctx, "+1415555123"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendText)
    assert "invalid" in events[0].text.lower()


async def test_transfer_call_unparseable_number(mock_ctx, anyio_backend):
    """Test that an unparseable phone number returns error message."""
    events = await collect_events(transfer_call.func(mock_ctx, "not-a-phone-number"))

    assert len(events) == 1
    assert isinstance(events[0], AgentSendText)
    assert "couldn't understand" in events[0].text.lower()


async def test_transfer_call_invalid_number_no_transfer(mock_ctx, anyio_backend):
    """Test that invalid number does not yield AgentTransferCall."""
    events = await collect_events(transfer_call.func(mock_ctx, "123"))

    # Should only have error message, no transfer
    for event in events:
        assert not isinstance(event, AgentTransferCall)


async def test_transfer_call_international_number(mock_ctx, anyio_backend):
    """Test that international numbers are validated correctly."""
    # Valid UK number
    events = await collect_events(transfer_call.func(mock_ctx, "+442071234567"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    assert events[0].target_phone_number == "+442071234567"


async def test_transfer_call_normalizes_spaces(mock_ctx, anyio_backend):
    """Test that phone numbers with spaces are normalized to E.164 format."""
    events = await collect_events(transfer_call.func(mock_ctx, "+1 415 555 1234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    # Should be normalized to E.164 without spaces
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_normalizes_dashes(mock_ctx, anyio_backend):
    """Test that phone numbers with dashes are normalized to E.164 format."""
    events = await collect_events(transfer_call.func(mock_ctx, "+1-415-555-1234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    # Should be normalized to E.164 without dashes
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_normalizes_mixed_formatting(mock_ctx, anyio_backend):
    """Test that phone numbers with mixed formatting are normalized to E.164."""
    events = await collect_events(transfer_call.func(mock_ctx, "+1 (415) 555-1234"))

    assert len(events) == 1
    assert isinstance(events[0], AgentTransferCall)
    # Should be normalized to E.164 without any formatting
    assert events[0].target_phone_number == "+14155551234"


async def test_transfer_call_normalizes_international_with_spaces(mock_ctx, anyio_backend):
    """Test that international numbers with spaces are normalized."""
    # UK number with spaces
    events = await collect_events(transfer_call.func(mock_ctx, "+44 20 7123 4567"))

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


async def test_end_call_default_eagerness(mock_ctx, anyio_backend):
    """Test that default end_call has normal eagerness."""
    assert end_call.eagerness == "normal"
    assert "Say goodbye" in end_call.description


async def test_end_call_yields_agent_end_call(mock_ctx, anyio_backend):
    """Test that end_call yields AgentEndCall event."""
    func_tool = end_call.as_function_tool()
    events = await collect_events(func_tool.func(mock_ctx))

    assert len(events) == 1
    assert isinstance(events[0], AgentEndCall)


async def test_end_call_low_eagerness(mock_ctx, anyio_backend):
    """Test that low eagerness has cautious description."""
    low_end_call = end_call(eagerness="low")

    assert low_end_call.eagerness == "low"
    assert "MUST first ask" in low_end_call.description
    assert "Never assume" in low_end_call.description


async def test_end_call_high_eagerness(mock_ctx, anyio_backend):
    """Test that high eagerness has prompt description."""
    high_end_call = end_call(eagerness="high")

    assert high_end_call.eagerness == "high"
    assert "promptly" in high_end_call.description
    assert "Don't ask follow-up" in high_end_call.description


async def test_end_call_custom_description(mock_ctx, anyio_backend):
    """Test that custom description overrides eagerness-based description."""
    custom_desc = "Only end when user says 'terminate'"
    custom_end_call = end_call(description=custom_desc)

    assert custom_end_call.description == custom_desc
    # Eagerness defaults to normal but description is overridden
    assert custom_end_call.eagerness == "normal"


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
    assert func_tool.tool_type == ToolType.PASSTHROUGH
    assert func_tool.is_background is False

    # Verify it's actually a FunctionTool instance (not duck-typed)
    from line.llm_agent.tools.utils import FunctionTool

    assert isinstance(func_tool, FunctionTool)


async def test_end_call_callable_returns_new_instance(mock_ctx, anyio_backend):
    """Test that calling end_call() returns a new configured instance."""
    configured = end_call(eagerness="low")

    # Should be a new instance
    assert configured is not end_call
    assert isinstance(configured, EndCallTool)

    # Original should be unchanged
    assert end_call.eagerness == "normal"
    assert configured.eagerness == "low"
