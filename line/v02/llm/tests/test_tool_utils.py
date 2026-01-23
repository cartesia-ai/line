"""
Tests for FunctionTool validation and parameter extraction.

uv run pytest line/v02/llm/tests/test_function_tool.py -v
"""

from typing import Annotated, Literal, Optional

import pytest

from line.v02.llm.agent import AgentSendText
from line.v02.llm.schema_converter import function_tool_to_openai
from line.v02.llm.tool_types import handoff_tool, loopback_tool, passthrough_tool

# =============================================================================
# Tests: Tool Signature Validation
# =============================================================================


def test_loopback_tool_missing_ctx_raises_error():
    """Test that loopback tool without ctx parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx' or 'context' as first parameter"):

        @loopback_tool()
        async def bad_tool(city: str) -> str:
            """Missing ctx parameter."""
            return city


def test_loopback_tool_wrong_first_param_raises_error():
    """Test that loopback tool with wrong first parameter name raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx' or 'context' as first parameter"):

        @loopback_tool()
        async def bad_tool(foo, city: str) -> str:
            """Wrong first parameter name."""
            return city


def test_passthrough_tool_missing_ctx_raises_error():
    """Test that passthrough tool without ctx parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx'"):

        @passthrough_tool()
        async def bad_tool():
            """Missing ctx parameter."""
            yield AgentSendText(text="Hello")


def test_handoff_tool_missing_ctx_raises_error():
    """Test that handoff tool without ctx parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx' or 'context' as first parameter"):

        @handoff_tool()
        async def bad_tool(event):
            """Missing ctx parameter."""
            yield AgentSendText(text="Hello")


def test_handoff_tool_missing_event_raises_error():
    """Test that handoff tool without event parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'event' parameter"):

        @handoff_tool()
        async def bad_tool(ctx):
            """Missing event parameter."""
            yield AgentSendText(text="Hello")


# =============================================================================
# Tests: Valid Tool Definitions
# =============================================================================


def test_loopback_tool_with_ctx_succeeds():
    """Test that loopback tool with ctx parameter succeeds."""

    @loopback_tool()
    async def good_tool(ctx, city: Annotated[str, "City name"]) -> str:
        """Valid loopback tool."""
        return f"Weather in {city}"

    assert good_tool.name == "good_tool"
    assert good_tool.tool_type.value == "loopback"
    assert "city" in good_tool.parameters
    assert good_tool.parameters["city"].description == "City name"


def test_loopback_tool_with_context_alias_succeeds():
    """Test that loopback tool can use 'context' instead of 'ctx'."""

    @loopback_tool()
    async def good_tool(context, city: str) -> str:
        """Uses 'context' instead of 'ctx'."""
        return city

    assert good_tool.name == "good_tool"
    assert "city" in good_tool.parameters


def test_passthrough_tool_with_ctx_succeeds():
    """Test that passthrough tool with ctx parameter succeeds."""

    @passthrough_tool()
    async def good_tool(ctx, message: Annotated[str, "Message"]):
        """Valid passthrough tool."""
        yield AgentSendText(text=message)

    assert good_tool.name == "good_tool"
    assert good_tool.tool_type.value == "passthrough"
    assert "message" in good_tool.parameters


def test_handoff_tool_with_ctx_and_event_succeeds():
    """Test that handoff tool with both ctx and event parameters succeeds."""

    @handoff_tool()
    async def good_tool(ctx, reason: Annotated[str, "Reason"], event):
        """Valid handoff tool."""
        yield AgentSendText(text="Transferring...")

    assert good_tool.name == "good_tool"
    assert good_tool.tool_type.value == "handoff"
    # event should NOT be in parameters (filtered out for LLM schema)
    assert "event" not in good_tool.parameters
    assert "reason" in good_tool.parameters


def test_handoff_tool_event_not_in_parameters():
    """Test that event parameter is filtered out of handoff tool parameters."""

    @handoff_tool()
    async def transfer(ctx, department: Annotated[str, "Dept"], event):
        """Transfer to department."""
        pass

    # Only 'department' should be in parameters, not 'event' or 'ctx'
    assert list(transfer.parameters.keys()) == ["department"]


# =============================================================================
# Tests: Parameter Extraction
# =============================================================================


def test_parameter_with_default_is_optional():
    """Test that parameters with defaults are marked as not required."""

    @loopback_tool()
    async def tool_with_default(
        ctx,
        required_param: Annotated[str, "Required"],
        optional_param: Annotated[int, "Optional"] = 10,
    ) -> str:
        """Tool with optional parameter."""
        return "done"

    assert tool_with_default.parameters["required_param"].required is True
    assert tool_with_default.parameters["optional_param"].required is False
    assert tool_with_default.parameters["optional_param"].default == 10


def test_optional_type_without_default_is_still_required():
    """Test that Optional[X] types without defaults are still required.

    Optional[X] only affects the type (allows None), not whether the param
    is required. Use a default value to make a param optional.
    """

    @loopback_tool()
    async def tool_with_optional(
        ctx,
        required_param: Annotated[str, "Required"],
        optional_type_param: Annotated[Optional[str], "Optional type but no default"],
    ) -> str:
        """Tool with Optional type parameter but no default."""
        return "done"

    assert tool_with_optional.parameters["required_param"].required is True
    # Optional[X] does NOT make param optional - only a default value does
    assert tool_with_optional.parameters["optional_type_param"].required is True


def test_optional_type_with_default_is_optional():
    """Test that Optional[X] with a default is not required."""

    @loopback_tool()
    async def tool_with_optional_default(
        ctx,
        optional_param: Annotated[Optional[str], "Optional with default"] = None,
    ) -> str:
        """Tool with Optional type and default."""
        return "done"

    assert tool_with_optional_default.parameters["optional_param"].required is False
    assert tool_with_optional_default.parameters["optional_param"].default is None


def test_parameter_with_literal_enum():
    """Test that Literal types create enum constraints."""

    @loopback_tool()
    async def tool_with_enum(
        ctx,
        category: Annotated[Literal["a", "b", "c"], "Category"],
    ) -> str:
        """Tool with enum parameter."""
        return category

    # Check that the schema has enum values
    schema = function_tool_to_openai(tool_with_enum)
    props = schema["function"]["parameters"]["properties"]
    assert props["category"]["enum"] == ["a", "b", "c"]


def test_custom_tool_name_and_description():
    """Test that custom name and description override function defaults."""

    @loopback_tool(name="custom_name", description="Custom description")
    async def original_name(ctx) -> str:
        """Original docstring."""
        return "done"

    assert original_name.name == "custom_name"
    assert original_name.description == "Custom description"
