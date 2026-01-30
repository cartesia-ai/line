"""
Tests for FunctionTool validation and parameter extraction.

uv run pytest tests/test_tool_utils.py -v
"""

from typing import Annotated, Literal, Optional

import pytest

from line.events import AgentSendText
from line.llm_agent.schema_converter import function_tool_to_openai
from line.llm_agent.tools.decorators import handoff_tool, loopback_tool, passthrough_tool

# =============================================================================
# Tests: Tool Signature Validation
# =============================================================================


def test_loopback_tool_missing_ctx_raises_error():
    """Test that loopback tool without ctx parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx' or 'context' as first parameter"):

        @loopback_tool
        async def bad_tool(city: str) -> str:
            """Missing ctx parameter."""
            return city


def test_loopback_tool_wrong_first_param_raises_error():
    """Test that loopback tool with wrong first parameter name raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx' or 'context' as first parameter"):

        @loopback_tool
        async def bad_tool(foo, city: str) -> str:
            """Wrong first parameter name."""
            return city


def test_passthrough_tool_missing_ctx_raises_error():
    """Test that passthrough tool without ctx parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx'"):

        @passthrough_tool
        async def bad_tool():
            """Missing ctx parameter."""
            yield AgentSendText(text="Hello")


def test_handoff_tool_missing_ctx_raises_error():
    """Test that handoff tool without ctx parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'ctx' or 'context' as first parameter"):

        @handoff_tool
        async def bad_tool(event):
            """Missing ctx parameter."""
            yield AgentSendText(text="Hello")


def test_handoff_tool_missing_event_raises_error():
    """Test that handoff tool without event parameter raises TypeError."""
    with pytest.raises(TypeError, match="must have 'event' parameter"):

        @handoff_tool
        async def bad_tool(ctx):
            """Missing event parameter."""
            yield AgentSendText(text="Hello")


# =============================================================================
# Tests: Valid Tool Definitions
# =============================================================================


def test_loopback_tool_with_ctx_succeeds():
    """Test that loopback tool with ctx parameter succeeds."""

    @loopback_tool
    async def good_tool(ctx, city: Annotated[str, "City name"]) -> str:
        """Valid loopback tool."""
        return f"Weather in {city}"

    assert good_tool.name == "good_tool"
    assert good_tool.tool_type.value == "loopback"
    assert "city" in good_tool.parameters
    assert good_tool.parameters["city"].description == "City name"


def test_loopback_tool_with_context_alias_succeeds():
    """Test that loopback tool can use 'context' instead of 'ctx'."""

    @loopback_tool
    async def good_tool(context, city: str) -> str:
        """Uses 'context' instead of 'ctx'."""
        return city

    assert good_tool.name == "good_tool"
    assert "city" in good_tool.parameters


def test_passthrough_tool_with_ctx_succeeds():
    """Test that passthrough tool with ctx parameter succeeds."""

    @passthrough_tool
    async def good_tool(ctx, message: Annotated[str, "Message"]):
        """Valid passthrough tool."""
        yield AgentSendText(text=message)

    assert good_tool.name == "good_tool"
    assert good_tool.tool_type.value == "passthrough"
    assert "message" in good_tool.parameters


def test_handoff_tool_with_ctx_and_event_succeeds():
    """Test that handoff tool with both ctx and event parameters succeeds."""

    @handoff_tool
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

    @handoff_tool
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

    @loopback_tool
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

    @loopback_tool
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

    @loopback_tool
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

    @loopback_tool
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


def test_tool_name_and_description_from_function():
    """Test that name comes from __name__ and description from __doc__."""

    @loopback_tool
    async def my_tool(ctx) -> str:
        """This is the tool description."""
        return "done"

    assert my_tool.name == "my_tool"
    assert my_tool.description == "This is the tool description."


# =============================================================================
# Tests: Method-based Tools (Descriptor Protocol)
# =============================================================================


def test_method_tool_binding():
    """Test that tool decorators work on class methods via descriptor protocol."""

    class MyAgent:
        def __init__(self, prefix: str):
            self.prefix = prefix

        @loopback_tool
        async def greet(self, ctx, name: Annotated[str, "Name to greet"]) -> str:
            """Greet someone with a prefix."""
            return f"{self.prefix} {name}!"

    agent = MyAgent(prefix="Hello")

    # Access the tool from the instance - should get a bound FunctionTool
    bound_tool = agent.greet

    # Check it's still a FunctionTool
    assert bound_tool.name == "greet"
    assert bound_tool.tool_type.value == "loopback"
    assert "name" in bound_tool.parameters
    # 'self' should NOT be in parameters
    assert "self" not in bound_tool.parameters


@pytest.mark.asyncio
async def test_method_tool_execution():
    """Test that bound method tools can be executed with self bound."""

    class Counter:
        def __init__(self):
            self.count = 0

        @loopback_tool
        async def increment(self, ctx, amount: Annotated[int, "Amount to add"] = 1) -> int:
            """Increment the counter."""
            self.count += amount
            return self.count

    counter = Counter()
    bound_tool = counter.increment

    # Execute the tool - self should be automatically bound
    result = await bound_tool.func(None, amount=5)  # ctx=None for test
    assert result == 5
    assert counter.count == 5

    # Execute again
    result = await bound_tool.func(None, amount=3)
    assert result == 8
    assert counter.count == 8


def test_method_passthrough_tool():
    """Test that passthrough tool decorator works on methods."""

    class FormFiller:
        def __init__(self, form_id: str):
            self.form_id = form_id

        @passthrough_tool
        async def submit(self, ctx):
            """Submit the form."""
            yield AgentSendText(text=f"Submitting form {self.form_id}")

    filler = FormFiller(form_id="abc123")
    bound_tool = filler.submit

    assert bound_tool.name == "submit"
    assert bound_tool.tool_type.value == "passthrough"


@pytest.mark.asyncio
async def test_method_passthrough_tool_execution():
    """Test that bound passthrough method tools yield events correctly."""

    class FormFiller:
        def __init__(self, form_id: str):
            self.form_id = form_id

        @passthrough_tool
        async def submit(self, ctx):
            """Submit the form."""
            yield AgentSendText(text=f"Submitting form {self.form_id}")

    filler = FormFiller(form_id="xyz789")
    bound_tool = filler.submit

    # Execute and collect events
    events = []
    async for event in bound_tool.func(None):  # ctx=None for test
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], AgentSendText)
    assert events[0].text == "Submitting form xyz789"


@pytest.mark.asyncio
async def test_multiple_instances_get_separate_bound_tools():
    """Test that different instances get separately bound tools."""

    class Counter:
        def __init__(self, start: int):
            self.value = start

        @loopback_tool
        async def get_value(self, ctx) -> int:
            """Get current value."""
            return self.value

    counter1 = Counter(start=10)
    counter2 = Counter(start=20)

    tool1 = counter1.get_value
    tool2 = counter2.get_value

    # Should be different bound tools with different behavior
    assert tool1 is not tool2

    # Verify they're bound to different instances by executing
    result1 = await tool1.func(None)  # ctx=None for test
    result2 = await tool2.func(None)

    assert result1 == 10
    assert result2 == 20
