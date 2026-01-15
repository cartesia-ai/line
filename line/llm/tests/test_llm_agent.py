"""
Tests for LlmAgent tool call handling.

These tests mock the LLM provider to verify the agent loop correctly handles
loopback, passthrough, and handoff tools.

uv run pytest line/llm/tests/test_llm_agent.py -v
"""

import json
import pytest

# Use anyio for async test support with asyncio backend only (trio not installed)
pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]
from typing import Annotated, Any, AsyncIterable, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from line.llm.agent import (
    AgentEndCall,
    AgentHandoff,
    AgentOutput,
    CallStarted,
    OutputEvent,
    ToolCallOutput,
    ToolResultOutput,
    TurnContext,
    UserTranscript,
)
from line.llm.config import LlmConfig
from line.llm.function_tool import Field, FunctionTool, ToolType
from line.llm.llm_agent import LlmAgent
from line.llm.providers.base import LLM, LLMStream, Message, StreamChunk, ToolCall
from line.llm.function_tool import function_tool
from line.llm.tool_types import handoff_tool, loopback_tool, passthrough_tool


# =============================================================================
# Mock LLM Provider
# =============================================================================


class MockStream(LLMStream):
    """Mock stream that yields predefined chunks."""

    def __init__(self, chunks: List[StreamChunk]):
        self._chunks = chunks
        self._index = 0

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockLLM(LLM):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: List[List[StreamChunk]]):
        """
        Args:
            responses: List of responses, one per chat() call.
                       Each response is a list of StreamChunks.
        """
        super().__init__(model="mock-model")
        self._responses = responses
        self._call_count = 0
        self._recorded_messages: List[List[Message]] = []
        self._recorded_tools: List[Optional[List[FunctionTool]]] = []

    def chat(self, messages: List[Message], tools: Optional[List[FunctionTool]] = None, **kwargs) -> MockStream:
        # Record what was passed
        self._recorded_messages.append(messages.copy())
        self._recorded_tools.append(tools)

        # Return next response
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return MockStream(response)
        else:
            # Return empty response if no more predefined
            return MockStream([StreamChunk(is_final=True)])

    async def aclose(self):
        pass


# =============================================================================
# Helper to create LlmAgent with mock LLM
# =============================================================================


def create_agent_with_mock(
    responses: List[List[StreamChunk]],
    tools: List[FunctionTool] = None,
    config: LlmConfig = None,
    max_tool_iterations: int = 10,
) -> tuple[LlmAgent, MockLLM]:
    """Create an LlmAgent with a MockLLM injected.

    Uses "gpt-4o" as the model name to pass provider detection,
    then replaces the LLM instance with a mock.
    """
    mock_llm = MockLLM(responses)

    agent = LlmAgent(
        model="gpt-4o",  # Use real model name for provider detection
        tools=tools or [],
        config=config or LlmConfig(),
        max_tool_iterations=max_tool_iterations,
    )
    # Inject mock LLM to replace the real one
    agent._llm = mock_llm

    return agent, mock_llm


# =============================================================================
# Helper to collect all outputs from agent
# =============================================================================


async def collect_outputs(agent: LlmAgent, ctx: TurnContext, event) -> List[OutputEvent]:
    """Collect all outputs from agent.process()."""
    outputs = []
    async for output in agent.process(ctx, event):
        outputs.append(output)
    return outputs


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def turn_context():
    """Create a basic TurnContext."""
    return TurnContext(
        call_id="test-call",
        turn_id="test-turn",
        conversation_history=[],
        metadata={},
    )


# =============================================================================
# Tests: Basic Text Response (No Tools)
# =============================================================================


async def test_simple_text_response(turn_context):
    """Test agent returns text when LLM generates no tool calls."""
    responses = [
        [
            StreamChunk(text="Hello "),
            StreamChunk(text="world!"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="Hi"))

    # Should have two AgentOutput events
    assert len(outputs) == 2
    assert isinstance(outputs[0], AgentOutput)
    assert outputs[0].text == "Hello "
    assert isinstance(outputs[1], AgentOutput)
    assert outputs[1].text == "world!"

    # LLM should have been called once with user message
    assert mock_llm._call_count == 1
    assert len(mock_llm._recorded_messages[0]) == 1
    assert mock_llm._recorded_messages[0][0].role == "user"
    assert mock_llm._recorded_messages[0][0].content == "Hi"


# =============================================================================
# Tests: Loopback Tool
# =============================================================================


async def test_loopback_tool_feeds_result_back_to_llm(turn_context):
    """Test that loopback tool results are fed back to the LLM."""

    # Define a loopback tool
    @loopback_tool
    async def get_weather(ctx, city: Annotated[str, Field(description="City name")]) -> str:
        """Get weather for a city."""
        return f"72°F in {city}"

    # Response 1: LLM calls the tool
    # Response 2: LLM generates final text after seeing tool result
    responses = [
        [
            StreamChunk(text="Let me check. "),
            StreamChunk(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments='{"city": "NYC"}',
                        is_complete=True,
                    )
                ]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="The weather in NYC is 72°F."),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[get_weather])

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="What's the weather in NYC?"))

    # Expected outputs:
    # 1. AgentOutput "Let me check. "
    # 2. ToolCallOutput
    # 3. ToolResultOutput
    # 4. AgentOutput "The weather in NYC is 72°F."
    assert len(outputs) == 4

    assert isinstance(outputs[0], AgentOutput)
    assert outputs[0].text == "Let me check. "

    assert isinstance(outputs[1], ToolCallOutput)
    assert outputs[1].tool_name == "get_weather"
    assert outputs[1].tool_args == {"city": "NYC"}

    assert isinstance(outputs[2], ToolResultOutput)
    assert outputs[2].tool_name == "get_weather"
    assert outputs[2].result == "72°F in NYC"

    assert isinstance(outputs[3], AgentOutput)
    assert outputs[3].text == "The weather in NYC is 72°F."

    # LLM should have been called twice
    assert mock_llm._call_count == 2

    # Second call should include tool result in messages
    second_call_messages = mock_llm._recorded_messages[1]
    # Should have: user message, assistant with tool call, tool result
    assert len(second_call_messages) >= 3

    # Find the tool result message
    tool_result_msg = next((m for m in second_call_messages if m.role == "tool"), None)
    assert tool_result_msg is not None
    assert tool_result_msg.content == "72°F in NYC"
    assert tool_result_msg.tool_call_id == "call_1"


async def test_loopback_tool_multiple_iterations(turn_context):
    """Test multiple loopback tool calls in sequence."""

    @loopback_tool
    async def get_weather(ctx, city: Annotated[str, Field(description="City")]) -> str:
        """Get weather."""
        temps = {"NYC": "72°F", "LA": "85°F"}
        return temps.get(city, "Unknown")

    responses = [
        # First call: LLM requests NYC weather
        [
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments='{"city": "NYC"}', is_complete=True)
                ]
            ),
            StreamChunk(is_final=True),
        ],
        # Second call: LLM requests LA weather
        [
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_2", name="get_weather", arguments='{"city": "LA"}', is_complete=True)
                ]
            ),
            StreamChunk(is_final=True),
        ],
        # Third call: LLM generates final response
        [
            StreamChunk(text="NYC is 72°F and LA is 85°F."),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[get_weather])

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="Weather in NYC and LA?"))

    # Should have: ToolCall, ToolResult, ToolCall, ToolResult, AgentOutput
    tool_calls = [o for o in outputs if isinstance(o, ToolCallOutput)]
    tool_results = [o for o in outputs if isinstance(o, ToolResultOutput)]
    agent_outputs = [o for o in outputs if isinstance(o, AgentOutput)]

    assert len(tool_calls) == 2
    assert len(tool_results) == 2
    assert len(agent_outputs) == 1

    assert tool_results[0].result == "72°F"
    assert tool_results[1].result == "85°F"

    # LLM called 3 times
    assert mock_llm._call_count == 3


# =============================================================================
# Tests: Passthrough Tool
# =============================================================================


async def test_passthrough_tool_bypasses_llm(turn_context):
    """Test that passthrough tool events go directly to output, not back to LLM."""

    @passthrough_tool
    async def end_call(ctx, message: Annotated[str, Field(description="Goodbye message")]):
        """End the call."""
        yield AgentOutput(text=message)
        yield AgentEndCall(reason="user requested")

    responses = [
        [
            StreamChunk(text="Goodbye! "),
            StreamChunk(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="end_call",
                        arguments='{"message": "Have a great day!"}',
                        is_complete=True,
                    )
                ]
            ),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[end_call])

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="Bye"))

    # Expected outputs:
    # 1. AgentOutput "Goodbye! "
    # 2. ToolCallOutput
    # 3. AgentOutput "Have a great day!" (from passthrough)
    # 4. AgentEndCall (from passthrough)
    # 5. ToolResultOutput

    assert len(outputs) == 5

    assert isinstance(outputs[0], AgentOutput)
    assert outputs[0].text == "Goodbye! "

    assert isinstance(outputs[1], ToolCallOutput)
    assert outputs[1].tool_name == "end_call"

    assert isinstance(outputs[2], AgentOutput)
    assert outputs[2].text == "Have a great day!"

    assert isinstance(outputs[3], AgentEndCall)
    assert outputs[3].reason == "user requested"

    assert isinstance(outputs[4], ToolResultOutput)

    # LLM should only be called ONCE - passthrough doesn't loop back
    assert mock_llm._call_count == 1


# =============================================================================
# Tests: Handoff Tool
# =============================================================================


async def test_handoff_tool_emits_handoff_event(turn_context):
    """Test that handoff tool emits AgentHandoff event."""

    class BillingAgent:
        pass

    @handoff_tool
    async def transfer_to_billing(ctx, reason: Annotated[str, Field(description="Reason")]):
        """Transfer to billing department."""
        return BillingAgent()

    responses = [
        [
            StreamChunk(text="Transferring you to billing. "),
            StreamChunk(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="transfer_to_billing",
                        arguments='{"reason": "payment issue"}',
                        is_complete=True,
                    )
                ]
            ),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[transfer_to_billing])

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="I have a billing question"))

    # Expected: AgentOutput, ToolCallOutput, AgentHandoff, ToolResultOutput
    assert len(outputs) == 4

    assert isinstance(outputs[0], AgentOutput)
    assert isinstance(outputs[1], ToolCallOutput)

    assert isinstance(outputs[2], AgentHandoff)
    assert isinstance(outputs[2].target_agent, BillingAgent)
    assert "transfer_to_billing" in outputs[2].reason

    assert isinstance(outputs[3], ToolResultOutput)

    # LLM called only once - handoff doesn't loop back
    assert mock_llm._call_count == 1


# =============================================================================
# Tests: Max Tool Iterations
# =============================================================================


async def test_max_tool_iterations_prevents_infinite_loop(turn_context):
    """Test that max_tool_iterations stops runaway loops."""

    @loopback_tool
    async def infinite_tool(ctx) -> str:
        """Tool that always gets called again."""
        return "call me again"

    # LLM always calls the tool (would loop forever without limit)
    def make_tool_response():
        return [
            StreamChunk(
                tool_calls=[
                    ToolCall(id=f"call_{i}", name="infinite_tool", arguments="{}", is_complete=True)
                    for i in range(1)
                ]
            ),
            StreamChunk(is_final=True),
        ]

    # Create many responses - more than max_tool_iterations
    responses = [make_tool_response() for _ in range(20)]

    agent, mock_llm = create_agent_with_mock(
        responses, tools=[infinite_tool], max_tool_iterations=3
    )

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="Start"))

    # Should have stopped after 3 iterations
    assert mock_llm._call_count == 3

    # Should have 3 tool calls and 3 results
    tool_calls = [o for o in outputs if isinstance(o, ToolCallOutput)]
    tool_results = [o for o in outputs if isinstance(o, ToolResultOutput)]

    assert len(tool_calls) == 3
    assert len(tool_results) == 3


# =============================================================================
# Tests: Introduction Message
# =============================================================================


async def test_introduction_sent_on_call_started(turn_context):
    """Test that introduction is sent on CallStarted event."""
    config = LlmConfig(introduction="Hello! How can I help you?")

    agent, _ = create_agent_with_mock([], config=config)

    outputs = await collect_outputs(agent, turn_context, CallStarted(call_id="test"))

    assert len(outputs) == 1
    assert isinstance(outputs[0], AgentOutput)
    assert outputs[0].text == "Hello! How can I help you?"


async def test_introduction_only_sent_once(turn_context):
    """Test that introduction is not sent on subsequent CallStarted events."""
    config = LlmConfig(introduction="Hello!")

    agent, _ = create_agent_with_mock([], config=config)

    # First CallStarted
    outputs1 = await collect_outputs(agent, turn_context, CallStarted(call_id="test"))
    assert len(outputs1) == 1
    assert outputs1[0].text == "Hello!"

    # Second CallStarted - should not send intro again
    outputs2 = await collect_outputs(agent, turn_context, CallStarted(call_id="test"))
    assert len(outputs2) == 0


# =============================================================================
# Tests: Tool Error Handling
# =============================================================================


async def test_tool_error_is_captured(turn_context):
    """Test that tool execution errors are captured in ToolResultOutput."""

    @loopback_tool
    async def failing_tool(ctx) -> str:
        """Tool that always fails."""
        raise ValueError("Something went wrong!")

    responses = [
        [
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_1", name="failing_tool", arguments="{}", is_complete=True)
                ]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Sorry, there was an error."),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[failing_tool])

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="Do something"))

    # Find the ToolResultOutput
    tool_result = next((o for o in outputs if isinstance(o, ToolResultOutput)), None)
    assert tool_result is not None
    assert tool_result.error is not None
    assert "Something went wrong!" in tool_result.error
    assert tool_result.result is None


# =============================================================================
# Tests: Conversation History in Context
# =============================================================================


async def test_conversation_history_passed_to_llm(turn_context):
    """Test that conversation history is included in messages to LLM."""
    # Pre-populate conversation history
    turn_context.conversation_history = [
        UserTranscript(text="Hi"),
        AgentOutput(text="Hello!"),
    ]

    responses = [
        [StreamChunk(text="I'm doing well!"), StreamChunk(is_final=True)]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    await collect_outputs(agent, turn_context, UserTranscript(text="How are you?"))

    # Check messages sent to LLM
    messages = mock_llm._recorded_messages[0]

    # Should have: Hi, Hello!, How are you?
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content == "Hi"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hello!"
    assert messages[2].role == "user"
    assert messages[2].content == "How are you?"


# =============================================================================
# Tests: Streaming Tool Calls (Incremental Arguments)
# =============================================================================


async def test_streaming_tool_call_accumulation(turn_context):
    """Test that tool call arguments are accumulated across chunks."""

    @loopback_tool
    async def greet(ctx, name: Annotated[str, Field(description="Name")]) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    # Tool call arrives in chunks (simulating streaming)
    responses = [
        [
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_1", name="greet", arguments='{"na', is_complete=False)
                ]
            ),
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_1", name="greet", arguments='me": "', is_complete=False)
                ]
            ),
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_1", name="greet", arguments='Alice"}', is_complete=True)
                ]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Hello, Alice!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[greet])

    outputs = await collect_outputs(agent, turn_context, UserTranscript(text="Greet Alice"))

    # Tool should have been called with complete arguments
    tool_result = next((o for o in outputs if isinstance(o, ToolResultOutput)), None)
    assert tool_result is not None
    assert tool_result.result == "Hello, Alice!"

    # Tool call output should have complete args
    tool_call = next((o for o in outputs if isinstance(o, ToolCallOutput)), None)
    assert tool_call is not None
    assert tool_call.tool_args == {"name": "Alice"}
