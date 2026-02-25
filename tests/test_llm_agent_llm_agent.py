"""
Tests for LlmAgent tool call handling.

These tests mock the LLM provider to verify the agent loop correctly handles
loopback, passthrough, and handoff tools.

uv run pytest tests/test_llm_agent.py -v
"""

from typing import Annotated, List, Optional

import pytest

from line.agent import TurnEnv
from line.events import (
    AgentEndCall,
    AgentHandedOff,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    CustomHistoryEntry,
    LogMetric,
    OutputEvent,
    UserTextSent,
)
from line.llm_agent.config import LlmConfig
from line.llm_agent.llm_agent import LlmAgent
from line.llm_agent.provider import Message, StreamChunk, ToolCall
from line.llm_agent.tools.decorators import handoff_tool, loopback_tool, passthrough_tool
from line.llm_agent.tools.utils import FunctionTool

# Use anyio for async test support with asyncio backend only (trio not installed)
pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]

# =============================================================================
# Mock LLM Provider
# =============================================================================


class MockStream:
    """Mock stream that yields chunks, accumulating tool call arguments like the real provider."""

    def __init__(self, chunks: List[StreamChunk]):
        self._chunks = chunks

    async def __aiter__(self):
        # Accumulate tool calls like the real provider does
        accumulated_tools: dict = {}
        for chunk in self._chunks:
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    if tc.id not in accumulated_tools:
                        accumulated_tools[tc.id] = ToolCall(
                            id=tc.id, name=tc.name, arguments=tc.arguments, is_complete=tc.is_complete
                        )
                    else:
                        # Accumulate arguments (like real provider for incremental streaming)
                        existing = accumulated_tools[tc.id]
                        if not existing.arguments.endswith("}"):
                            existing.arguments += tc.arguments
                        existing.is_complete = tc.is_complete
                # Yield with accumulated tool calls
                yield StreamChunk(
                    text=chunk.text,
                    tool_calls=list(accumulated_tools.values()),
                    is_final=chunk.is_final,
                )
            else:
                yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockLLM:
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: List[List[StreamChunk]]):
        self._responses = responses
        self._call_count = 0
        self._recorded_messages: List[List[Message]] = []
        self._recorded_tools: List[Optional[List[FunctionTool]]] = []

    async def chat(
        self, messages: List[Message], tools: Optional[List[FunctionTool]] = None, **kwargs
    ) -> MockStream:
        self._recorded_messages.append(messages.copy())
        self._recorded_tools.append(tools)

        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return MockStream(response)
        else:
            return MockStream([StreamChunk(is_final=True)])

    async def warmup(self, config=None):
        pass

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
        api_key="test-key",
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


async def collect_outputs(
    agent: LlmAgent, env: TurnEnv, event, include_metrics: bool = False, **kwargs
) -> List[OutputEvent]:
    """Collect all outputs from agent.process().

    Args:
        include_metrics: If False (default), filters out LogMetric events.
    """
    outputs = []
    async for output in agent.process(env, event, **kwargs):
        if not include_metrics and isinstance(output, LogMetric):
            continue
        outputs.append(output)
    return outputs


# =============================================================================
# Helper to build messages with explicit history state
# =============================================================================


async def build_messages_with(agent, input_history, local_history, current_event_id):
    """Set up agent.history state and call _build_messages().

    This replaces the old pattern of calling agent._build_messages(input_history, local_history, eid)
    directly, since _build_messages now reads from agent.history.
    """
    agent.history._input_history = input_history
    agent.history._local_history = local_history
    agent.history._current_event_id = current_event_id
    agent.history._cache = None
    return await agent._build_messages()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def turn_env():
    """Create a basic TurnEnv."""
    return TurnEnv()


# =============================================================================
# Tests: Basic Text Response (No Tools)
# =============================================================================


async def test_simple_text_response(turn_env):
    """Test agent returns text when LLM generates no tool calls."""
    responses = [
        [
            StreamChunk(text="Hello "),
            StreamChunk(text="world!"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Hi", history=[UserTextSent(content="Hi")])
    )

    # Should have two AgentSendText events
    assert len(outputs) == 2
    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Hello "
    assert isinstance(outputs[1], AgentSendText)
    assert outputs[1].text == "world!"

    # LLM should have been called once with user message from history
    assert mock_llm._call_count == 1
    assert len(mock_llm._recorded_messages[0]) == 1
    assert mock_llm._recorded_messages[0][0].role == "user"
    assert mock_llm._recorded_messages[0][0].content == "Hi"


async def test_timing_metrics_emitted(turn_env):
    """Test that TTFT and first AgentSendText timing metrics are emitted."""
    responses = [
        [
            StreamChunk(text="Hello "),
            StreamChunk(text="world!"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    # Collect outputs including metrics
    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Hi", history=[UserTextSent(content="Hi")]),
        include_metrics=True,
    )

    # Should have 4 events: 2 LogMetric + 2 AgentSendText
    assert len(outputs) == 4

    # First two should be LogMetric events
    assert isinstance(outputs[0], LogMetric)
    assert outputs[0].name == "llm_first_chunk_ms"
    assert isinstance(outputs[0].value, float)
    assert outputs[0].value >= 0

    assert isinstance(outputs[1], LogMetric)
    assert outputs[1].name == "llm_first_text_ms"
    assert isinstance(outputs[1].value, float)
    assert outputs[1].value >= 0

    # First chunk should be <= first text time (or very close)
    assert outputs[0].value <= outputs[1].value + 1  # Allow 1ms tolerance

    # Last two should be AgentSendText events
    assert isinstance(outputs[2], AgentSendText)
    assert outputs[2].text == "Hello "
    assert isinstance(outputs[3], AgentSendText)
    assert outputs[3].text == "world!"


# =============================================================================
# Tests: Loopback Tool
# =============================================================================


async def test_loopback_tool_feeds_result_back_to_llm(turn_env):
    """Test that loopback tool results are fed back to the LLM."""

    # Define a loopback tool
    @loopback_tool
    async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
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

    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(
            content="What's the weather in NYC?",
            history=[UserTextSent(content="What's the weather in NYC?")],
        ),
    )

    # Expected outputs:
    # 1. AgentSendText "Let me check. "
    # 2. AgentToolCalled
    # 3. AgentToolReturned
    # 4. AgentSendText "The weather in NYC is 72°F."
    assert len(outputs) == 4

    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Let me check. "

    assert isinstance(outputs[1], AgentToolCalled)
    assert outputs[1].tool_name == "get_weather"
    assert outputs[1].tool_args == {"city": "NYC"}

    assert isinstance(outputs[2], AgentToolReturned)
    assert outputs[2].tool_name == "get_weather"
    assert outputs[2].result == "72°F in NYC"

    assert isinstance(outputs[3], AgentSendText)
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
    # Tool call ID now uses -n suffix format for all loopback tool results
    assert tool_result_msg.tool_call_id == "call_1-0"


async def test_loopback_tool_multiple_iterations(turn_env):
    """Test multiple loopback tool calls in sequence."""

    @loopback_tool
    async def get_weather(ctx, city: Annotated[str, "City"]) -> str:
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

    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(
            content="Weather in NYC and LA?", history=[UserTextSent(content="Weather in NYC and LA?")]
        ),
    )

    # Should have: ToolCall, ToolResult, ToolCall, ToolResult, AgentSendText
    tool_calls = [o for o in outputs if isinstance(o, AgentToolCalled)]
    tool_results = [o for o in outputs if isinstance(o, AgentToolReturned)]
    agent_outputs = [o for o in outputs if isinstance(o, AgentSendText)]

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


async def test_passthrough_tool_bypasses_llm(turn_env):
    """Test that passthrough tool events go directly to output, not back to LLM."""

    @passthrough_tool
    async def end_call(ctx, message: Annotated[str, "Goodbye message"]):
        """End the call."""
        yield AgentSendText(text=message)
        yield AgentEndCall()

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

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Bye", history=[UserTextSent(content="Bye")])
    )

    # Expected outputs:
    # 1. AgentSendText "Goodbye! "
    # 2. AgentToolCalled
    # 3. AgentSendText "Have a great day!" (from passthrough)
    # 4. AgentEndCall (from passthrough)
    # 5. AgentToolReturned (emitted at the end of tool execution)

    assert len(outputs) == 5

    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Goodbye! "

    assert isinstance(outputs[1], AgentToolCalled)
    assert outputs[1].tool_name == "end_call"

    assert isinstance(outputs[2], AgentSendText)
    assert outputs[2].text == "Have a great day!"

    assert isinstance(outputs[3], AgentEndCall)

    assert isinstance(outputs[4], AgentToolReturned)
    assert outputs[4].result == "success"

    # LLM should only be called ONCE - passthrough doesn't loop back
    assert mock_llm._call_count == 1


# =============================================================================
# Tests: Handoff Tool
# =============================================================================


async def test_handoff_delegates_subsequent_calls(turn_env):
    """Test that after handoff, subsequent process() calls go to the handoff target."""

    handoff_events_received = []

    class BillingAgent:
        """A mock billing agent that tracks calls."""

        async def process(self, env, event):
            handoff_events_received.append(event)
            yield AgentSendText(text=f"Billing received: {type(event).__name__}")

    billing_agent = BillingAgent()

    @handoff_tool
    async def transfer_to_billing(ctx, event):
        """Transfer to billing."""
        async for output in billing_agent.process(ctx.turn_env, event):
            yield output

    responses = [
        [
            StreamChunk(
                tool_calls=[
                    ToolCall(id="call_1", name="transfer_to_billing", arguments="{}", is_complete=True)
                ]
            ),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[transfer_to_billing])

    # First call triggers handoff
    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Transfer me", history=[UserTextSent(content="Transfer me")]),
    )

    # Second call should go to billing agent, not the LLM
    outputs2 = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Follow up question", history=[UserTextSent(content="Follow up question")]),
    )

    assert len(outputs2) == 1
    assert isinstance(outputs2[0], AgentSendText)
    assert outputs2[0].text == "Billing received: UserTextSent"

    # LLM should NOT have been called again
    assert mock_llm._call_count == 1

    # Billing agent should have received both events (initial handoff + follow-up)
    assert len(handoff_events_received) == 2
    assert isinstance(handoff_events_received[0], AgentHandedOff)
    assert isinstance(handoff_events_received[1], UserTextSent)


# =============================================================================
# Tests: Max Tool Iterations
# =============================================================================


async def test_max_tool_iterations_prevents_infinite_loop(turn_env):
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

    agent, mock_llm = create_agent_with_mock(responses, tools=[infinite_tool], max_tool_iterations=3)

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Start", history=[UserTextSent(content="Start")])
    )

    # Should have stopped after 3 iterations
    assert mock_llm._call_count == 3

    # Should have 3 tool calls and 3 results
    tool_calls = [o for o in outputs if isinstance(o, AgentToolCalled)]
    tool_results = [o for o in outputs if isinstance(o, AgentToolReturned)]

    assert len(tool_calls) == 3
    assert len(tool_results) == 3


# =============================================================================
# Tests: Introduction Message
# =============================================================================


async def test_introduction_sent_on_call_started(turn_env):
    """Test that introduction is sent on CallStarted event."""
    config = LlmConfig(introduction="Hello! How can I help you?")

    agent, _ = create_agent_with_mock([], config=config)

    outputs = await collect_outputs(agent, turn_env, CallStarted())

    assert len(outputs) == 1
    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Hello! How can I help you?"


async def test_introduction_only_sent_once(turn_env):
    """Test that introduction is not sent on subsequent CallStarted events."""
    config = LlmConfig(introduction="Hello!")

    agent, _ = create_agent_with_mock([], config=config)

    # First CallStarted
    outputs1 = await collect_outputs(agent, turn_env, CallStarted())
    assert len(outputs1) == 1
    assert outputs1[0].text == "Hello!"

    # Second CallStarted - should not send intro again
    outputs2 = await collect_outputs(agent, turn_env, CallStarted())
    assert len(outputs2) == 0


# =============================================================================
# Tests: Tool Error Handling
# =============================================================================


async def test_tool_error_is_captured(turn_env):
    """Test that tool execution errors are captured in AgentToolReturned."""

    @loopback_tool
    async def failing_tool(ctx) -> str:
        """Tool that always fails."""
        raise ValueError("Something went wrong!")

    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="failing_tool", arguments="{}", is_complete=True)]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Sorry, there was an error."),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[failing_tool])

    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Do something", history=[UserTextSent(content="Do something")]),
    )

    # Find the AgentToolReturned
    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    # Error is passed in the result field
    assert "Something went wrong!" in str(tool_result.result)


# =============================================================================
# Tests: Conversation History in Context
# =============================================================================


async def test_conversation_history_passed_to_llm(turn_env):
    """Test that conversation history is included in messages to LLM."""
    # Pre-populate history on the event (including current user message)
    history = [
        UserTextSent(content="Hi"),
        AgentTextSent(content="Hello!"),
        UserTextSent(content="How are you?"),
    ]

    responses = [[StreamChunk(text="I'm doing well!"), StreamChunk(is_final=True)]]

    agent, mock_llm = create_agent_with_mock(responses)

    await collect_outputs(agent, turn_env, UserTextSent(content="How are you?", history=history))

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


async def test_streaming_tool_call_accumulation(turn_env):
    """Test that tool call arguments are accumulated across chunks."""

    @loopback_tool
    async def greet(ctx, name: Annotated[str, "Name"]) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    # Tool call arrives in chunks (simulating streaming)
    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="greet", arguments='{"na', is_complete=False)]
            ),
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="greet", arguments='me": "', is_complete=False)]
            ),
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="greet", arguments='Alice"}', is_complete=True)]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Hello, Alice!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[greet])

    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Greet Alice", history=[UserTextSent(content="Greet Alice")]),
    )

    # Tool should have been called with complete arguments
    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    assert tool_result.result == "Hello, Alice!"

    # Tool call output should have complete args
    tool_call = next((o for o in outputs if isinstance(o, AgentToolCalled)), None)
    assert tool_call is not None
    assert tool_call.tool_args == {"name": "Alice"}


# =============================================================================
# Tests: Loopback Tool Return Types
# =============================================================================


async def test_loopback_tool_returns_coroutine(turn_env):
    """Test that loopback tools can return a coroutine that gets awaited."""
    import asyncio

    async def fetch_data():
        await asyncio.sleep(0)  # Simulate async operation
        return "fetched data"

    @loopback_tool
    async def async_fetcher(ctx) -> str:
        """Fetch data asynchronously."""
        # Return a coroutine (simulating delegating to another async function)
        return await fetch_data()

    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="async_fetcher", arguments="{}", is_complete=True)]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Got the data!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[async_fetcher])

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Fetch", history=[UserTextSent(content="Fetch")])
    )

    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    assert tool_result.result == "fetched data"


async def test_loopback_tool_returns_async_iterable(turn_env):
    """Test that loopback tools can return an async iterable that gets collected."""

    async def stream_items():
        yield "item1"
        yield "item2"
        yield "item3"

    @loopback_tool
    def streaming_tool(ctx):
        """Return a stream of items."""
        return stream_items()

    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="streaming_tool", arguments="{}", is_complete=True)]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Got all items!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[streaming_tool])

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Stream", history=[UserTextSent(content="Stream")])
    )

    # New behavior: yields AgentToolReturned for each item in the stream
    tool_results = [o for o in outputs if isinstance(o, AgentToolReturned)]
    assert len(tool_results) == 3
    assert tool_results[0].result == "item1"
    assert tool_results[1].result == "item2"
    assert tool_results[2].result == "item3"


async def test_loopback_tool_returns_single_item_async_iterable(turn_env):
    """Test that async iterable with single item returns that item directly."""

    async def single_item():
        yield "only one"

    @loopback_tool
    def single_item_tool(ctx):
        """Return a single item stream."""
        return single_item()

    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="single_item_tool", arguments="{}", is_complete=True)]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Done!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[single_item_tool])

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Get", history=[UserTextSent(content="Get")])
    )

    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    # Single item is returned directly, not as a list
    assert tool_result.result == "only one"


async def test_loopback_tool_returns_bare_value(turn_env):
    """Test that loopback tools can return a plain value."""

    @loopback_tool
    def sync_tool(ctx) -> str:
        """Return a plain string."""
        return "plain value"

    responses = [
        [
            StreamChunk(
                tool_calls=[ToolCall(id="call_1", name="sync_tool", arguments="{}", is_complete=True)]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Got it!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[sync_tool])

    outputs = await collect_outputs(
        agent, turn_env, UserTextSent(content="Do", history=[UserTextSent(content="Do")])
    )

    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    assert tool_result.result == "plain value"


# =============================================================================
# Tests: Plain Functions Auto-Wrapped as Loopback Tools
# =============================================================================


async def test_plain_function_wrapped_as_loopback_tool():
    """Test that plain functions are wrapped as loopback tools when resolved."""
    from line.llm_agent.tools.utils import ToolType

    # Plain function without any decorator
    async def my_tool(ctx, query: Annotated[str, "Search query"]) -> str:
        """Search for something."""
        return f"Results for: {query}"

    agent = LlmAgent(model="gpt-4o", api_key="test-key")

    # Resolve tools - this is where wrapping happens
    resolved_tools, _ = agent._resolve_tools([my_tool])

    # Verify the tool was wrapped
    assert len(resolved_tools) == 1
    tool = resolved_tools[0]

    # Should be a FunctionTool
    assert isinstance(tool, FunctionTool)

    # Should be loopback type
    assert tool.tool_type == ToolType.LOOPBACK

    # Name and description should come from the function
    assert tool.name == "my_tool"
    assert tool.description == "Search for something."

    # Parameters should be extracted
    assert "query" in tool.parameters
    assert tool.parameters["query"].description == "Search query"


async def test_plain_function_works_as_loopback_tool(turn_env):
    """Test that plain functions work end-to-end as loopback tools."""

    # Plain function without decorator
    async def get_info(ctx, topic: Annotated[str, "Topic to look up"]) -> str:
        """Look up information about a topic."""
        return f"Info about {topic}: It's great!"

    # Response 1: LLM calls the tool
    # Response 2: LLM generates final text after seeing tool result
    responses = [
        [
            StreamChunk(text="Looking that up. "),
            StreamChunk(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_info",
                        arguments='{"topic": "Python"}',
                        is_complete=True,
                    )
                ]
            ),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Python is great!"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses, tools=[get_info])

    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(
            content="Tell me about Python",
            history=[UserTextSent(content="Tell me about Python")],
        ),
    )

    # Should work like a loopback tool
    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Looking that up. "

    assert isinstance(outputs[1], AgentToolCalled)
    assert outputs[1].tool_name == "get_info"

    assert isinstance(outputs[2], AgentToolReturned)
    assert outputs[2].result == "Info about Python: It's great!"

    assert isinstance(outputs[3], AgentSendText)
    assert outputs[3].text == "Python is great!"

    # LLM called twice (tool result fed back)
    assert mock_llm._call_count == 2


async def test_mixed_decorated_and_plain_functions(turn_env):
    """Test that decorated and plain functions can be mixed and both resolve to loopback tools."""
    from line.llm_agent.tools.utils import ToolType

    @loopback_tool
    async def decorated_tool(ctx) -> str:
        """A decorated tool."""
        return "decorated"

    async def plain_tool(ctx) -> str:
        """A plain tool."""
        return "plain"

    agent = LlmAgent(model="gpt-4o", api_key="test-key")

    # Resolve tools - plain functions get wrapped here
    resolved_tools, _ = agent._resolve_tools([decorated_tool, plain_tool])

    assert len(resolved_tools) == 2

    # Both should be loopback tools after resolution
    assert resolved_tools[0].tool_type == ToolType.LOOPBACK
    assert resolved_tools[1].tool_type == ToolType.LOOPBACK

    # Names should be correct
    assert resolved_tools[0].name == "decorated_tool"
    assert resolved_tools[1].name == "plain_tool"


# =============================================================================
# Tests: _build_messages - Pending Tool Results
# =============================================================================


class TestBuildMessagesPendingToolResults:
    """Tests for _build_messages handling of tool calls without matching results.

    When an AgentToolCalled event doesn't have a matching AgentToolReturned event,
    _build_messages should automatically add a pending result message with content="pending".
    """

    @staticmethod
    def _annotate(events: list, event_id: str) -> list[tuple[str, "OutputEvent"]]:
        """Annotate events with a triggering event_id."""
        return [(event_id, e) for e in events]

    async def test_tool_call_with_result_no_pending(self):
        """Tool call with matching result should NOT get a pending message."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="call_1", tool_name="get_weather", tool_args={"city": "NYC"}),
                AgentToolReturned(
                    tool_call_id="call_1", tool_name="get_weather", tool_args={"city": "NYC"}, result="sunny"
                ),
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user message, assistant with tool call, tool result
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert messages[2].role == "tool"
        assert messages[2].content == "sunny"
        assert messages[2].tool_call_id == "call_1"

    async def test_tool_call_without_result_gets_pending(self):
        """Tool call without matching result should get a pending message."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="call_1", tool_name="get_weather", tool_args={"city": "NYC"}),
                # No AgentToolReturned!
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user message, assistant with tool call, pending tool result
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert messages[2].role == "tool"
        assert messages[2].content == "pending"
        assert messages[2].tool_call_id == "call_1"
        assert messages[2].name == "get_weather"

    async def test_multiple_tool_calls_one_pending(self):
        """Multiple tool calls where one has result and one is pending."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="call_1", tool_name="get_weather", tool_args={}),
                AgentToolReturned(
                    tool_call_id="call_1", tool_name="get_weather", tool_args={}, result="sunny"
                ),
                AgentToolCalled(tool_call_id="call_2", tool_name="get_time", tool_args={}),
                # No AgentToolReturned for call_2!
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user, tool_call_1, result_1, tool_call_2, pending_2
        assert len(messages) == 5
        assert messages[0].role == "user"

        # First tool call and result
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls[0].id == "call_1"
        assert messages[2].role == "tool"
        assert messages[2].content == "sunny"
        assert messages[2].tool_call_id == "call_1"

        # Second tool call and pending result
        assert messages[3].role == "assistant"
        assert messages[3].tool_calls[0].id == "call_2"
        assert messages[4].role == "tool"
        assert messages[4].content == "pending"
        assert messages[4].tool_call_id == "call_2"
        assert messages[4].name == "get_time"

    async def test_multiple_tool_calls_all_pending(self):
        """Multiple tool calls all without results should all get pending."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="call_1", tool_name="tool_a", tool_args={}),
                AgentToolCalled(tool_call_id="call_2", tool_name="tool_b", tool_args={}),
                # No AgentToolReturned for either!
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user, tool_call_1, pending_1, tool_call_2, pending_2
        assert len(messages) == 5
        assert messages[0].role == "user"

        assert messages[1].role == "assistant"
        assert messages[1].tool_calls[0].id == "call_1"
        assert messages[2].role == "tool"
        assert messages[2].content == "pending"
        assert messages[2].tool_call_id == "call_1"

        assert messages[3].role == "assistant"
        assert messages[3].tool_calls[0].id == "call_2"
        assert messages[4].role == "tool"
        assert messages[4].content == "pending"
        assert messages[4].tool_call_id == "call_2"

    async def test_tool_call_result_out_of_order_still_matches(self):
        """Tool result appearing after other events still matches its call."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        # Use AgentToolCalled (unobservable) between call and result
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="call_1", tool_name="get_weather", tool_args={}),
                AgentToolCalled(tool_call_id="call_2", tool_name="other_tool", tool_args={}),
                AgentToolReturned(
                    tool_call_id="call_1", tool_name="get_weather", tool_args={}, result="sunny"
                ),
                # call_2 has no result - should be pending
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user, tool_call_1, tool_call_2, pending_2, result_1
        # (pending is inserted immediately after call_2 since it has no result)
        assert len(messages) == 5
        assert messages[0].role == "user"

        # First tool call
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls[0].id == "call_1"

        # Second tool call (no result yet)
        assert messages[2].role == "assistant"
        assert messages[2].tool_calls[0].id == "call_2"

        # Pending for call_2
        assert messages[3].role == "tool"
        assert messages[3].content == "pending"
        assert messages[3].tool_call_id == "call_2"

        # Result for call_1
        assert messages[4].role == "tool"
        assert messages[4].content == "sunny"
        assert messages[4].tool_call_id == "call_1"

    async def test_pending_result_preserves_tool_name(self):
        """Pending result message should have correct tool name."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(
                    tool_call_id="call_xyz",
                    tool_name="my_special_tool",
                    tool_args={"arg1": "value1"},
                ),
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        pending_msg = messages[-1]
        assert pending_msg.role == "tool"
        assert pending_msg.content == "pending"
        assert pending_msg.tool_call_id == "call_xyz"
        assert pending_msg.name == "my_special_tool"

    async def test_current_events_tool_call_without_result_gets_pending(self):
        """Current (not yet observed) tool calls without results get pending."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Question")
        input_history = [user0]
        # Event is annotated with "current" event_id
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="call_1", tool_name="async_tool", tool_args={}),
            ],
            event_id="current",
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user, tool_call, pending
        assert len(messages) == 3
        assert messages[2].role == "tool"
        assert messages[2].content == "pending"

    async def test_empty_history_no_pending(self):
        """Empty history should produce no messages."""
        agent, _ = create_agent_with_mock([])

        messages = await build_messages_with(agent, [], [], "current")

        assert len(messages) == 0


# =============================================================================
# Tests: add_history_entry
# =============================================================================


class TestAddHistoryEntry:
    """Tests for LlmAgent.add_history_entry."""

    @staticmethod
    def _annotate(events: list, event_id: str) -> list[tuple[str, "OutputEvent"]]:
        """Annotate events with a triggering event_id."""
        return [(event_id, e) for e in events]

    async def test_adds_custom_history_entry_as_mutation(self):
        """add_history_entry stores a mutation that produces a CustomHistoryEntry."""
        agent, _ = create_agent_with_mock([])

        agent.add_history_entry("injected context")

        assert len(agent.history._mutations) == 1
        result = list(agent.history)
        assert len(result) == 1
        assert isinstance(result[0], CustomHistoryEntry)
        assert result[0].content == "injected context"

    async def test_add_history_entry_before_first_process(self):
        """add_history_entry called before process() appears before other events in messages."""
        agent, _ = create_agent_with_mock([])

        # Called before any process() — entry should appear at beginning
        agent.add_history_entry("pre-process context")

        user0 = UserTextSent(content="Hello")
        input_history = [user0]

        messages = await build_messages_with(
            agent, input_history, agent.history._local_history, user0.event_id
        )

        # Entry should appear before the user message
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "pre-process context"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"

    async def test_add_history_entry_before_first_process_no_input_history(self):
        """add_history_entry before process() works even with empty input history."""
        agent, _ = create_agent_with_mock([])

        agent.add_history_entry("early context")

        messages = await build_messages_with(agent, [], agent.history._local_history, "current")

        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content == "early context"

    async def test_multiple_add_history_entry_calls(self):
        """Multiple add_history_entry calls each produce a mutation."""
        agent, _ = create_agent_with_mock([])

        agent.add_history_entry("first")
        agent.add_history_entry("second")

        assert len(agent.history._mutations) == 2
        result = list(agent.history)
        # Both should appear at beginning, in order (first call first)
        # Each appends after the last event (FIFO)
        assert len(result) == 2
        assert result[0].content == "first"
        assert result[1].content == "second"

    async def test_custom_entry_defaults_to_system_message(self):
        """CustomHistoryEntry in local_history defaults to a system message."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Hello")
        input_history = [user0]
        local_history = self._annotate(
            [CustomHistoryEntry(content="extra context")],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # Should have: user message from input, then custom entry as system message
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "system"
        assert messages[1].content == "extra context"

    async def test_custom_entry_with_user_role(self):
        """CustomHistoryEntry with role='user' produces a user message."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Hello")
        input_history = [user0]
        local_history = self._annotate(
            [CustomHistoryEntry(content="user context", role="user")],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "user"
        assert messages[1].content == "user context"

    async def test_custom_entry_in_current_local(self):
        """CustomHistoryEntry in current_local events uses its role."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Hello")
        input_history = [user0]
        local_history = self._annotate(
            [CustomHistoryEntry(content="current context")],
            event_id="current",
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "system"
        assert messages[1].content == "current context"

    async def test_custom_entry_interleaved_with_tool_calls(self):
        """CustomHistoryEntry preserves position relative to tool calls."""
        agent, _ = create_agent_with_mock([])

        user0 = UserTextSent(content="Start")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="c1", tool_name="lookup", tool_args={}),
                AgentToolReturned(tool_call_id="c1", tool_name="lookup", tool_args={}, result="data"),
                CustomHistoryEntry(content="injected after tool"),
            ],
            event_id=user0.event_id,
        )

        messages = await build_messages_with(agent, input_history, local_history, "current")

        # user, assistant(tool_call), tool(result), system(custom)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[0].content == "Start"
        assert messages[1].role == "assistant"
        assert messages[2].role == "tool"
        assert messages[2].content == "data"
        assert messages[3].role == "system"
        assert messages[3].content == "injected after tool"

    async def test_add_history_entry_end_to_end(self, turn_env):
        """add_history_entry during a tool call is visible to the LLM on loopback."""
        # The tool calls add_history_entry on the agent, then the LLM sees it
        agent, mock_llm = create_agent_with_mock([])

        @loopback_tool
        def inject_context(ctx, info: str):
            """Inject context."""
            agent.add_history_entry(f"Context: {info}")
            return "done"

        agent._tools = [inject_context]
        agent._tool_map = {inject_context.name: inject_context}

        # First LLM call: calls the tool
        # Second LLM call: responds with final text after seeing the injected context
        mock_llm._responses = [
            [
                StreamChunk(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="inject_context",
                            arguments='{"info": "important data"}',
                            is_complete=True,
                        )
                    ]
                ),
                StreamChunk(is_final=True),
            ],
            [
                StreamChunk(text="Got it!"),
                StreamChunk(is_final=True),
            ],
        ]

        outputs = await collect_outputs(
            agent,
            turn_env,
            UserTextSent(content="Do something", history=[UserTextSent(content="Do something")]),
        )

        # Should have: AgentToolCalled, AgentToolReturned, AgentSendText
        tool_called = [o for o in outputs if isinstance(o, AgentToolCalled)]
        tool_returned = [o for o in outputs if isinstance(o, AgentToolReturned)]
        text_outputs = [o for o in outputs if isinstance(o, AgentSendText)]

        assert len(tool_called) == 1
        assert len(tool_returned) == 1
        assert len(text_outputs) == 1
        assert text_outputs[0].text == "Got it!"

        # Verify the injected context was visible to the LLM on the second call
        second_call_messages = mock_llm._recorded_messages[1]
        system_messages = [m for m in second_call_messages if m.role == "system"]
        system_contents = [m.content for m in system_messages]
        assert "Context: important data" in system_contents


# =============================================================================
# Tests: Tool and Config Replacement in process()
# =============================================================================


async def test_process_replaces_tools_with_same_name(turn_env):
    """Test that tools passed to process() replace tools with the same name in self._tools."""

    # Define two versions of the same tool
    @loopback_tool
    async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
        """Get weather - original version."""
        return f"Original: 72°F in {city}"

    @loopback_tool
    async def get_weather_v2(ctx, city: Annotated[str, "City name"]) -> str:
        """Get weather - override version."""
        return f"Override: 85°F in {city}"

    # Rename the second tool to match the first one's name
    get_weather_v2.name = "get_weather"

    # Create agent with the original tool
    agent, mock_llm = create_agent_with_mock(
        [
            [
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
                StreamChunk(text="Got it!"),
                StreamChunk(is_final=True),
            ],
        ],
        tools=[get_weather],
    )

    # Call process with the override tool
    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(
            content="What's the weather?",
            history=[UserTextSent(content="What's the weather?")],
        ),
        tools=[get_weather_v2],
    )

    # Find the tool result
    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    # Should use the override version, not the original
    assert tool_result.result == "Override: 85°F in NYC"
    assert "Original" not in tool_result.result


async def test_process_replaces_config(turn_env):
    """Test that config passed to process() overrides the agent's config."""

    # Create agent with base config
    base_config = LlmConfig(temperature=0.5, max_tokens=100)
    agent, mock_llm = create_agent_with_mock(
        [
            [
                StreamChunk(text="Response"),
                StreamChunk(is_final=True),
            ],
        ],
        config=base_config,
    )

    # Call process with override config
    override_config = LlmConfig(temperature=0.9, max_tokens=500)
    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Hi", history=[UserTextSent(content="Hi")]),
        config=override_config,
    )

    # Verify output was generated
    assert len(outputs) == 1
    assert isinstance(outputs[0], AgentSendText)

    # The agent should still have its original config
    assert agent._config.temperature == 0.5
    assert agent._config.max_tokens == 100


# =============================================================================
# Tests: context and history params in process()
# =============================================================================


async def test_process_with_context_string(turn_env):
    """Test that context string appears as system message at end of history."""
    responses = [
        [
            StreamChunk(text="Got it!"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Hi", history=[UserTextSent(content="Hi")]),
        context="You are a helpful assistant.",
    )

    # Check messages sent to LLM
    messages = mock_llm._recorded_messages[0]

    # Should have: user message, then system context at end
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hi"
    assert messages[1].role == "system"
    assert messages[1].content == "You are a helpful assistant."


async def test_process_with_context_list(turn_env):
    """Test that context list events appear at end of history."""
    responses = [
        [
            StreamChunk(text="Got it!"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    context_events = [
        UserTextSent(content="Extra user context"),
        AgentTextSent(content="Extra agent context"),
    ]

    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Hi", history=[UserTextSent(content="Hi")]),
        context=context_events,
    )

    messages = mock_llm._recorded_messages[0]

    # Should have: user message from history, then context events at end
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content == "Hi"
    assert messages[1].role == "user"
    assert messages[1].content == "Extra user context"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Extra agent context"


async def test_process_with_history_override(turn_env):
    """Test that history override replaces managed history for LLM messages."""
    responses = [
        [
            StreamChunk(text="Response"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    override_history = [
        UserTextSent(content="Override message 1"),
        AgentTextSent(content="Override response 1"),
        UserTextSent(content="Override message 2"),
    ]

    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Real message", history=[UserTextSent(content="Real message")]),
        history=override_history,
    )

    messages = mock_llm._recorded_messages[0]

    # Should use override history, not the managed history
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content == "Override message 1"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Override response 1"
    assert messages[2].role == "user"
    assert messages[2].content == "Override message 2"


async def test_process_with_history_and_context(turn_env):
    """Test that context is appended to the history override."""
    responses = [
        [
            StreamChunk(text="Response"),
            StreamChunk(is_final=True),
        ]
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    override_history = [
        UserTextSent(content="Override message"),
    ]

    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Real message", history=[UserTextSent(content="Real message")]),
        history=override_history,
        context="Extra system context",
    )

    messages = mock_llm._recorded_messages[0]

    # Should use override history with context appended
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Override message"
    assert messages[1].role == "system"
    assert messages[1].content == "Extra system context"


async def test_context_does_not_persist(turn_env):
    """Test that context from one process() call does not leak into the next."""
    responses = [
        [
            StreamChunk(text="First response"),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Second response"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    # First call with context
    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Hi", history=[UserTextSent(content="Hi")]),
        context="Temporary context",
    )

    # Second call without context
    user1 = UserTextSent(content="Hi")
    agent1 = AgentTextSent(content="First response")
    user2 = UserTextSent(content="Follow up")
    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Follow up", history=[user1, agent1, user2]),
    )

    # Second call should NOT have the context from the first call
    second_messages = mock_llm._recorded_messages[1]
    system_messages = [m for m in second_messages if m.role == "system"]
    assert len(system_messages) == 0


async def test_history_override_does_not_affect_managed(turn_env):
    """Test that history override does not mutate self.history."""
    responses = [
        [
            StreamChunk(text="First response"),
            StreamChunk(is_final=True),
        ],
        [
            StreamChunk(text="Second response"),
            StreamChunk(is_final=True),
        ],
    ]

    agent, mock_llm = create_agent_with_mock(responses)

    # First call with history override
    override_history = [
        UserTextSent(content="Override only"),
    ]
    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Real message", history=[UserTextSent(content="Real message")]),
        history=override_history,
    )

    # First call LLM should see override history
    first_messages = mock_llm._recorded_messages[0]
    assert len(first_messages) == 1
    assert first_messages[0].content == "Override only"

    # Second call without override - should see managed history (with first call's events)
    user1 = UserTextSent(content="Real message")
    agent1 = AgentTextSent(content="First response")
    user2 = UserTextSent(content="Second message")
    await collect_outputs(
        agent,
        turn_env,
        UserTextSent(content="Second message", history=[user1, agent1, user2]),
    )

    second_messages = mock_llm._recorded_messages[1]
    # Should see the managed history, not the override
    assert any(m.content == "Real message" for m in second_messages)
    assert not any(m.content == "Override only" for m in second_messages)
