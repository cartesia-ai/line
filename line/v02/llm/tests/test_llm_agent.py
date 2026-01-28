"""
Tests for LlmAgent tool call handling.

These tests mock the LLM provider to verify the agent loop correctly handles
loopback, passthrough, and handoff tools.

uv run pytest line/v02/llm/tests/test_llm_agent.py -v
"""

from typing import Annotated, List, Optional

import pytest

from line.v02.llm.agent import (
    AgentEndCall,
    AgentHandedOff,
    AgentSendText,
    AgentToolCalled,
    AgentToolReturned,
    CallStarted,
    OutputEvent,
    SpecificAgentTextSent,
    SpecificCallEnded,
    SpecificUserTextSent,
    TurnEnv,
    UserTextSent,
)
from line.v02.llm.config import LlmConfig
from line.v02.llm.llm_agent import LlmAgent, _build_full_history
from line.v02.llm.provider import Message, StreamChunk, ToolCall
from line.v02.llm.tool_types import handoff_tool, loopback_tool, passthrough_tool
from line.v02.llm.tool_utils import FunctionTool

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

    def chat(
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


async def collect_outputs(agent: LlmAgent, env: TurnEnv, event) -> List[OutputEvent]:
    """Collect all outputs from agent.process()."""
    outputs = []
    async for output in agent.process(env, event):
        outputs.append(output)
    return outputs


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
        agent, turn_env, UserTextSent(content="Hi", history=[SpecificUserTextSent(content="Hi")])
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
            history=[SpecificUserTextSent(content="What's the weather in NYC?")],
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
            content="Weather in NYC and LA?", history=[SpecificUserTextSent(content="Weather in NYC and LA?")]
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
        agent, turn_env, UserTextSent(content="Bye", history=[SpecificUserTextSent(content="Bye")])
    )

    # Expected outputs:
    # 1. AgentSendText "Goodbye! "
    # 2. AgentToolCalled
    # 3. AgentSendText "Have a great day!" (from passthrough)
    # 4. AgentEndCall (from passthrough)
    # 5. AgentToolReturned

    assert len(outputs) == 5

    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Goodbye! "

    assert isinstance(outputs[1], AgentToolCalled)
    assert outputs[1].tool_name == "end_call"

    assert isinstance(outputs[2], AgentSendText)
    assert outputs[2].text == "Have a great day!"

    assert isinstance(outputs[3], AgentEndCall)

    assert isinstance(outputs[4], AgentToolReturned)

    # LLM should only be called ONCE - passthrough doesn't loop back
    assert mock_llm._call_count == 1


# =============================================================================
# Tests: Handoff Tool
# =============================================================================


async def test_handoff_tool_emits_handoff_event(turn_env):
    """Test that handoff tool emits AgentHandedOff event and sets handoff target."""

    class BillingAgent:
        """A mock billing agent."""

        async def process(self, env, event):
            yield AgentSendText(text="Billing agent here!")

    billing_agent = BillingAgent()

    @handoff_tool
    async def transfer_to_billing(ctx, reason: Annotated[str, "Reason"], event):
        """Transfer to billing department."""

        if isinstance(event, AgentHandedOff):
            # Handoff tools are async generators that yield events and the handoff target
            yield AgentSendText(text="Transferring you now...")

        async for output in billing_agent.process(ctx.turn_env, event):
            yield output

    responses = [
        [
            StreamChunk(text="Let me transfer you. "),
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

    outputs = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(
            content="I have a billing question",
            history=[SpecificUserTextSent(content="I have a billing question")],
        ),
    )

    # Expected: AgentSendText (LLM), AgentToolCalled, AgentSendText (from tool),
    # AgentSendText (from billing_agent.process), AgentToolReturned
    assert len(outputs) == 5

    assert isinstance(outputs[0], AgentSendText)
    assert outputs[0].text == "Let me transfer you. "

    assert isinstance(outputs[1], AgentToolCalled)
    assert outputs[1].tool_name == "transfer_to_billing"

    assert isinstance(outputs[2], AgentToolReturned)

    assert isinstance(outputs[3], AgentSendText)
    assert outputs[3].text == "Transferring you now..."

    assert isinstance(outputs[4], AgentSendText)
    assert outputs[4].text == "Billing agent here!"

    # LLM called only once - handoff doesn't loop back
    assert mock_llm._call_count == 1

    # Verify handoff target is set (behavior tested in test_handoff_delegates_subsequent_calls)
    assert agent.handoff_target is not None
    assert callable(agent.handoff_target)


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
        UserTextSent(content="Transfer me", history=[SpecificUserTextSent(content="Transfer me")]),
    )

    # Second call should go to billing agent, not the LLM
    outputs2 = await collect_outputs(
        agent,
        turn_env,
        UserTextSent(
            content="Follow up question", history=[SpecificUserTextSent(content="Follow up question")]
        ),
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
        agent, turn_env, UserTextSent(content="Start", history=[SpecificUserTextSent(content="Start")])
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
        UserTextSent(content="Do something", history=[SpecificUserTextSent(content="Do something")]),
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
        SpecificUserTextSent(content="Hi"),
        SpecificAgentTextSent(content="Hello!"),
        SpecificUserTextSent(content="How are you?"),
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
        UserTextSent(content="Greet Alice", history=[SpecificUserTextSent(content="Greet Alice")]),
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
        agent, turn_env, UserTextSent(content="Fetch", history=[SpecificUserTextSent(content="Fetch")])
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
        agent, turn_env, UserTextSent(content="Stream", history=[SpecificUserTextSent(content="Stream")])
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
        agent, turn_env, UserTextSent(content="Get", history=[SpecificUserTextSent(content="Get")])
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
        agent, turn_env, UserTextSent(content="Do", history=[SpecificUserTextSent(content="Do")])
    )

    tool_result = next((o for o in outputs if isinstance(o, AgentToolReturned)), None)
    assert tool_result is not None
    assert tool_result.result == "plain value"


# =============================================================================
# Tests: Plain Functions Auto-Wrapped as Loopback Tools
# =============================================================================


async def test_plain_function_wrapped_as_loopback_tool():
    """Test that plain functions passed to LlmAgent are wrapped as loopback tools."""
    from line.v02.llm.tool_utils import ToolType

    # Plain function without any decorator
    async def my_tool(ctx, query: Annotated[str, "Search query"]) -> str:
        """Search for something."""
        return f"Results for: {query}"

    agent = LlmAgent(model="test-model", tools=[my_tool])

    # Verify the tool was wrapped
    assert len(agent.tools) == 1
    tool = agent.tools[0]

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
            history=[SpecificUserTextSent(content="Tell me about Python")],
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
    """Test that decorated and plain functions can be mixed."""
    from line.v02.llm.tool_utils import ToolType

    @loopback_tool
    async def decorated_tool(ctx) -> str:
        """A decorated tool."""
        return "decorated"

    async def plain_tool(ctx) -> str:
        """A plain tool."""
        return "plain"

    agent = LlmAgent(model="test-model", tools=[decorated_tool, plain_tool])

    assert len(agent.tools) == 2

    # Both should be loopback tools
    assert agent.tools[0].tool_type == ToolType.LOOPBACK
    assert agent.tools[1].tool_type == ToolType.LOOPBACK

    # Names should be correct
    assert agent.tools[0].name == "decorated_tool"
    assert agent.tools[1].name == "plain_tool"


# =============================================================================
# Tests: _build_full_history
# =============================================================================


class TestBuildFullHistory:
    """Tests for the _build_full_history function.

    The function merges input_history (canonical) with local_history using these rules:
    1. Input history is the source of truth for all events
    2. Local events are grouped by the event_id of the input that triggered them
    3. For input events with responsive local events, interpolation/matching is applied
    4. For input events without responsive local events, they pass through as-is
    5. Current local events (not yet observed) are appended at the end

    Observable output events: AgentSendText, AgentSendDtmf, AgentEndCall
    Observable input events: SpecificAgentTextSent, SpecificAgentDtmfSent, SpecificCallEnded

    local_history format: List[tuple[str, OutputEvent]] where str is the triggering event_id
    """

    @staticmethod
    def _annotate(events: list, event_id: str) -> list[tuple[str, "OutputEvent"]]:
        """Annotate events with a triggering event_id."""
        return [(event_id, e) for e in events]

    async def test_both_histories_empty(self):
        """When both histories are empty, return empty list."""
        result = _build_full_history([], [], current_event_id="current")
        assert result == []

    async def test_only_input_history_with_user_message(self):
        """When only input_history exists with non-observable event, include it."""
        input_history = [SpecificUserTextSent(content="Hello")]
        result = _build_full_history(input_history, [], current_event_id="current")

        assert len(result) == 1
        assert isinstance(result[0], SpecificUserTextSent)
        assert result[0].content == "Hello"

    async def test_only_input_history_with_observable_event(self):
        """When only input_history exists with observable event, include it (pass through)."""
        input_history = [SpecificAgentTextSent(content="Hi there")]
        result = _build_full_history(input_history, [], current_event_id="current")

        # Observable input with no responsive local passes through
        assert len(result) == 1
        assert isinstance(result[0], SpecificAgentTextSent)
        assert result[0].content == "Hi there"

    async def test_only_local_history_with_unobservable_event_current(self):
        """When only local_history exists with unobservable current event, include it."""
        # Current events (event_id == current_event_id) are appended
        local_history = self._annotate(
            [AgentToolCalled(tool_call_id="1", tool_name="test", tool_args={})],
            event_id="current",
        )
        result = _build_full_history([], local_history, current_event_id="current")

        assert len(result) == 1
        assert isinstance(result[0], AgentToolCalled)
        assert result[0].tool_name == "test"

    async def test_only_local_history_with_observable_event_prior_excluded(self):
        """Prior local observable events without matching input are excluded."""
        # Prior events (event_id != current_event_id) with no input to match
        local_history = self._annotate(
            [AgentSendText(text="Unmatched response")],
            event_id="prior",
        )
        result = _build_full_history([], local_history, current_event_id="current")

        # Observable prior local events without matching input events are excluded
        assert len(result) == 0

    async def test_matching_observable_events_uses_input_canonical(self):
        """When observable events match, use the input_history version (canonical)."""
        # Create input event and use its event_id for local
        input_evt = SpecificAgentTextSent(content="Hello world")
        input_history = [input_evt]
        local_history = self._annotate(
            [AgentSendText(text="Hello world")],
            event_id=input_evt.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 1
        # Should be the input_history version (SpecificAgentTextSent), not local (AgentSendText)
        assert isinstance(result[0], SpecificAgentTextSent)
        assert result[0].content == "Hello world"

    async def test_unobservable_local_event_interpolated_before_observable(self):
        """Unobservable local events appear before their following observable event."""
        # Input: [User0, Agent1] - User0 triggers local events
        user0 = SpecificUserTextSent(content="Question")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Response"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="lookup", tool_args={"q": "test"}),
                AgentToolReturned(
                    tool_call_id="1", tool_name="lookup", tool_args={"q": "test"}, result="data"
                ),
                AgentSendText(text="Response"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Expected order: User, ToolCalled, ToolReturned, AgentTextSent
        assert len(result) == 4
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], SpecificAgentTextSent)

    async def test_non_observable_input_event_included(self):
        """Non-observable input events (like SpecificUserTextSent) are always included."""
        user0 = SpecificUserTextSent(content="User question")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Agent answer"),
        ]
        local_history = self._annotate(
            [AgentSendText(text="Agent answer")],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 2
        assert isinstance(result[0], SpecificUserTextSent)
        assert result[0].content == "User question"
        assert isinstance(result[1], SpecificAgentTextSent)
        assert result[1].content == "Agent answer"

    async def test_unmatched_observable_local_event_excluded(self):
        """Observable local events without matching input events are excluded."""
        user0 = SpecificUserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentSendText(text="First response"),  # Observable, no match in input
                AgentSendText(text="Second response"),  # Observable, no match in input
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Only the user message should be included
        assert len(result) == 1
        assert isinstance(result[0], SpecificUserTextSent)

    async def test_complex_conversation_with_tools(self):
        """Test a realistic conversation with user messages, tool calls, and agent responses."""
        user0 = SpecificUserTextSent(content="What's the weather?")
        input_history = [
            user0,
            SpecificAgentTextSent(content="The weather is sunny."),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="get_weather", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="get_weather", tool_args={}, result="sunny"),
                AgentSendText(text="The weather is sunny."),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Expected: UserText, ToolCalled, ToolReturned, AgentTextSent (canonical)
        assert len(result) == 4
        assert isinstance(result[0], SpecificUserTextSent)
        assert result[0].content == "What's the weather?"
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], SpecificAgentTextSent)
        assert result[3].content == "The weather is sunny."

    async def test_multiple_tool_calls_interleaved(self):
        """Test multiple tool calls with responses interleaved."""
        user0 = SpecificUserTextSent(content="Get weather and time")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Weather is sunny, time is 3pm."),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="get_weather", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="get_weather", tool_args={}, result="sunny"),
                AgentToolCalled(tool_call_id="2", tool_name="get_time", tool_args={}),
                AgentToolReturned(tool_call_id="2", tool_name="get_time", tool_args={}, result="3pm"),
                AgentSendText(text="Weather is sunny, time is 3pm."),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # All unobservable events interpolated, then the matching observable
        assert len(result) == 6
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert result[1].tool_name == "get_weather"
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentToolCalled)
        assert result[3].tool_name == "get_time"
        assert isinstance(result[4], AgentToolReturned)
        assert isinstance(result[5], SpecificAgentTextSent)

    async def test_multiple_matching_observable_events(self):
        """Test conversation with multiple turns, each with matching observables."""
        user0 = SpecificUserTextSent(content="Hi")
        user2 = SpecificUserTextSent(content="How are you?")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Hello!"),
            user2,
            SpecificAgentTextSent(content="I'm good!"),
        ]
        # First turn (user0) triggers Hello, second turn (user2) triggers I'm good
        local_history = self._annotate(
            [AgentSendText(text="Hello!")],
            event_id=user0.event_id,
        ) + self._annotate(
            [AgentSendText(text="I'm good!")],
            event_id=user2.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 4
        assert isinstance(result[0], SpecificUserTextSent)
        assert result[0].content == "Hi"
        assert isinstance(result[1], SpecificAgentTextSent)
        assert result[1].content == "Hello!"
        assert isinstance(result[2], SpecificUserTextSent)
        assert result[2].content == "How are you?"
        assert isinstance(result[3], SpecificAgentTextSent)
        assert result[3].content == "I'm good!"

    async def test_unobservable_events_between_observables(self):
        """Test unobservable events appear in correct position between observables."""
        user0 = SpecificUserTextSent(content="Start")
        user2 = SpecificUserTextSent(content="User interjects")
        input_history = [
            user0,
            SpecificAgentTextSent(content="First"),
            user2,
            SpecificAgentTextSent(content="Second"),
        ]
        # user0 triggers First + tools, user2 triggers Second
        local_history = self._annotate(
            [
                AgentSendText(text="First"),
                AgentToolCalled(tool_call_id="1", tool_name="middle_tool", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="middle_tool", tool_args={}, result="done"),
            ],
            event_id=user0.event_id,
        ) + self._annotate(
            [AgentSendText(text="Second")],
            event_id=user2.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User0, First, tools (drained after First), User2, Second
        assert len(result) == 6
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], SpecificAgentTextSent)
        assert result[1].content == "First"
        assert isinstance(result[2], AgentToolCalled)
        assert isinstance(result[3], AgentToolReturned)
        assert isinstance(result[4], SpecificUserTextSent)
        assert isinstance(result[5], SpecificAgentTextSent)
        assert result[5].content == "Second"

    async def test_input_observable_without_local_match_passes_through(self):
        """Input observable events pass through when no local responds to them.

        This happens when input_history has events from previous agents or turns
        that the current agent instance didn't generate.
        """
        input_history = [
            SpecificAgentTextSent(content="Previous agent said this"),
            SpecificUserTextSent(content="User reply"),
        ]
        local_history = []  # This agent hasn't generated anything yet

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Both events pass through (no local to match/interpolate)
        assert len(result) == 2
        assert isinstance(result[0], SpecificAgentTextSent)
        assert isinstance(result[1], SpecificUserTextSent)

    async def test_partial_match_only_matching_local_observable_kept(self):
        """When local has more observables than input, only matching ones are kept."""
        user0 = SpecificUserTextSent(content="Question")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Answer"),
        ]
        local_history = self._annotate(
            [
                AgentSendText(text="Partial..."),  # This was streamed but not in input yet
                AgentSendText(text="Answer"),  # This matches
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # "Partial..." should be excluded (no match), "Answer" should match
        assert len(result) == 2
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], SpecificAgentTextSent)
        assert result[1].content == "Answer"

    async def test_empty_local_with_full_input_history(self):
        """When local is empty, full input history passes through."""
        input_history = [
            SpecificUserTextSent(content="First"),
            SpecificAgentTextSent(content="Response 1"),
            SpecificUserTextSent(content="Second"),
            SpecificAgentTextSent(content="Response 2"),
        ]
        local_history = []

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # All input events pass through (no local to match/interpolate)
        assert len(result) == 4
        assert all(isinstance(r, (SpecificUserTextSent, SpecificAgentTextSent)) for r in result)

    async def test_mismatched_text_content_not_matched(self):
        """Observable events with different content don't match."""
        agent0 = SpecificAgentTextSent(content="Hello")
        input_history = [agent0]
        local_history = self._annotate(
            [AgentSendText(text="Goodbye")],  # Different content
            event_id=agent0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # local event is excluded (no match), input event is matched with slice
        # but no match found, so slice produces nothing for this observable
        # Since local responds to agent0, we process slice with local
        # but they don't match, so local is skipped and input passes through
        assert len(result) == 1
        assert isinstance(result[0], SpecificAgentTextSent)
        assert result[0].content == "Hello"

    async def test_all_unobservable_local_history(self):
        """When local history contains only unobservable events, all are included."""
        user0 = SpecificUserTextSent(content="Do something")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
                AgentToolCalled(tool_call_id="2", tool_name="tool2", tool_args={}),
                AgentToolReturned(tool_call_id="2", tool_name="tool2", tool_args={}, result="r2"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentToolCalled)
        assert isinstance(result[4], AgentToolReturned)

    async def test_agent_end_call_observable_matching(self):
        """Test that AgentEndCall matches with SpecificCallEnded."""
        user0 = SpecificUserTextSent(content="Goodbye")
        input_history = [
            user0,
            SpecificCallEnded(),
        ]
        local_history = self._annotate(
            [
                AgentSendText(text="Bye!"),  # Unmatched, will be excluded
                AgentEndCall(),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User message + CallEnded (canonical), AgentSendText excluded (no match)
        assert len(result) == 2
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], SpecificCallEnded)

    # =========================================================================
    # Tests: Prefix Matching
    # =========================================================================

    async def test_prefix_match_input_is_prefix_of_local(self):
        """When input text is a prefix of local text, match and carry forward suffix."""
        user0 = SpecificUserTextSent(content="Question")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Hello"),  # Prefix of "Hello world!"
        ]
        local_history = self._annotate(
            [AgentSendText(text="Hello world!")],  # Has suffix " world!"
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Should include user message and the canonical input (prefix match)
        # The suffix " world!" should be excluded since there's no matching input for it
        assert len(result) == 2
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], SpecificAgentTextSent)
        assert result[1].content == "Hello"

    async def test_prefix_match_suffix_matches_next_input(self):
        """Suffix from prefix match can match the next input event."""
        # user0 triggers local, slice includes [User0, Agent1, User2, Agent3]
        user0 = SpecificUserTextSent(content="Start")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Hello"),  # Prefix of "Hello world!"
            SpecificUserTextSent(content="..."),
            SpecificAgentTextSent(content=" world!"),  # Matches carried suffix
        ]
        local_history = self._annotate(
            [AgentSendText(text="Hello world!")],  # Will be split via prefix match
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User0, Hello (prefix match), user message, then " world!" matches suffix
        assert len(result) == 4
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], SpecificAgentTextSent)
        assert result[1].content == "Hello"
        assert isinstance(result[2], SpecificUserTextSent)
        assert isinstance(result[3], SpecificAgentTextSent)
        assert result[3].content == " world!"

    async def test_prefix_match_with_tool_calls_between(self):
        """Prefix matching works when tool calls precede the text."""
        user0 = SpecificUserTextSent(content="Get weather")
        input_history = [
            user0,
            SpecificAgentTextSent(content="The weather "),
            SpecificAgentTextSent(content="is sunny today!"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="weather", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="weather", tool_args={}, result="sunny"),
                AgentSendText(text="The weather is sunny today!"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User, ToolCalled, ToolReturned, then two text events (prefix match then suffix)
        assert len(result) == 5
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], SpecificAgentTextSent)
        assert result[3].content == "The weather "
        assert isinstance(result[4], SpecificAgentTextSent)
        assert result[4].content == "is sunny today!"

    async def test_prefix_match_single_input_split(self):
        """Single input text that is prefix of local text - suffix is dropped."""
        agent0 = SpecificAgentTextSent(content="Hello")  # Prefix of "Hello world!"
        input_history = [agent0]
        local_history = self._annotate(
            [AgentSendText(text="Hello world!")],
            event_id=agent0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Only the input (prefix) is included, suffix " world!" is dropped (no matching input)
        assert len(result) == 1
        assert isinstance(result[0], SpecificAgentTextSent)
        assert result[0].content == "Hello"

    async def test_prefix_match_multiple_splits_with_separators(self):
        """Local text split across multiple non-contiguous input events."""
        user0 = SpecificUserTextSent(content="Start")
        input_history = [
            user0,
            SpecificAgentTextSent(content="A"),
            SpecificUserTextSent(content="u1"),
            SpecificAgentTextSent(content="B"),
            SpecificUserTextSent(content="u2"),
            SpecificAgentTextSent(content="C"),
        ]
        local_history = self._annotate(
            [AgentSendText(text="ABC")],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User0, A (prefix), u1, B (prefix of "BC"), u2, C (exact match)
        assert len(result) == 6
        assert isinstance(result[0], SpecificUserTextSent)
        assert result[1].content == "A"
        assert isinstance(result[2], SpecificUserTextSent)
        assert result[3].content == "B"
        assert isinstance(result[4], SpecificUserTextSent)
        assert result[5].content == "C"

    async def test_preprocessing_local_longer_than_input(self):
        """Local text is longer than input after preprocessing - prefix match."""
        agent0 = SpecificAgentTextSent(content="Hello")  # Single event
        input_history = [agent0]
        local_history = self._annotate(
            [
                # These get concatenated to "Hello world!"
                AgentSendText(text="Hello"),
                AgentSendText(text=" world!"),
            ],
            event_id=agent0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Input "Hello" is prefix of local "Hello world!"
        # Suffix " world!" has no matching input, so dropped
        assert len(result) == 1
        assert isinstance(result[0], SpecificAgentTextSent)
        assert result[0].content == "Hello"

    async def test_pre_tool_call_result_grouped_with_agent_text(self):
        """Tool call and result are grouped together in the final history."""
        user0 = SpecificUserTextSent(content="Question1")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Response1"),
            SpecificUserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
                AgentSendText(text="Response1"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], SpecificAgentTextSent)
        assert isinstance(result[4], SpecificUserTextSent)

    # ==== TOOL CALL ORDERING ====

    async def test_circum_tool_call_result_grouped_with_agent_text(self):
        """Tool call appears before text, tool result after due to local ordering."""
        user0 = SpecificUserTextSent(content="Question1")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Response1"),
            SpecificUserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentSendText(text="Response1 and more"),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], SpecificAgentTextSent)
        assert isinstance(result[3], AgentToolReturned)
        assert isinstance(result[4], SpecificUserTextSent)

    async def test_pre_tool_call_result_grouped_with_trimmed_agent_text(self):
        """Tool call and result are grouped together in the final history."""
        user0 = SpecificUserTextSent(content="Question1")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Response1"),
            SpecificUserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
                AgentSendText(text="Response1 and more"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], SpecificAgentTextSent)
        assert isinstance(result[4], SpecificUserTextSent)

    async def test_circum_tool_call_result_grouped_with_trimmed_agent_text(self):
        """Tool call appears before text, tool result after due to local ordering."""
        user0 = SpecificUserTextSent(content="Question1")
        input_history = [
            user0,
            SpecificAgentTextSent(content="Response1"),
            SpecificUserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentSendText(text="Response1"),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Tool call interpolated around matched text
        assert len(result) == 5
        assert isinstance(result[0], SpecificUserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], SpecificAgentTextSent)
        assert isinstance(result[3], AgentToolReturned)
        assert isinstance(result[4], SpecificUserTextSent)
