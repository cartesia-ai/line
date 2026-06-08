"""Unit + integration tests for the built-in voicemail tool and its removal.

These tests mock the LLM provider — no network. They cover:
  - unit: VoicemailTool resolves through tool normalization; the _is_voicemail_tool helper.
  - integration: the tool fires (speaks the message + ends the call with
    reason="voicemail_detected"), and the agent drops the tool from its options
    after `voicemail_tool_active_turns` user turns so it can't hang up mid-call.

    uv run pytest tests/test_llm_agent_voicemail_tool.py -v
"""

from typing import List, Optional

from line.agent import AgentEnv, TurnEnv
from line.events import AgentEndCall, AgentSendText, LogMetric, UserTextSent, UserTurnEnded
from line.llm_agent.llm_agent import LlmAgent, _is_voicemail_tool
from line.llm_agent.provider import Message, StreamChunk, ToolCall, parse_model_id
from line.llm_agent.tools.system import VoicemailTool, end_call, voicemail
from line.llm_agent.tools.utils import _normalize_tools

# =============================================================================
# Mocks / helpers
# =============================================================================


class _MockStream:
    def __init__(self, chunks: List[StreamChunk]):
        self._chunks = chunks

    async def __aiter__(self):
        accumulated: dict = {}
        for chunk in self._chunks:
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    accumulated[tc.id] = tc
                yield StreamChunk(
                    text=chunk.text, tool_calls=list(accumulated.values()), is_final=chunk.is_final
                )
            else:
                yield chunk


class _MockLLM:
    """Mock main LM that records the tools it was called with each turn."""

    def __init__(self, responses: List[List[StreamChunk]]):
        self._responses = responses
        self._call_count = 0
        self.recorded_tools: List[Optional[List]] = []
        self.closed = False

    def chat(self, messages: List[Message], tools=None, **kwargs):
        self.recorded_tools.append(tools)
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
            self._call_count += 1
            return _MockStream(response)
        return _MockStream([StreamChunk(is_final=True)])

    async def warmup(self, config=None, tools=None):
        pass

    def _set_tools(self, tools):
        pass

    async def aclose(self):
        self.closed = True


def _agent(responses, tools, *, active_turns=1) -> tuple:
    agent = LlmAgent(
        model="gpt-4o",
        api_key="test-key",
        tools=tools,
        voicemail_tool_active_turns=active_turns,
    )
    mock = _MockLLM(responses)
    agent._llm = mock
    return agent, mock


def _turn(text: str) -> UserTurnEnded:
    user_msg = UserTextSent(content=text)
    return UserTurnEnded(content=[UserTextSent(content=text)], history=[user_msg])


async def _collect(agent: LlmAgent, event, tools=None):
    env = TurnEnv(agent_env=AgentEnv())
    return [o async for o in agent.process(env, event, tools=tools) if not isinstance(o, LogMetric)]


def _names(recorded) -> List[str]:
    return [t.name for t in (recorded or [])]


def _voicemail_call() -> List[StreamChunk]:
    return [
        StreamChunk(
            tool_calls=[ToolCall(id="c1", name="voicemail", arguments="{}", is_complete=True)], is_final=True
        )
    ]


# =============================================================================
# Unit
# =============================================================================


def test_is_voicemail_tool_matches_instance_and_function_tool():
    assert _is_voicemail_tool(voicemail) is True
    assert _is_voicemail_tool(voicemail(message="hi")) is True
    assert _is_voicemail_tool(voicemail.as_function_tool()) is True
    assert _is_voicemail_tool(end_call) is False


def test_normalize_tools_resolves_voicemail():
    """VoicemailTool must resolve to a FunctionTool named 'voicemail' (not a loopback callable)."""
    fts, _ = _normalize_tools([voicemail(message="hi"), end_call], parse_model_id("gpt-4o"))
    assert "voicemail" in [t.name for t in fts]
    vm = next(t for t in fts if t.name == "voicemail")
    assert vm.parameters == {}  # exposes no LLM-facing params


# =============================================================================
# Integration: the tool fires
# =============================================================================


async def test_voicemail_tool_fires_speaks_message_and_ends_call():
    agent, _ = _agent([_voicemail_call()], tools=[voicemail(message="Sorry we missed you."), end_call])
    outputs = await _collect(agent, _turn("please leave a message after the tone"))

    sends = [o for o in outputs if isinstance(o, AgentSendText)]
    ends = [o for o in outputs if isinstance(o, AgentEndCall)]
    assert [s.text for s in sends] == ["Sorry we missed you."]
    assert sends[0].interruptible is False
    assert len(ends) == 1
    assert ends[0].reason == "voicemail_detected"
    assert ends[0].interruptible is False


async def test_voicemail_tool_silent_end_when_no_message():
    agent, _ = _agent([_voicemail_call()], tools=[voicemail, end_call])
    outputs = await _collect(agent, _turn("at the tone please record"))

    assert [o for o in outputs if isinstance(o, AgentSendText)] == []
    ends = [o for o in outputs if isinstance(o, AgentEndCall)]
    assert len(ends) == 1 and ends[0].reason == "voicemail_detected"


# =============================================================================
# Integration: tool removal after the conversation starts
# =============================================================================


async def test_voicemail_tool_removed_after_first_turn():
    """Default active_turns=1: tool present on turn 1, gone from turn 2."""
    agent, mock = _agent(
        [[StreamChunk(text="hi", is_final=True)], [StreamChunk(text="hello", is_final=True)]],
        tools=[voicemail, end_call],
        active_turns=1,
    )
    await _collect(agent, _turn("hello, who's this?"))
    await _collect(agent, _turn("okay, go on"))

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "end_call" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])
    assert "end_call" in _names(mock.recorded_tools[1])


async def test_default_active_turns_is_two():
    """Default (no active_turns passed) keeps the tool for turns 1-2, drops it on turn 3."""
    agent = LlmAgent(model="gpt-4o", api_key="test-key", tools=[voicemail, end_call])
    mock = _MockLLM([[StreamChunk(text="a", is_final=True)]] * 3)
    agent._llm = mock
    for _ in range(3):
        await _collect(agent, _turn("hi"))

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" in _names(mock.recorded_tools[1])
    assert "voicemail" not in _names(mock.recorded_tools[2])


async def test_per_call_tools_override_cannot_reintroduce_voicemail_after_window():
    """A per-call `tools` override must not resurrect voicemail once the window closed."""
    agent, mock = _agent([[StreamChunk(text="a", is_final=True)]] * 2, tools=[end_call], active_turns=1)
    await _collect(agent, _turn("hi"), tools=[voicemail(message="x")])  # turn 1: within window
    await _collect(agent, _turn("hi"), tools=[voicemail(message="x")])  # turn 2: window closed

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])


async def test_voicemail_readded_after_window_is_stripped_again():
    """No sticky flag: re-adding voicemail via set_tools after the window is re-stripped."""
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail, end_call], active_turns=1
    )
    await _collect(agent, _turn("hi"))  # turn 1: present
    await _collect(agent, _turn("hi"))  # turn 2: window closed → stripped
    agent.set_tools([voicemail, end_call])  # caller re-adds it dynamically
    await _collect(agent, _turn("hi"))  # turn 3: still closed → stripped again

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])
    assert "voicemail" not in _names(mock.recorded_tools[2])


async def test_voicemail_tool_kept_for_two_turns():
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail, end_call], active_turns=2
    )
    for _ in range(3):
        await _collect(agent, _turn("hi"))

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" in _names(mock.recorded_tools[1])
    assert "voicemail" not in _names(mock.recorded_tools[2])


async def test_voicemail_tool_kept_for_whole_call_when_none():
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail, end_call], active_turns=None
    )
    for _ in range(3):
        await _collect(agent, _turn("hi"))

    for recorded in mock.recorded_tools:
        assert "voicemail" in _names(recorded)


async def test_no_voicemail_tool_is_a_noop():
    """Removal logic must not disturb agents that don't use the voicemail tool."""
    agent, mock = _agent([[StreamChunk(text="a", is_final=True)]] * 2, tools=[end_call], active_turns=1)
    for _ in range(2):
        await _collect(agent, _turn("hi"))
    for recorded in mock.recorded_tools:
        assert _names(recorded) == ["end_call"]


def test_constructor_accepts_voicemail_tool():
    """VoicemailTool instances pass tool validation."""
    agent = LlmAgent(model="gpt-4o", api_key="test-key", tools=[voicemail(message="x"), end_call])
    assert any(isinstance(t, VoicemailTool) for t in agent._tools)
