"""Unit + integration tests for the built-in voicemail tool and turn-limited removal.

These tests mock the LLM provider — no network. They cover:
  - unit: VoicemailTool resolves through tool normalization; the ClassTool protocol;
    `active_turns` defaults/chaining.
  - integration: the tool fires (speaks the message + ends the call with
    reason="voicemail_detected"), and the agent drops any tool from its options
    once that tool's `active_turns` window has elapsed so it can't fire mid-call.

    uv run pytest tests/test_llm_agent_voicemail_tool.py -v
"""

from typing import List, Optional

from line.agent import AgentEnv, TurnEnv
from line.events import (
    AgentEndCall,
    AgentSendText,
    CallStarted,
    LogMetric,
    UserTextSent,
    UserTurnEnded,
)
from line.llm_agent.llm_agent import LlmAgent
from line.llm_agent.provider import Message, StreamChunk, ToolCall, parse_model_id
from line.llm_agent.tools.system import VoicemailTool, end_call, knowledge_base, transfer_call, voicemail
from line.llm_agent.tools.utils import ClassTool, _normalize_tools

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


def _agent(responses, tools) -> tuple:
    """Build an LlmAgent with a mock LM. `active_turns` lives on the tools themselves."""
    agent = LlmAgent(model="gpt-4o", api_key="test-key", tools=tools)
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


def test_class_tools_satisfy_protocol():
    """The built-in class tools satisfy ClassTool; web/function tools do not."""
    from line.llm_agent.tools.system import web_search
    from line.llm_agent.tools.utils import FunctionTool

    assert isinstance(voicemail, ClassTool)
    assert isinstance(end_call, ClassTool)
    assert isinstance(transfer_call, ClassTool)
    assert isinstance(knowledge_base, ClassTool)
    assert not isinstance(web_search, ClassTool)
    assert not isinstance(voicemail.as_function_tool(), ClassTool)
    assert not isinstance(
        FunctionTool(name="x", description="d", func=lambda c: None, parameters={}), ClassTool
    )


def test_active_turns_defaults_and_chaining():
    """active_turns defaults to None for all built-ins; chaining inherits/overrides."""
    assert voicemail.active_turns is None
    assert end_call.active_turns is None
    assert voicemail(active_turns=5).active_turns == 5
    assert voicemail(active_turns=None).active_turns is None  # explicit None disables removal
    assert voicemail(active_turns=2)(message="x").active_turns == 2  # omitted → inherited
    assert end_call(active_turns=3).active_turns == 3


def test_voicemail_chaining_preserves_prior_config():
    """Re-configuring voicemail inherits omitted fields (chain-safe)."""
    configured = voicemail(message="Call us back.", interruptible=True, active_turns=5)
    rechained = configured(active_turns=1)  # only change active_turns
    assert rechained.message == "Call us back."
    assert rechained.interruptible is True
    assert rechained.active_turns == 1


def test_as_function_tool_carries_active_turns():
    """active_turns travels onto the resolved FunctionTool so removal still applies."""
    assert voicemail(active_turns=3).as_function_tool().active_turns == 3
    assert end_call.as_function_tool().active_turns is None


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
# Integration: turn-limited removal (generic over any ClassTool's active_turns)
# =============================================================================


async def test_voicemail_tool_removed_after_first_turn():
    """active_turns=1: tool present on turn 1, gone from turn 2."""
    agent, mock = _agent(
        [[StreamChunk(text="hi", is_final=True)], [StreamChunk(text="hello", is_final=True)]],
        tools=[voicemail(active_turns=1), end_call],
    )
    await _collect(agent, _turn("hello, who's this?"))
    await _collect(agent, _turn("okay, go on"))

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "end_call" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])
    assert "end_call" in _names(mock.recorded_tools[1])


async def test_default_active_turns_is_none():
    """Default voicemail (active_turns=None) stays available for the whole call.

    Real voicemail greetings fragment into a variable number of short turns that
    can exhaust a small finite window before the agent first responds, so the
    default keeps the tool available; set a finite active_turns to force removal.
    """
    agent, mock = _agent([[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail, end_call])
    for _ in range(3):
        await _collect(agent, _turn("hi"))

    for recorded in mock.recorded_tools:
        assert "voicemail" in _names(recorded)


async def test_any_class_tool_with_active_turns_is_removed():
    """Removal is generic — not voicemail-specific. end_call(active_turns=1) is dropped too."""
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 2,
        tools=[end_call(active_turns=1), voicemail(active_turns=None)],
    )
    await _collect(agent, _turn("hi"))
    await _collect(agent, _turn("hi"))

    assert "end_call" in _names(mock.recorded_tools[0])
    assert "end_call" not in _names(mock.recorded_tools[1])  # dropped after its window
    assert "voicemail" in _names(mock.recorded_tools[1])  # active_turns=None → kept


async def test_per_call_tools_override_cannot_reintroduce_windowed_tool():
    """A per-call `tools` override must not resurrect a tool once its window closed."""
    agent, mock = _agent([[StreamChunk(text="a", is_final=True)]] * 2, tools=[end_call])
    await _collect(agent, _turn("hi"), tools=[voicemail(active_turns=1)])  # turn 1: within window
    await _collect(agent, _turn("hi"), tools=[voicemail(active_turns=1)])  # turn 2: window closed

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])


async def test_prereresolved_function_tool_is_still_windowed():
    """A pre-resolved FunctionTool (voicemail(...).as_function_tool()) is still dropped."""
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 2,
        tools=[voicemail(active_turns=1).as_function_tool(), end_call],
    )
    await _collect(agent, _turn("hi"))
    await _collect(agent, _turn("hi"))

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])


async def test_tool_readded_after_window_is_stripped_again():
    """No sticky flag: re-adding a windowed tool via set_tools after the window is re-stripped."""
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail(active_turns=1), end_call]
    )
    await _collect(agent, _turn("hi"))  # turn 1: present
    await _collect(agent, _turn("hi"))  # turn 2: window closed → stripped
    agent.set_tools([voicemail(active_turns=1), end_call])  # caller re-adds it dynamically
    await _collect(agent, _turn("hi"))  # turn 3: still closed → stripped again

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])
    assert "voicemail" not in _names(mock.recorded_tools[2])


async def test_tool_kept_for_two_turns():
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail(active_turns=2), end_call]
    )
    for _ in range(3):
        await _collect(agent, _turn("hi"))

    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" in _names(mock.recorded_tools[1])
    assert "voicemail" not in _names(mock.recorded_tools[2])


async def test_tool_kept_for_whole_call_when_active_turns_none():
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 3, tools=[voicemail(active_turns=None), end_call]
    )
    for _ in range(3):
        await _collect(agent, _turn("hi"))

    for recorded in mock.recorded_tools:
        assert "voicemail" in _names(recorded)


async def test_tools_without_active_turns_are_a_noop():
    """Tools with no active_turns (here end_call default None) are never dropped."""
    agent, mock = _agent([[StreamChunk(text="a", is_final=True)]] * 3, tools=[end_call])
    for _ in range(3):
        await _collect(agent, _turn("hi"))
    for recorded in mock.recorded_tools:
        assert _names(recorded) == ["end_call"]


async def test_call_started_resets_turn_window():
    """A reused agent: CallStarted resets the window so voicemail is available again."""
    agent, mock = _agent(
        [[StreamChunk(text="a", is_final=True)]] * 5, tools=[voicemail(active_turns=1), end_call]
    )
    # First "call": tool present turn 1, dropped turn 2.
    await _collect(agent, _turn("hi"))
    await _collect(agent, _turn("hi"))
    assert "voicemail" in _names(mock.recorded_tools[0])
    assert "voicemail" not in _names(mock.recorded_tools[1])

    # New call on the same instance resets the counter.
    await _collect(agent, CallStarted())
    await _collect(agent, _turn("new call opening"))
    assert "voicemail" in _names(mock.recorded_tools[-1])  # available again on turn 1


def test_constructor_accepts_voicemail_tool():
    """VoicemailTool instances pass tool validation."""
    agent = LlmAgent(model="gpt-4o", api_key="test-key", tools=[voicemail(message="x"), end_call])
    assert any(isinstance(t, VoicemailTool) for t in agent._tools)
