"""LlmAgent voicemail tests for both evaluation approaches.

Approach 1: the built-in ``voicemail`` tool is dropped from the agent's options
once the conversation is "deemed started" (after
``VoicemailDetectionConfig.tool_active_turns`` completed user turns).

Approach 2: the cheap-LM voicemail detection sidecar buffers the main LM's first
user-visible output behind a short gate and suppresses it on a ``voicemail``
verdict, ending the call with ``reason="voicemail_detected"``.

    uv run pytest tests/test_llm_agent_voicemail.py -v
"""

import asyncio
from typing import List, Optional

from line.agent import AgentEnv, TurnEnv
from line.events import (
    AgentEndCall,
    AgentSendText,
    CallEnded,
    LogMetric,
    UserTextSent,
    UserTurnEnded,
)
from line.llm_agent.config import LlmConfig
from line.llm_agent.llm_agent import LlmAgent
from line.llm_agent.provider import Message, StreamChunk
from line.llm_agent.tools.system import end_call, voicemail
from line.llm_agent.voicemail_detection import VoicemailDetectionConfig, VoicemailDetectionResult

# =============================================================================
# Mocks
# =============================================================================


class _MockStream:
    def __init__(self, chunks: List[StreamChunk]):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
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


class _FakeDetector:
    """Stand-in for _VoicemailDetector with a fixed verdict and optional delay."""

    def __init__(self, result: VoicemailDetectionResult, delay: float = 0.0):
        self._result = result
        self._delay = delay
        self.calls: List[str] = []
        self.closed = False

    async def classify(self, transcript: str) -> VoicemailDetectionResult:
        self.calls.append(transcript)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._result

    async def aclose(self):
        self.closed = True


def _turn(text: str) -> UserTurnEnded:
    """A completed user turn carrying a single transcript line + matching history."""
    user_msg = UserTextSent(content=text)
    return UserTurnEnded(content=[UserTextSent(content=text)], history=[user_msg])


async def _collect(agent: LlmAgent, env: TurnEnv, event, include_metrics: bool = False):
    outputs = []
    async for output in agent.process(env, event):
        if not include_metrics and isinstance(output, LogMetric):
            continue
        outputs.append(output)
    return outputs


def _env() -> TurnEnv:
    return TurnEnv(agent_env=AgentEnv())


def _tool_names(recorded) -> List[str]:
    return [t.name for t in (recorded or [])]


# =============================================================================
# Approach 1: voicemail tool removal after N user turns
# =============================================================================


async def test_voicemail_tool_removed_after_first_turn():
    """Default active_turns=1: tool present turn 1, gone turn 2."""
    agent = LlmAgent(
        model="gpt-4o",
        api_key="test-key",
        tools=[voicemail, end_call],
        voicemail_detection=VoicemailDetectionConfig(tool_active_turns=1),
    )
    mock = _MockLLM([[StreamChunk(text="hi", is_final=True)], [StreamChunk(text="hello", is_final=True)]])
    agent._llm = mock

    await _collect(agent, _env(), _turn("please leave a message"))
    await _collect(agent, _env(), _turn("ok thanks"))

    assert "voicemail" in _tool_names(mock.recorded_tools[0])
    assert "end_call" in _tool_names(mock.recorded_tools[0])
    assert "voicemail" not in _tool_names(mock.recorded_tools[1])
    assert "end_call" in _tool_names(mock.recorded_tools[1])


async def test_voicemail_tool_kept_for_two_turns():
    """active_turns=2 keeps the tool through turn 2, drops it on turn 3."""
    agent = LlmAgent(
        model="gpt-4o",
        api_key="test-key",
        tools=[voicemail, end_call],
        voicemail_detection=VoicemailDetectionConfig(tool_active_turns=2),
    )
    mock = _MockLLM([[StreamChunk(text="a", is_final=True)]] * 3)
    agent._llm = mock

    for _ in range(3):
        await _collect(agent, _env(), _turn("hi"))

    assert "voicemail" in _tool_names(mock.recorded_tools[0])
    assert "voicemail" in _tool_names(mock.recorded_tools[1])
    assert "voicemail" not in _tool_names(mock.recorded_tools[2])


async def test_voicemail_tool_kept_when_active_turns_none():
    """active_turns=None keeps the tool for the whole call."""
    agent = LlmAgent(
        model="gpt-4o",
        api_key="test-key",
        tools=[voicemail, end_call],
        voicemail_detection=VoicemailDetectionConfig(tool_active_turns=None),
    )
    mock = _MockLLM([[StreamChunk(text="a", is_final=True)]] * 3)
    agent._llm = mock

    for _ in range(3):
        await _collect(agent, _env(), _turn("hi"))

    for recorded in mock.recorded_tools:
        assert "voicemail" in _tool_names(recorded)


# =============================================================================
# Approach 2: voicemail detection sidecar
# =============================================================================


def _detection_agent(
    detector: _FakeDetector,
    responses,
    *,
    message="Call us back.",
    gate_ms=200,
    min_words=0,
    active_turns=None,
):
    agent = LlmAgent(
        model="gpt-4o",
        api_key="test-key",
        config=LlmConfig(),
        voicemail_detection=VoicemailDetectionConfig(
            model="openai/gpt-4o-mini",
            api_key="test-key",
            message=message,
            initial_gate_ms=gate_ms,
            min_transcript_words=min_words,
            active_turns=active_turns,
        ),
    )
    agent._llm = _MockLLM(responses)
    agent._voicemail_detector = detector
    return agent


async def test_voicemail_in_gate_suppresses_and_ends_call():
    detector = _FakeDetector(VoicemailDetectionResult("voicemail", "beep"))
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="This is the main agent response.", is_final=True)]],
        message="Sorry we missed you, call us back.",
    )
    outputs = await _collect(agent, _env(), _turn("please leave a message after the tone"))

    # Main text is suppressed; only the fixed voicemail message + end call remain.
    texts = [o for o in outputs if isinstance(o, AgentSendText)]
    ends = [o for o in outputs if isinstance(o, AgentEndCall)]
    assert [t.text for t in texts] == ["Sorry we missed you, call us back."]
    assert texts[0].interruptible is False
    assert len(ends) == 1
    assert ends[0].reason == "voicemail_detected"
    assert ends[0].interruptible is False
    assert detector.calls == ["please leave a message after the tone"]


async def test_voicemail_in_gate_without_message_ends_silently():
    detector = _FakeDetector(VoicemailDetectionResult("voicemail"))
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="main response", is_final=True)]],
        message=None,
    )
    outputs = await _collect(agent, _env(), _turn("at the tone please record"))

    assert [o for o in outputs if isinstance(o, AgentSendText)] == []
    ends = [o for o in outputs if isinstance(o, AgentEndCall)]
    assert len(ends) == 1 and ends[0].reason == "voicemail_detected"


async def test_human_verdict_releases_main_output():
    detector = _FakeDetector(VoicemailDetectionResult("human", "interactive"))
    agent = _detection_agent(detector, [[StreamChunk(text="How can I help you today?", is_final=True)]])
    outputs = await _collect(agent, _env(), _turn("hello who is this"))

    texts = [o.text for o in outputs if isinstance(o, AgentSendText)]
    assert texts == ["How can I help you today?"]
    assert [o for o in outputs if isinstance(o, AgentEndCall)] == []


async def test_unknown_verdict_releases_main_output():
    detector = _FakeDetector(VoicemailDetectionResult("unknown", "noisy"))
    agent = _detection_agent(detector, [[StreamChunk(text="Hi there.", is_final=True)]])
    outputs = await _collect(agent, _env(), _turn("...static..."))

    assert [o.text for o in outputs if isinstance(o, AgentSendText)] == ["Hi there."]
    assert [o for o in outputs if isinstance(o, AgentEndCall)] == []


async def test_detector_slower_than_gate_does_not_suppress():
    """A voicemail verdict that arrives after the gate is ignored (output released)."""
    detector = _FakeDetector(VoicemailDetectionResult("voicemail"), delay=0.3)
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="Main agent text.", is_final=True)]],
        gate_ms=20,
    )
    outputs = await _collect(agent, _env(), _turn("ambiguous greeting"))

    # Gate elapsed before the detector returned → main output released, no end call.
    assert [o.text for o in outputs if isinstance(o, AgentSendText)] == ["Main agent text."]
    assert [o for o in outputs if isinstance(o, AgentEndCall)] == []


async def test_late_voicemail_after_release_is_ignored():
    """Once the first main output is released, a later voicemail verdict is ignored."""
    detector = _FakeDetector(VoicemailDetectionResult("voicemail"), delay=0.05)
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="First. ", is_final=False), StreamChunk(text="Second.", is_final=True)]],
        gate_ms=10,
    )
    outputs = await _collect(agent, _env(), _turn("greeting"))

    texts = [o.text for o in outputs if isinstance(o, AgentSendText)]
    assert "First. " in texts and "Second." in texts
    assert [o for o in outputs if isinstance(o, AgentEndCall)] == []


async def test_empty_user_turn_skips_detection():
    detector = _FakeDetector(VoicemailDetectionResult("voicemail"))
    agent = LlmAgent(
        model="gpt-4o",
        api_key="test-key",
        voicemail_detection=VoicemailDetectionConfig(
            model="openai/gpt-4o-mini", api_key="test-key", message="cb"
        ),
    )
    agent._llm = _MockLLM([[StreamChunk(text="normal", is_final=True)]])
    agent._voicemail_detector = detector

    # An empty user turn (no transcript text) should not invoke the detector.
    event = UserTurnEnded(content=[], history=[UserTextSent(content="prior context")])
    outputs = await _collect(agent, _env(), event)

    assert detector.calls == []
    assert [o.text for o in outputs if isinstance(o, AgentSendText)] == ["normal"]


async def test_short_turn_below_min_words_skips_detection():
    """A turn shorter than min_transcript_words is not classified (never hangs up)."""
    detector = _FakeDetector(VoicemailDetectionResult("voicemail"))
    agent = _detection_agent(detector, [[StreamChunk(text="Hi there!", is_final=True)]], min_words=5)
    # "Hello?" is one word — below the threshold, so detection is skipped.
    outputs = await _collect(agent, _env(), _turn("Hello?"))

    assert detector.calls == []
    assert [o.text for o in outputs if isinstance(o, AgentSendText)] == ["Hi there!"]
    assert [o for o in outputs if isinstance(o, AgentEndCall)] == []


async def test_turn_at_min_words_runs_detection():
    """A turn meeting min_transcript_words is classified normally."""
    detector = _FakeDetector(VoicemailDetectionResult("voicemail"))
    agent = _detection_agent(detector, [[StreamChunk(text="main", is_final=True)]], min_words=5)
    outputs = await _collect(agent, _env(), _turn("please leave a message after the tone"))

    assert detector.calls == ["please leave a message after the tone"]
    assert [o for o in outputs if isinstance(o, AgentEndCall)][0].reason == "voicemail_detected"


async def test_detection_limited_to_active_turns():
    """With active_turns=1, detection runs on turn 1 only, not later turns."""
    detector = _FakeDetector(VoicemailDetectionResult("human"))
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="a", is_final=True)], [StreamChunk(text="b", is_final=True)]],
        active_turns=1,
    )
    await _collect(agent, _env(), _turn("opening line here"))
    await _collect(agent, _env(), _turn("second turn here"))

    # Detector only consulted on the opening turn; the in-conversation turn skips it.
    assert detector.calls == ["opening line here"]


async def test_detection_runs_every_turn_when_active_turns_none():
    detector = _FakeDetector(VoicemailDetectionResult("human"))
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="a", is_final=True)], [StreamChunk(text="b", is_final=True)]],
        active_turns=None,
    )
    await _collect(agent, _env(), _turn("first turn text"))
    await _collect(agent, _env(), _turn("second turn text"))

    assert detector.calls == ["first turn text", "second turn text"]


async def test_cleanup_closes_detector_provider():
    detector = _FakeDetector(VoicemailDetectionResult("human"))
    agent = _detection_agent(detector, [[StreamChunk(text="hi", is_final=True)]])

    await agent.cleanup()
    assert detector.closed is True


async def test_call_ended_closes_detector_provider():
    detector = _FakeDetector(VoicemailDetectionResult("human"))
    agent = _detection_agent(detector, [[StreamChunk(text="hi", is_final=True)]])

    await _collect(agent, _env(), CallEnded())
    assert detector.closed is True


async def test_voicemail_tool_call_count_unaffected_by_detection():
    """The two approaches are independent — detection alone yields one classify call/turn."""
    detector = _FakeDetector(VoicemailDetectionResult("human"))
    agent = _detection_agent(
        detector,
        [[StreamChunk(text="a", is_final=True)], [StreamChunk(text="b", is_final=True)]],
    )
    await _collect(agent, _env(), _turn("first"))
    await _collect(agent, _env(), _turn("second"))
    assert detector.calls == ["first", "second"]
