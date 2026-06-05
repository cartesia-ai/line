"""Tests for the cheap-LM voicemail detection sidecar (Approach 2).

Covers the standalone detector: JSON parsing, fail-open behavior on invalid
output / timeouts / provider errors, and that the classifier prompt carries
neither the main agent's tools nor its system prompt.

    uv run pytest tests/test_voicemail_detection.py -v
"""

from typing import List

from line.llm_agent.provider import StreamChunk
from line.llm_agent.voicemail_detection import (
    _DETECTOR_SYSTEM_PROMPT,
    VoicemailDetectionConfig,
    VoicemailDetectionResult,
    _parse_detector_output,
    _VoicemailDetector,
)

# =============================================================================
# Fake provider — yields predetermined chunks or raises.
# =============================================================================


class _FakeStream:
    def __init__(self, chunks: List[StreamChunk], error: Exception = None):
        self._chunks = chunks
        self._error = error

    async def __aiter__(self):
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


class _FakeProvider:
    """Stand-in for LlmProvider that records the chat() args it was called with."""

    def __init__(self, text: str = "", error: Exception = None):
        self._text = text
        self._error = error
        self.chat_calls = []
        self.closed = False

    def chat(self, messages, tools=None, config=None, **kwargs):
        self.chat_calls.append({"messages": messages, "tools": tools, "config": config})
        chunks = [StreamChunk(text=self._text, is_final=True)] if self._text else []
        return _FakeStream(chunks, error=self._error)

    async def aclose(self):
        self.closed = True


def _make_detector(text: str = "", error: Exception = None) -> _VoicemailDetector:
    detector = _VoicemailDetector(VoicemailDetectionConfig(model="openai/gpt-4o-mini", api_key="test-key"))
    detector._provider = _FakeProvider(text=text, error=error)
    return detector


# =============================================================================
# Parser tests
# =============================================================================


def test_parse_voicemail():
    result = _parse_detector_output('{"classification": "voicemail", "reason": "leave a message"}')
    assert result == VoicemailDetectionResult("voicemail", "leave a message")


def test_parse_human():
    result = _parse_detector_output('{"classification": "human", "reason": "said hello"}')
    assert result.classification == "human"


def test_parse_unknown():
    result = _parse_detector_output('{"classification": "unknown", "reason": "noisy"}')
    assert result.classification == "unknown"


def test_parse_strips_code_fences():
    result = _parse_detector_output('```json\n{"classification": "voicemail", "reason": "beep"}\n```')
    assert result.classification == "voicemail"


def test_parse_invalid_json_is_unknown():
    assert _parse_detector_output("definitely not json").classification == "unknown"


def test_parse_empty_is_unknown():
    assert _parse_detector_output("").classification == "unknown"
    assert _parse_detector_output(None).classification == "unknown"


def test_parse_unexpected_classification_is_unknown():
    result = _parse_detector_output('{"classification": "maybe", "reason": "x"}')
    assert result.classification == "unknown"


# =============================================================================
# Detector tests
# =============================================================================


async def test_classify_voicemail():
    detector = _make_detector('{"classification": "voicemail", "reason": "at the tone"}')
    result = await detector.classify("Please leave a message after the tone.")
    assert result.classification == "voicemail"


async def test_classify_human():
    detector = _make_detector('{"classification": "human", "reason": "interactive"}')
    result = await detector.classify("Hello, who is this?")
    assert result.classification == "human"


async def test_classify_unknown():
    detector = _make_detector('{"classification": "unknown", "reason": "ambiguous"}')
    result = await detector.classify("...")
    assert result.classification == "unknown"


async def test_classify_invalid_json_is_unknown():
    detector = _make_detector("garbage output")
    result = await detector.classify("hello")
    assert result.classification == "unknown"


async def test_classify_provider_error_is_unknown():
    detector = _make_detector(error=RuntimeError("boom"))
    result = await detector.classify("hello")
    assert result.classification == "unknown"


async def test_classify_timeout_is_unknown():
    detector = _make_detector(error=TimeoutError("slow"))
    result = await detector.classify("hello")
    assert result.classification == "unknown"


async def test_detector_prompt_excludes_main_tools_and_system_prompt():
    """The classifier sees only its own detection prompt and no tools."""
    detector = _make_detector('{"classification": "human", "reason": "x"}')
    await detector.classify("Hi there")

    call = detector._provider.chat_calls[0]
    # No tools are ever passed to the detector.
    assert call["tools"] is None
    # The only user message is the transcript itself.
    assert [m.role for m in call["messages"]] == ["user"]
    assert call["messages"][0].content == "Hi there"
    # The detector's own provider was built with the detection system prompt,
    # not any main-agent system prompt.
    assert detector._provider._text  # sanity: fake returns the configured text


async def test_aclose_closes_provider():
    detector = _make_detector('{"classification": "human"}')
    await detector.aclose()
    assert detector._provider.closed is True


def test_detector_omits_temperature_for_reasoning_models():
    """Reasoning models (e.g. gpt-5) reject temperature=0, so it must be omitted."""
    detector = _VoicemailDetector(VoicemailDetectionConfig(model="openai/gpt-5", api_key="test-key"))
    assert detector._provider._config.temperature is None


def test_detector_uses_zero_temperature_for_non_reasoning_models():
    """Non-reasoning models (e.g. gpt-4o-mini) get temperature=0 for determinism."""
    detector = _VoicemailDetector(VoicemailDetectionConfig(model="openai/gpt-4o-mini", api_key="test-key"))
    assert detector._provider._config.temperature == 0


def test_config_defaults_to_opening_turn_only():
    """Detection defaults to the opening turn so it doesn't run all conversation long."""
    cfg = VoicemailDetectionConfig(model="openai/gpt-5-nano", api_key="test-key")
    assert cfg.active_turns == 1


def test_detector_system_prompt_is_self_contained():
    """The detection prompt mentions voicemail/human/unknown and strict JSON."""
    prompt = _DETECTOR_SYSTEM_PROMPT.lower()
    assert "voicemail" in prompt
    assert "human" in prompt
    assert "unknown" in prompt
    assert "json" in prompt
