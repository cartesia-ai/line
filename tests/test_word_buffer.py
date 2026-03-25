"""
Tests for the word_buffer wrapper.

uv run pytest tests/test_word_buffer.py -v
"""

from typing import List, Optional

import pytest

from line.agent import TurnEnv
from line.events import (
    AgentSendText,
    AgentToolCalled,
    InputEvent,
    LogMetric,
    OutputEvent,
    UserTurnEnded,
)
from line.word_buffer import WordBufferingWrapper, word_buffer

# Use anyio for async test support with asyncio backend only
pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


# =============================================================================
# Mock Agent
# =============================================================================


class MockAgent:
    """Agent that yields a predefined sequence of OutputEvents."""

    def __init__(self, outputs: List[OutputEvent]) -> None:
        self._outputs = outputs
        self._cleanup_called = False

    async def process(self, env: TurnEnv, event: InputEvent):
        for output in self._outputs:
            yield output

    async def cleanup(self) -> None:
        self._cleanup_called = True


async def _collect(wrapper: WordBufferingWrapper, event: Optional[InputEvent] = None) -> List[OutputEvent]:
    """Helper to collect all outputs from a wrapper."""
    if event is None:
        event = UserTurnEnded()
    results = []
    async for output in wrapper.process(TurnEnv(), event):
        results.append(output)
    return results


def _texts(outputs: List[OutputEvent]) -> List[str]:
    """Extract text strings from AgentSendText events."""
    return [o.text for o in outputs if isinstance(o, AgentSendText)]


# =============================================================================
# Basic Buffering
# =============================================================================


async def test_basic_word_buffering(anyio_backend):
    """Words split across tokens are reassembled: 'Hum' + 'ana' + ' is' → 'Humana ' then 'is'."""
    agent = MockAgent(
        [
            AgentSendText(text="Hum"),
            AgentSendText(text="ana"),
            AgentSendText(text=" is"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Humana ", "is"]


async def test_first_token_no_leading_space(anyio_backend):
    """First token without leading space is buffered until next space."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello"),
            AgentSendText(text=" world"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello ", "world"]


async def test_punctuation_attached(anyio_backend):
    """Punctuation tokens stay attached to their word."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello"),
            AgentSendText(text=","),
            AgentSendText(text=" world"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello, ", "world"]


async def test_multiple_words_in_one_chunk(anyio_backend):
    """A chunk containing multiple spaces emits all complete words."""
    agent = MockAgent(
        [
            AgentSendText(text="the quick "),
            AgentSendText(text="brown"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["the quick ", "brown"]


async def test_newline_as_word_boundary(anyio_backend):
    """Newlines are treated as word boundaries."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello\n"),
            AgentSendText(text="world"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello\n", "world"]


async def test_tab_as_word_boundary(anyio_backend):
    """Tabs are treated as word boundaries."""
    agent = MockAgent(
        [
            AgentSendText(text="col1\t"),
            AgentSendText(text="col2"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["col1\t", "col2"]


# =============================================================================
# Edge Cases
# =============================================================================


async def test_empty_text(anyio_backend):
    """Empty AgentSendText produces no output."""
    agent = MockAgent(
        [
            AgentSendText(text=""),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == []


async def test_whitespace_only(anyio_backend):
    """Whitespace-only chunks flush correctly."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello"),
            AgentSendText(text=" "),
            AgentSendText(text="world"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello ", "world"]


async def test_single_word_no_spaces(anyio_backend):
    """A single word with no spaces is emitted on final flush."""
    agent = MockAgent(
        [
            AgentSendText(text="Hum"),
            AgentSendText(text="ana"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Humana"]


async def test_final_flush(anyio_backend):
    """Remaining buffer is flushed when stream ends."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello "),
            AgentSendText(text="world"),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello ", "world"]


# =============================================================================
# Non-Text Events
# =============================================================================


async def test_non_text_events_pass_through(anyio_backend):
    """Non-AgentSendText events are yielded immediately, not buffered."""
    metric = LogMetric(name="test", value=42)
    tool_called = AgentToolCalled(tool_call_id="tc1", tool_name="test_tool")
    agent = MockAgent(
        [
            AgentSendText(text="Hello"),
            metric,
            AgentSendText(text=" world"),
            tool_called,
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)

    # metric and tool_called should pass through in order
    assert outputs[0] == metric  # metric before any text (text is buffered)
    assert isinstance(outputs[1], AgentSendText)
    assert outputs[1].text == "Hello "
    assert outputs[2] == tool_called
    assert isinstance(outputs[3], AgentSendText)
    assert outputs[3].text == "world"


# =============================================================================
# Interruptible Flag
# =============================================================================


async def test_interruptible_flag_preserved(anyio_backend):
    """The last interruptible value is used for emitted chunks."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello ", interruptible=False),
            AgentSendText(text="world", interruptible=True),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    text_events = [o for o in outputs if isinstance(o, AgentSendText)]

    assert text_events[0].text == "Hello "
    assert text_events[0].interruptible is False
    # "world" flushed at end — interruptible resets after emission
    assert text_events[1].text == "world"
    assert text_events[1].interruptible is True


async def test_interruptible_and_logic(anyio_backend):
    """If any chunk in a buffered emission is non-interruptible, the emission is non-interruptible."""
    agent = MockAgent(
        [
            AgentSendText(text="Hum", interruptible=True),
            AgentSendText(text="ana", interruptible=False),
            AgentSendText(text=" is", interruptible=True),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    text_events = [o for o in outputs if isinstance(o, AgentSendText)]

    # "Humana " includes a chunk with interruptible=False → entire emission is False
    assert text_events[0].text == "Humana "
    assert text_events[0].interruptible is False
    # "is" remains False — buffer wasn't fully drained after split, so the
    # non-interruptible flag from the earlier chunk is conservatively preserved
    assert text_events[1].text == "is"
    assert text_events[1].interruptible is False


async def test_interruptible_survives_partial_split(anyio_backend):
    """Non-interruptible flag is preserved for text remaining in buffer after a whitespace split."""
    agent = MockAgent(
        [
            AgentSendText(text="world is great", interruptible=False),
            AgentSendText(text=" end", interruptible=True),
        ]
    )
    wrapper = word_buffer(agent)
    outputs = await _collect(wrapper)
    text_events = [o for o in outputs if isinstance(o, AgentSendText)]

    # First emission from the split at "is " boundary
    assert text_events[0].text == "world is "
    assert text_events[0].interruptible is False
    # "great" remained in buffer from the False chunk; second split keeps False
    assert text_events[1].text == "great "
    assert text_events[1].interruptible is False
    # Final flush — "end" still tainted by the non-interruptible buffer
    assert text_events[2].text == "end"
    assert text_events[2].interruptible is False


# =============================================================================
# CJK Auto-Detection (strategy="auto")
# =============================================================================


async def test_cjk_chinese_emitted_immediately(anyio_backend):
    """Chinese characters are emitted immediately without buffering."""
    agent = MockAgent(
        [
            AgentSendText(text="你"),
            AgentSendText(text="好"),
            AgentSendText(text="世界"),
        ]
    )
    wrapper = word_buffer(agent, strategy="auto")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    # Each chunk emitted immediately (no buffering)
    assert texts == ["你", "好", "世界"]


async def test_cjk_japanese_emitted_immediately(anyio_backend):
    """Japanese hiragana/katakana triggers immediate emission."""
    agent = MockAgent(
        [
            AgentSendText(text="こんに"),
            AgentSendText(text="ちは"),
        ]
    )
    wrapper = word_buffer(agent, strategy="auto")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["こんに", "ちは"]


async def test_cjk_korean_emitted_immediately(anyio_backend):
    """Korean hangul triggers immediate emission."""
    agent = MockAgent(
        [
            AgentSendText(text="안녕"),
            AgentSendText(text="하세요"),
        ]
    )
    wrapper = word_buffer(agent, strategy="auto")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["안녕", "하세요"]


async def test_thai_emitted_immediately(anyio_backend):
    """Thai characters trigger immediate emission."""
    agent = MockAgent(
        [
            AgentSendText(text="สวัส"),
            AgentSendText(text="ดี"),
        ]
    )
    wrapper = word_buffer(agent, strategy="auto")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["สวัส", "ดี"]


async def test_mixed_cjk_latin_flushes_buffer(anyio_backend):
    """When CJK arrives while Latin text is buffered, everything flushes."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello"),
            AgentSendText(text="世界"),
        ]
    )
    wrapper = word_buffer(agent, strategy="auto")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    # "Hello" was buffered, then "世界" arrives → CJK detected → entire buffer flushed
    assert texts == ["Hello世界"]


# =============================================================================
# Strategy: "space"
# =============================================================================


async def test_space_strategy_buffers_cjk(anyio_backend):
    """With strategy='space', CJK text is buffered like anything else."""
    agent = MockAgent(
        [
            AgentSendText(text="你好"),
            AgentSendText(text="世界"),
        ]
    )
    wrapper = word_buffer(agent, strategy="space")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    # No spaces → everything flushed at end as one chunk
    assert texts == ["你好世界"]


async def test_space_strategy_still_splits_on_whitespace(anyio_backend):
    """strategy='space' still splits on whitespace for any text."""
    agent = MockAgent(
        [
            AgentSendText(text="Hello "),
            AgentSendText(text="world"),
        ]
    )
    wrapper = word_buffer(agent, strategy="space")
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello ", "world"]


# =============================================================================
# Invalid Strategy
# =============================================================================


async def test_invalid_strategy_raises(anyio_backend):
    """An unknown strategy raises ValueError."""
    agent = MockAgent([])
    with pytest.raises(ValueError, match="Unknown strategy"):
        word_buffer(agent, strategy="invalid")


# =============================================================================
# Cleanup Delegation
# =============================================================================


async def test_cleanup_delegates_to_inner_agent(anyio_backend):
    """cleanup() calls inner agent's cleanup()."""
    agent = MockAgent([])
    wrapper = word_buffer(agent)
    await wrapper.cleanup()
    assert agent._cleanup_called is True


async def test_cleanup_works_without_inner_cleanup(anyio_backend):
    """cleanup() is safe when inner agent has no cleanup method."""

    class NoCleanupAgent:
        async def process(self, env, event):
            return
            yield  # make it an async generator

    wrapper = word_buffer(NoCleanupAgent())
    await wrapper.cleanup()  # should not raise


# =============================================================================
# Callable Agent Support
# =============================================================================


async def test_callable_agent(anyio_backend):
    """word_buffer works with callable agents (not just AgentClass)."""

    async def my_agent(env: TurnEnv, event):
        yield AgentSendText(text="Hello")
        yield AgentSendText(text=" world")

    wrapper = word_buffer(my_agent)
    outputs = await _collect(wrapper)
    texts = _texts(outputs)
    assert texts == ["Hello ", "world"]


# =============================================================================
# Stream Mode (AsyncIterable input)
# =============================================================================


async def test_stream_mode(anyio_backend):
    """word_buffer accepts an AsyncIterable[OutputEvent] directly."""

    async def event_stream():
        yield AgentSendText(text="Hum")
        yield AgentSendText(text="ana")
        yield AgentSendText(text=" is")
        yield AgentSendText(text=" great")

    results = []
    async for output in word_buffer(event_stream()):
        results.append(output)
    texts = _texts(results)
    assert texts == ["Humana ", "is ", "great"]


async def test_stream_mode_with_non_text_events(anyio_backend):
    """Stream mode passes non-text events through immediately."""
    metric = LogMetric(name="test", value=1)

    async def event_stream():
        yield AgentSendText(text="Hello")
        yield metric
        yield AgentSendText(text=" world")

    results = []
    async for output in word_buffer(event_stream()):
        results.append(output)

    assert results[0] == metric
    assert isinstance(results[1], AgentSendText)
    assert results[1].text == "Hello "
    assert isinstance(results[2], AgentSendText)
    assert results[2].text == "world"
