"""Word-buffering wrapper for pronunciation dictionary support.

LLM streaming breaks words across token boundaries (e.g., "Humana" → " Hum" + "ana").
TTS pronunciation dictionaries match per-chunk, so split words never match.
This wrapper buffers streamed text and emits complete words.
"""

from __future__ import annotations

from typing import AsyncIterable, Union, overload

from line.agent import Agent, TurnEnv, call_agent
from line.events import AgentSendText, InputEvent, OutputEvent


@overload
def word_buffer(
    source: AsyncIterable[OutputEvent], *, strategy: str = "auto"
) -> AsyncIterable[OutputEvent]: ...


@overload
def word_buffer(source: Agent, *, strategy: str = "auto") -> WordBufferingWrapper: ...


def word_buffer(
    source: Union[Agent, AsyncIterable[OutputEvent]], *, strategy: str = "auto"
) -> Union[WordBufferingWrapper, AsyncIterable[OutputEvent]]:
    """Buffer text and emit complete words for pronunciation dictionary support.

    Accepts either an Agent or an AsyncIterable of OutputEvents.

    Args:
        source: An Agent to wrap, or an AsyncIterable[OutputEvent] to buffer.
        strategy: Buffering strategy.
            "auto" (default): Buffer on whitespace for space-delimited scripts,
                emit immediately for CJK/Thai/Khmer/Lao scripts.
            "space": Always buffer on whitespace boundaries only.

    Returns:
        If given an Agent: a wrapped agent satisfying the Agent protocol.
        If given an AsyncIterable: a buffered AsyncIterable[OutputEvent].
    """
    if strategy not in ("auto", "space"):
        raise ValueError(f"Unknown strategy: {strategy!r}. Must be 'auto' or 'space'.")
    if hasattr(source, "__aiter__"):
        return _buffer_events(source, strategy)  # type: ignore[arg-type]
    return WordBufferingWrapper(source, strategy=strategy)  # type: ignore[arg-type]


class WordBufferingWrapper:
    """Buffers AgentSendText and emits complete words.

    Ensures words are never split across chunks, allowing pronunciation
    overrides to work correctly.
    """

    def __init__(self, agent: Agent, strategy: str = "auto") -> None:
        self._agent = agent
        self._strategy = strategy

    async def process(self, env: TurnEnv, event: InputEvent) -> AsyncIterable[OutputEvent]:
        async for output in _buffer_events(call_agent(self._agent, env, event), self._strategy):
            yield output

    async def cleanup(self) -> None:
        """Delegate cleanup to the inner agent if it supports it."""
        if hasattr(self._agent, "cleanup"):
            await self._agent.cleanup()


async def _buffer_events(events: AsyncIterable[OutputEvent], strategy: str) -> AsyncIterable[OutputEvent]:
    """Core buffering logic operating on an async iterable of events."""
    text_buffer = ""
    last_interruptible = True

    async for output in events:
        if not isinstance(output, AgentSendText):
            yield output
            continue

        text_buffer += output.text
        last_interruptible = last_interruptible and output.interruptible

        # Auto strategy: if new text contains CJK/spaceless script, flush immediately
        if strategy == "auto" and _contains_spaceless_script(output.text):
            if text_buffer:
                yield AgentSendText(text=text_buffer, interruptible=last_interruptible)
                text_buffer = ""
                last_interruptible = True
            continue

        # Emit complete words (up to last whitespace)
        last_ws = _rfind_whitespace(text_buffer)
        if last_ws >= 0:
            to_emit = text_buffer[: last_ws + 1]
            text_buffer = text_buffer[last_ws + 1 :]
            if to_emit:
                yield AgentSendText(text=to_emit, interruptible=last_interruptible)
                if not text_buffer:
                    last_interruptible = True

    # Flush remaining buffer
    if text_buffer:
        yield AgentSendText(text=text_buffer, interruptible=last_interruptible)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHITESPACE = frozenset(" \t\n\r\x0b\x0c")


def _rfind_whitespace(text: str) -> int:
    """Return the index of the last whitespace character, or -1."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] in _WHITESPACE:
            return i
    return -1


def _contains_spaceless_script(text: str) -> bool:
    """Check if text contains characters from scripts that don't use spaces between words."""
    for ch in text:
        if _is_spaceless_char(ch):
            return True
    return False


def _is_spaceless_char(ch: str) -> bool:
    """Check if a character belongs to a script without inter-word spaces."""
    cp = ord(ch)
    return (
        # CJK Unified Ideographs
        0x4E00 <= cp <= 0x9FFF
        # CJK Extension A
        or 0x3400 <= cp <= 0x4DBF
        # CJK Compatibility Ideographs
        or 0xF900 <= cp <= 0xFAFF
        # Hiragana
        or 0x3040 <= cp <= 0x309F
        # Katakana
        or 0x30A0 <= cp <= 0x30FF
        # Hangul Syllables
        or 0xAC00 <= cp <= 0xD7AF
        # Hangul Jamo
        or 0x1100 <= cp <= 0x11FF
        # Thai
        or 0x0E00 <= cp <= 0x0E7F
        # Lao
        or 0x0E80 <= cp <= 0x0EFF
        # Khmer
        or 0x1780 <= cp <= 0x17FF
    )
