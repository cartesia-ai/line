"""Call-scoped DTMF collection and speech turn suppression."""

import asyncio
from contextlib import contextmanager
from typing import Iterator, Literal, Optional

from line.events import InputEvent, UserDtmfSent, UserTextSent, UserTurnEnded, UserTurnStarted

_BUTTONS = "0123456789*#"
_Status = Literal["complete", "no_input", "too_many_digits"]


class _DtmfCapture:
    def __init__(
        self, inter_digit_timeout: float, first_digit_timeout: float, finish_on_key: str, max_digits: int
    ) -> None:
        self.digits = ""
        self._inter_digit_timeout = inter_digit_timeout
        self._finish_on_key = finish_on_key
        self._max_digits = max_digits
        self._loop = asyncio.get_running_loop()
        self.result: asyncio.Future[_Status] = self._loop.create_future()
        self._timer = self._loop.call_later(first_digit_timeout, self._finish)

    def feed(self, button: str) -> None:
        if self.result.done() or len(button) != 1 or button not in _BUTTONS:
            return
        if button == self._finish_on_key:
            self._finish()
        elif len(self.digits) >= self._max_digits:
            self._finish("too_many_digits")
        else:
            self.digits += button
            self._timer.cancel()
            self._timer = self._loop.call_later(self._inter_digit_timeout, self._finish)

    def _finish(self, status: Optional[_Status] = None) -> None:
        # Timer expiry and a terminator can be ready in the same event-loop turn.
        # Freeze the result synchronously before the tool callback can run.
        if not self.result.done():
            self._timer.cancel()
            self.result.set_result(status or ("complete" if self.digits else "no_input"))

    def close(self) -> None:
        self._timer.cancel()
        self.result.cancel()


class _DtmfInput:
    def __init__(self) -> None:
        self._capture: Optional[_DtmfCapture] = None
        self._suppress_turn = False
        self._speech_idle = asyncio.Event()
        self._speech_idle.set()

    @property
    def active(self) -> bool:
        return self._capture is not None

    @contextmanager
    def collect(
        self, inter_digit_timeout: float, first_digit_timeout: float, finish_on_key: str, max_digits: int
    ) -> Iterator[_DtmfCapture]:
        if self.active:
            raise RuntimeError("A DTMF collection is already active for this call")
        capture = _DtmfCapture(inter_digit_timeout, first_digit_timeout, finish_on_key, max_digits)
        self._capture = capture
        self._suppress_turn = not self._speech_idle.is_set()
        try:
            yield capture
        finally:
            # A cancelled generator may close after a new collection starts.
            if self._capture is capture:
                self.cancel()

    def cancel(self) -> None:
        """Release the collector while preserving the overlapping speech turn."""
        if self._capture is not None:
            self._capture.close()
            self._capture = None

    def handle_event(self, event: InputEvent) -> bool:
        """Return whether this event belongs to the protected keypad step."""
        if isinstance(event, UserDtmfSent) and self._capture is not None:
            self._capture.feed(event.button)
            return True
        if isinstance(event, UserTurnStarted):
            self._speech_idle.clear()
            self._suppress_turn = self.active
            return self._suppress_turn
        if isinstance(event, UserTextSent):
            if self.active:
                self._suppress_turn = True
            return self._suppress_turn
        if isinstance(event, UserTurnEnded):
            suppress = self.active or self._suppress_turn
            self._speech_idle.set()
            self._suppress_turn = False
            return suppress
        return False

    async def wait_for_speech_end(self) -> None:
        while not self._speech_idle.is_set():
            await self._speech_idle.wait()

    def close(self) -> None:
        self.cancel()
        self._suppress_turn = False
        self._speech_idle.set()
