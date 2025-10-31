from collections.abc import Generator
import re
from typing import Union

from line.events import AgentResponse, DTMFOutputEvent

DTMF_EXPRESSION = "dtmf="


class DTMFLookAheadStringBuffer:
    """
    Wrapper ontop of DTMFLookAheadCharacterBuffer, but will yield strings instead of characters
    """

    def __init__(self):
        self.buffer = DTMFLookAheadCharacterBuffer()

    def feed(self, string: str) -> Generator[Union[AgentResponse, DTMFOutputEvent], None, None]:
        for char in string:
            for item in self.buffer.feed(char):
                if isinstance(item, DTMFOutputEvent):
                    for digit in item.button:
                        yield DTMFOutputEvent(button=digit)
                else:
                    yield item

    def flush(self) -> Generator[Union[AgentResponse, DTMFOutputEvent], None, None]:
        for item in self.buffer.flush():
            if isinstance(item, DTMFOutputEvent):
                for digit in split_dtmf_output(item):
                    yield digit
            else:
                yield item


def split_dtmf_output(item: DTMFOutputEvent) -> Generator[DTMFOutputEvent, None, None]:
    """
    DTMFOutputEvent(button="12") -> [DTMFOutputEvent(button="1"), DTMFOutputEvent(button="2")]
    """
    for digit in item.button:
        yield DTMFOutputEvent(button=digit)


class DTMFLookAheadCharacterBuffer:
    """
    A look ahead buffer that will replace DTMF expressions with DTMF output events

    Why do we have this:
    - Sometimes, gemini will yield ["hello dtmf", "=123 world"]
    - We want to yield [AgentResponse("hello"), DTMFOutputEvent("123"), AgentResponse("world")]
    - This is a look ahead buffer, so we need to keep track of the buffer and the chunks
    """

    def __init__(self):
        self.non_dtmf_buffer = ""
        self.dtmf_buffer = ""

        self.chunks = []

        self.full_expression = re.compile(r"dtmf=(\d+)")
        self.dtmf_preamble = "dtmf="

    def feed(self, char: str) -> Generator[Union[AgentResponse, DTMFOutputEvent], None, None]:
        """
        Feed a character into the buffer and see if we yield anything
        """
        # new character is in a word boundary: flush
        is_word_char = re.match(r"[\w0-9=]", char)
        # No match and will not be a match: buffer next char onto non-dtmf buffer
        if not self.dtmf_buffer and not self.dtmf_preamble.startswith(char):
            self.non_dtmf_buffer += char
            return

        # No match and might be a match: buffer next char to dtmf buffer
        if not self.dtmf_buffer and self.dtmf_preamble.startswith(char):
            self.dtmf_buffer += char
            return

        if self.dtmf_buffer and not is_word_char:
            # We are in a match, flush both
            if self.dtmf_buffer.startswith(self.dtmf_preamble):
                if self.non_dtmf_buffer:
                    yield AgentResponse(content=self.non_dtmf_buffer)

                captured = self.dtmf_buffer.replace(self.dtmf_preamble, "", 1)
                yield DTMFOutputEvent(
                    button=captured,
                )

            else:
                # Otherwise, we didn't build enough in the dtmf buffer so move the buffer and then flush
                to_flush = self.non_dtmf_buffer + self.dtmf_buffer
                if to_flush:
                    yield AgentResponse(content=to_flush)

            # Next reset
            self.non_dtmf_buffer = ""
            self.dtmf_buffer = ""

            # Finally, recursively feed the character back in
            for item in self.feed(char):
                yield item
            return

        # New character is not in a word boundary, keep accumulating and checking
        if self.dtmf_buffer and is_word_char:
            self.dtmf_buffer += char

            return

        raise RuntimeError(f"Invalid state: {char=} {self.dtmf_buffer=} {self.non_dtmf_buffer=}")

    def flush(self) -> Generator[Union[AgentResponse, DTMFOutputEvent], None, None]:
        if self.dtmf_buffer.startswith(self.dtmf_preamble):
            captured = self.dtmf_buffer.replace(self.dtmf_preamble, "", 1)
            if self.non_dtmf_buffer:
                yield AgentResponse(content=self.non_dtmf_buffer)

            yield DTMFOutputEvent(button=captured)
        else:
            content = self.non_dtmf_buffer + self.dtmf_buffer
            if content:
                yield AgentResponse(content=content)

        # Cleanup
        self.non_dtmf_buffer = ""
        self.dtmf_buffer = ""
        return
