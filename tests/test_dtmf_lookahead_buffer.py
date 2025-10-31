from line.events import AgentResponse, DTMFOutputEvent
from line.utils.dtmf_lookahead_buffer import (
    DTMFLookAheadCharacterBuffer,
    DTMFLookAheadStringBuffer,
)


def test_dtmf_char_buffer_base_case():
    actual = []
    buffer = DTMFLookAheadCharacterBuffer()

    actual.extend(list(buffer.feed("h")))
    actual.extend(list(buffer.feed("e")))
    actual.extend(list(buffer.feed("l")))
    actual.extend(list(buffer.feed("l")))
    actual.extend(list(buffer.feed("o")))

    assert actual == []

    actual = list(buffer.flush())
    assert actual == [AgentResponse(content="hello")]


def test_dtmf_char_buffer_dtmf_case():
    actual = []
    buffer = DTMFLookAheadCharacterBuffer()

    actual.extend(list(buffer.feed("d")))
    assert actual == []
    actual.extend(list(buffer.feed("t")))
    assert actual == []
    actual.extend(list(buffer.feed("m")))
    assert actual == []
    actual.extend(list(buffer.feed("f")))
    assert actual == []
    actual.extend(list(buffer.feed("=")))

    assert actual == []

    actual.extend(list(buffer.feed("1")))
    assert actual == []

    actual.extend(list(buffer.feed("2")))
    assert actual == []

    actual.extend(list(buffer.feed(" ")))
    assert actual == [DTMFOutputEvent(button="12")]

    actual = list(buffer.flush())
    assert actual == [AgentResponse(content=" ")]


def test_dtmf_char_buffer_mixed_case():
    """
    'hello dtmf=123 dtmf=3world'
    """
    actual = []
    buffer = DTMFLookAheadCharacterBuffer()

    actual.extend(list(buffer.feed("h")))
    actual.extend(list(buffer.feed("e")))
    actual.extend(list(buffer.feed("l")))
    actual.extend(list(buffer.feed("l")))
    actual.extend(list(buffer.feed("o")))
    actual.extend(list(buffer.feed(" ")))
    actual.extend(list(buffer.feed("d")))
    actual.extend(list(buffer.feed("t")))
    actual.extend(list(buffer.feed("m")))
    actual.extend(list(buffer.feed("f")))
    actual.extend(list(buffer.feed("=")))
    actual.extend(list(buffer.feed("1")))
    actual.extend(list(buffer.feed("2")))
    actual.extend(list(buffer.feed("3")))

    assert actual == []

    actual.extend(list(buffer.feed(" ")))
    assert actual == [AgentResponse(content="hello "), DTMFOutputEvent(button="123")]

    actual.extend(list(buffer.feed("d")))
    actual.extend(list(buffer.feed("t")))
    actual.extend(list(buffer.feed("m")))
    actual.extend(list(buffer.feed("f")))
    actual.extend(list(buffer.feed("=")))
    actual.extend(list(buffer.feed("3")))
    assert actual == [AgentResponse(content="hello "), DTMFOutputEvent(button="123")]

    actual.extend(list(buffer.feed(" ")))
    assert actual == [
        AgentResponse(content="hello "),
        DTMFOutputEvent(button="123"),
        AgentResponse(content=" "),
        DTMFOutputEvent(button="3"),
    ]

    actual.extend(list(buffer.feed("w")))
    actual.extend(list(buffer.feed("o")))
    actual.extend(list(buffer.feed("r")))
    actual.extend(list(buffer.feed("l")))
    actual.extend(list(buffer.feed("d")))
    actual.extend(list(buffer.flush()))

    assert actual == [
        AgentResponse(content="hello "),
        DTMFOutputEvent(button="123"),
        AgentResponse(content=" "),
        DTMFOutputEvent(button="3"),
        AgentResponse(content=" world"),
    ]


def test_dtmf_string_buffer_base_case():
    actual = []
    buffer = DTMFLookAheadStringBuffer()

    actual.extend(list(buffer.feed("hello")))
    assert actual == []

    actual.extend(list(buffer.flush()))
    assert actual == [AgentResponse(content="hello")]


def test_dtmf_string_buffer_dtmf_case():
    actual = []
    buffer = DTMFLookAheadStringBuffer()

    actual.extend(list(buffer.feed("dtmf=3")))
    assert actual == []

    actual.extend(list(buffer.flush()))
    assert actual == [DTMFOutputEvent(button="3")]


def test_dtmf_string_buffer_mixed_case():
    actual = []
    buffer = DTMFLookAheadStringBuffer()

    actual.extend(list(buffer.feed("hello ")))
    actual.extend(list(buffer.feed("dtm")))
    actual.extend(list(buffer.feed("f=123 dtmf=3 w")))

    assert actual == [
        AgentResponse(content="hello "),
        DTMFOutputEvent(button="1"),
        DTMFOutputEvent(button="2"),
        DTMFOutputEvent(button="3"),
        AgentResponse(content=" "),
        DTMFOutputEvent(button="3"),
    ]
    actual.extend(list(buffer.feed("orld")))
    actual.extend(list(buffer.flush()))
    assert actual == [
        AgentResponse(content="hello "),
        DTMFOutputEvent(button="1"),
        DTMFOutputEvent(button="2"),
        DTMFOutputEvent(button="3"),
        AgentResponse(content=" "),
        DTMFOutputEvent(button="3"),
        AgentResponse(content=" world"),
    ]


def test_dtmf_string_buffer_mixed_case_2():
    actual = []
    buffer = DTMFLookAheadStringBuffer()

    actual.extend(list(buffer.feed("Hi")))
    actual.extend(list(buffer.feed(", this is Angela calling Andrew Clement?")))
    actual.extend(list(buffer.flush()))
    assert [
        AgentResponse(content="Hi, this is Angela calling Andrew"),
        AgentResponse(content=" Clement?"),
    ] == actual
