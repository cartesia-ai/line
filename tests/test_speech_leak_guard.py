"""Tests for the speech-leak guard in the HTTPS Responses-API provider.

Some reasoning models (observed in production: ``gpt-5.4-mini``) sometimes
write a tool call's payload into the user-visible ``commentary`` message item
— as bare JSON arguments or as a whole spoken call expression
(``record_call_summary({...})``) — instead of, or in addition to, the proper
``function_call`` item. The delta sequences in these fixtures replay the
shapes captured from four production calls.

Two layers under test:

- ``_HttpResponseEventStream`` + ``SpeechLeakGuardConfig``: bounded-lookahead
  buffering, search-mode detection anywhere in the item, holdback flush on
  clean items, phase scoping, released-any tracking.
- ``_HttpResponsesProvider.chat()``: abort + retry-once with the corrective
  note / reasoning override applied to the retry request only, bridge text
  when the caller heard part of the failed attempt, fallback text when the
  retry leaks too, and history committed only for the accepted attempt.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
import pytest

from line.llm_agent.config import (
    DEFAULT_SPEECH_LEAK_FALLBACK_TEXT,
    LlmConfig,
    SpeechLeakGuardConfig,
    _normalize_config,
)
from line.llm_agent.http_responses_provider import (
    _HttpResponseEventStream,
    _HttpResponsesProvider,
    _SpeechLeakDetected,
)
from line.llm_agent.provider import Message, ParsedModelId, StreamChunk


def _run(coro):
    return asyncio.run(coro)


def _pydantify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return BaseLiteLLMOpenAIResponseObject(**{k: _pydantify(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_pydantify(x) for x in obj]
    return obj


async def _aiter(events: List[Dict[str, Any]]) -> AsyncIterator[Any]:
    for event in events:
        yield _pydantify(event)


async def _drive(events: List[Dict[str, Any]], guard: SpeechLeakGuardConfig) -> List[StreamChunk]:
    stream = _HttpResponseEventStream(_aiter(events), lambda response: None, speech_leak_guard=guard)
    chunks: List[StreamChunk] = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _msg_added(idx: int, phase: Optional[str], item_id: str) -> Dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "output_index": idx,
        "item": {"type": "message", "phase": phase, "id": item_id},
    }


def _delta(idx: int, item_id: str, text: str) -> Dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "output_index": idx,
        "item_id": item_id,
        "content_index": 0,
        "delta": text,
    }


def _msg_done(idx: int, phase: str, item_id: str, text: str) -> Dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "output_index": idx,
        "item": {
            "type": "message",
            "phase": phase,
            "id": item_id,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _completed(response_id: str = "resp_1") -> Dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {"id": response_id, "status": "completed", "output": []},
    }


def _message_events(
    deltas: List[str], *, idx: int = 0, phase: str = "commentary", item_id: str = "msg_0"
) -> List[Dict[str, Any]]:
    """A full clean single-message response from its delta sequence."""
    text = "".join(deltas)
    return [
        _msg_added(idx, phase, item_id),
        *[_delta(idx, item_id, d) for d in deltas],
        _msg_done(idx, phase, item_id, text),
        _completed(),
    ]


# Production leak, first observed shape:
# bare record_call_summary arguments streamed as 1-5 char commentary deltas
# on output_index=1 (a reasoning item held index 0).
_BARE_JSON_LEAK_DELTAS = [
    '{"',
    "summary",
    '":"',
    "The",
    " caller",
    " called",
    " about",
    " getting",
    " more",
    " information",
    '.","',
    "property",
    "_address",
    "_out",
    "come",
    '":"',
    "not",
    "_est",
    "ablished",
    '"}',
]

# Production leak, second observed shape:
# the model verbalized the whole call expression, tool name included.
_SPOKEN_CALL_LEAK_DELTAS = [
    "record",
    "_call",
    "_summary",
    "({\n",
    " ",
    ' "',
    "summary",
    '":"',
    "Caller",
    " called",
    " for",
    " more",
    " information",
    '."',
    " })",
]


def _leak_events(deltas: List[str], *, idx: int = 1, phase: str = "commentary") -> List[Dict[str, Any]]:
    """A response whose streamed message item is a leak. No done/completed
    events follow the deltas — detection must abort before the stream ends,
    exactly as the provider does in production (early abort)."""
    return [
        _msg_added(idx, phase, f"msg_{idx}"),
        *[_delta(idx, f"msg_{idx}", d) for d in deltas],
    ]


# ---------------------------------------------------------------------------
# Stream-level: detection
# ---------------------------------------------------------------------------


def test_bare_json_leak_detected_before_anything_released():
    """Call-1 shape: `{"summary":...}` as commentary. The very first delta
    matches the JSON-object alternative; nothing reaches TTS."""
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(_BARE_JSON_LEAK_DELTAS), guard)

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(drive())
    leak = exc_info.value
    assert leak.released_any is False
    assert leak.released_chars == 0
    assert leak.phase == "commentary"


def test_spoken_call_expression_leak_detected():
    """Call-2 shape: `record_call_summary({. "summary":...})` — the tool-name
    form, only detectable once `({`/`("` arrives. The identifier sits inside
    the holdback window until then, so nothing is released."""
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(_SPOKEN_CALL_LEAK_DELTAS), guard)

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(drive())
    assert exc_info.value.released_any is False


def test_pattern_straddling_delta_boundary_is_detected():
    """`{` and `"` arriving in separate deltas must still match — the scan
    runs over the accumulated item text, not per-delta."""
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(["{", '"summary": "x"}']), guard)

    with pytest.raises(_SpeechLeakDetected):
        _run(drive())


def test_leak_after_released_prefix_reports_released_any():
    """Tripwire: a natural prefix longer than the holdback window streams
    out, then the JSON starts. The leak is still caught (search mode), and
    released_any tells the provider a bridge line is needed."""
    guard = SpeechLeakGuardConfig(enabled=True, lookahead_chars=16)
    prefix = "Thank you for holding, I have noted everything down now. "
    events = _leak_events([prefix, '{"summary": "x"}'])

    released: List[str] = []

    async def drive():
        stream = _HttpResponseEventStream(_aiter(events), lambda response: None, speech_leak_guard=guard)
        async for chunk in stream:
            if chunk.text:
                released.append(chunk.text)

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(drive())
    leak = exc_info.value
    assert leak.released_any is True
    assert leak.released_chars > 0
    spoken = "".join(released)
    assert "{" not in spoken, f"leak content must never reach TTS; got {spoken!r}"
    assert prefix.startswith(spoken.split("{")[0][:8])


# ---------------------------------------------------------------------------
# Stream-level: clean traffic is unharmed
# ---------------------------------------------------------------------------


def test_clean_short_item_flushes_full_text_on_item_done():
    """An item shorter than the holdback window is held entirely, then
    flushed intact when the item completes. No text is lost."""
    guard = SpeechLeakGuardConfig(enabled=True)
    text = "Your code is on the way."
    chunks = _run(_drive(_message_events([text]), guard))
    assert "".join(c.text for c in chunks if c.text) == text
    assert chunks[-1].is_final is True


def test_clean_long_item_streams_with_holdback_and_loses_nothing():
    """A long clean item releases text as it accumulates (all but the last
    lookahead_chars), and the item-done flush delivers the tail."""
    guard = SpeechLeakGuardConfig(enabled=True, lookahead_chars=16)
    deltas = [
        "Sorry — I want to make sure I have this right. ",
        "Is your property address 35 North Green Bay Road, ",
        "Lake Forest, IL 60045?",
    ]
    events = _message_events(deltas)

    async def drive():
        stream = _HttpResponseEventStream(_aiter(events), lambda response: None, speech_leak_guard=guard)
        collected: List[StreamChunk] = []
        pre_done_text_len = 0
        seen_done = False
        async for chunk in stream:
            collected.append(chunk)
            if chunk.text and not seen_done:
                pre_done_text_len += len(chunk.text)
        return collected, pre_done_text_len

    chunks, _ = _run(drive())
    assert "".join(c.text for c in chunks if c.text) == "".join(deltas)
    # More than one text chunk proves streaming happened before the flush.
    assert len([c for c in chunks if c.text]) > 1


def test_email_style_readback_is_not_flagged():
    """`name(at)domain` spoken read-backs must not match the call-expression
    alternative (it requires `{` or `"` right after the paren)."""
    guard = SpeechLeakGuardConfig(enabled=True, lookahead_chars=8)
    text = "I have your email as john_doe(at)example.com. Is that right?"
    chunks = _run(_drive(_message_events([text]), guard))
    assert "".join(c.text for c in chunks if c.text) == text


def test_final_answer_phase_skipped_by_default():
    """final_answer is in the default skip set: a JSON-looking final_answer
    item streams through untouched (leaks have only been observed on
    commentary)."""
    guard = SpeechLeakGuardConfig(enabled=True)
    text = '{"summary": "x"}'
    chunks = _run(_drive(_message_events([text], phase="final_answer"), guard))
    assert "".join(c.text for c in chunks if c.text) == text


def test_final_answer_guarded_with_empty_skip_set():
    """skip_phases=frozenset() guards every spoken item, final_answer included."""
    guard = SpeechLeakGuardConfig(enabled=True, skip_phases=frozenset())

    async def drive():
        return await _drive(_leak_events(['{"a": 1}'], phase="final_answer"), guard)

    with pytest.raises(_SpeechLeakDetected):
        _run(drive())


# ---------------------------------------------------------------------------
# Stream-level: unlabeled / unknown phases are guarded (fail closed)
# ---------------------------------------------------------------------------


def _msg_added_no_phase(idx: int, item_id: str) -> Dict[str, Any]:
    """An output_item.added whose message item carries NO phase attribute —
    the wire shape of models predating the phase field (gpt-5-mini, gpt-4.1)."""
    return {
        "type": "response.output_item.added",
        "output_index": idx,
        "item": {"type": "message", "id": item_id},
    }


def test_leak_in_item_without_phase_attribute_is_guarded():
    """Models that never send ``phase`` must still be protected: absence of a
    label is not evidence of safety."""
    guard = SpeechLeakGuardConfig(enabled=True)
    events = [
        _msg_added_no_phase(0, "msg_0"),
        _delta(0, "msg_0", '{"summary": "x"}'),
    ]

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(_drive(events, guard))
    assert exc_info.value.phase is None


def test_leak_in_item_with_none_phase_is_guarded():
    """litellm's typed objects may materialize an absent wire field as an
    explicit ``phase=None`` — must be treated the same as attr-missing."""
    guard = SpeechLeakGuardConfig(enabled=True)
    events = [
        _msg_added(0, None, "msg_0"),
        _delta(0, "msg_0", '{"summary": "x"}'),
    ]

    with pytest.raises(_SpeechLeakDetected):
        _run(_drive(events, guard))


def test_leak_in_unregistered_item_is_guarded():
    """Deltas arriving for an output_index with no output_item.added at all
    (degraded stream) must be guarded, not silently exempted."""
    guard = SpeechLeakGuardConfig(enabled=True)
    events = [_delta(2, "msg_2", '{"summary": "x"}')]

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(_drive(events, guard))
    assert exc_info.value.phase is None


def test_leak_in_unrecognized_phase_label_is_guarded():
    """A future/unknown phase label is not in the skip set → guarded."""
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(['{"a": 1}'], phase="plan"), guard)

    with pytest.raises(_SpeechLeakDetected):
        _run(drive())


def test_clean_unlabeled_item_streams_in_full():
    """The guard on unlabeled items must not cost any text: a clean item from
    a phase-less model is held back, then flushed intact."""
    guard = SpeechLeakGuardConfig(enabled=True)
    text = "Your appointment is confirmed for Tuesday at three."
    events = [
        _msg_added_no_phase(0, "msg_0"),
        _delta(0, "msg_0", text),
        _msg_done(0, "commentary", "msg_0", text),
        _completed(),
    ]
    chunks = _run(_drive(events, guard))
    assert "".join(c.text for c in chunks if c.text) == text
    assert chunks[-1].is_final is True


def test_disabled_guard_streams_leak_verbatim():
    """Regression pin for default-off behavior: without the guard the leak
    streams to TTS exactly as before (this is the production bug)."""
    guard = SpeechLeakGuardConfig(enabled=False)
    events = _leak_events(_BARE_JSON_LEAK_DELTAS) + [
        _msg_done(1, "commentary", "msg_1", "".join(_BARE_JSON_LEAK_DELTAS)),
        _completed(),
    ]
    chunks = _run(_drive(events, guard))
    assert "".join(c.text for c in chunks if c.text) == "".join(_BARE_JSON_LEAK_DELTAS)


# ---------------------------------------------------------------------------
# Provider-level: retry / bridge / fallback via chat()
# ---------------------------------------------------------------------------

_CLEAN_TEXT = "Let me get you to one of our team members now."


def _clean_response_events(response_id: str = "resp_clean") -> List[Dict[str, Any]]:
    return [
        _msg_added(0, "commentary", "msg_c"),
        _delta(0, "msg_c", _CLEAN_TEXT),
        _msg_done(0, "commentary", "msg_c", _CLEAN_TEXT),
        {
            "type": "response.completed",
            "response": {"id": response_id, "status": "completed", "output": []},
        },
    ]


class _FakeAresponses:
    """Stands in for litellm.aresponses: returns scripted event streams and
    records each request's kwargs for assertions."""

    def __init__(self, scripts: List[List[Dict[str, Any]]]):
        self._scripts = list(scripts)
        self.requests: List[Dict[str, Any]] = []

    async def __call__(self, **request_kwargs: Any) -> AsyncIterator[Any]:
        self.requests.append(request_kwargs)
        if not self._scripts:
            raise AssertionError("aresponses called more times than scripted")
        return _aiter(self._scripts.pop(0))


def _make_provider_and_config(guard: SpeechLeakGuardConfig):
    # zdr_enabled=False so history commits are observable: the ZDR planner's
    # per-turn history update is deliberately a no-op (full input each turn),
    # which would make "was this attempt committed?" unobservable.
    provider = _HttpResponsesProvider(ParsedModelId("openai", "gpt-5.4-mini"), api_key="test")
    config = _normalize_config(
        LlmConfig(
            system_prompt="You are a test agent.",
            zdr_enabled=False,
            speech_leak_guard=guard,
        )
    )
    return provider, config


async def _collect_chat(provider, config) -> List[StreamChunk]:
    chunks: List[StreamChunk] = []
    async with provider.chat([Message(role="user", content="hello")], config=config) as stream:
        async for chunk in stream:
            chunks.append(chunk)
    return chunks


def test_chat_retries_once_and_streams_clean_second_attempt(monkeypatch):
    guard = SpeechLeakGuardConfig(enabled=True, retry_reasoning_effort="medium")
    fake = _FakeAresponses([_leak_events(_BARE_JSON_LEAK_DELTAS), _clean_response_events()])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    chunks = _run(_collect_chat(provider, config))

    assert "".join(c.text for c in chunks if c.text) == _CLEAN_TEXT
    assert chunks[-1].is_final is True
    assert len(fake.requests) == 2

    # The retry request — and only the retry request — carries the
    # corrective developer note and the reasoning-effort override.
    first, second = fake.requests
    assert not any(item.get("role") == "developer" for item in first.get("input", []))
    dev_items = [item for item in second["input"] if item.get("role") == "developer"]
    assert len(dev_items) == 1
    assert "discarded" in dev_items[0]["content"][0]["text"]
    assert second["reasoning"] == {"effort": "medium"}

    # Only the accepted attempt was committed to history (non-ZDR mode, so
    # a successful commit is visible as populated history).
    assert len(provider._history) > 0


def test_chat_double_leak_speaks_fallback_and_ends_turn(monkeypatch):
    guard = SpeechLeakGuardConfig(enabled=True)
    fake = _FakeAresponses([_leak_events(_BARE_JSON_LEAK_DELTAS), _leak_events(_SPOKEN_CALL_LEAK_DELTAS)])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    chunks = _run(_collect_chat(provider, config))

    texts = [c.text for c in chunks if c.text]
    assert texts == [guard.fallback_text]
    assert chunks[-1].is_final is True
    assert chunks[-1].tool_calls == []
    assert len(fake.requests) == 2  # initial + max_retries(1), then gave up
    assert provider._history == []  # neither leaked attempt was committed


def test_chat_speaks_bridge_when_caller_heard_part_of_failed_attempt(monkeypatch):
    guard = SpeechLeakGuardConfig(enabled=True, lookahead_chars=16)
    prefix = "Thank you for holding, I have everything I need from you now. "
    leak_attempt = _leak_events([prefix, '{"summary": "x"}'])
    fake = _FakeAresponses([leak_attempt, _clean_response_events()])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    chunks = _run(_collect_chat(provider, config))

    texts = [c.text for c in chunks if c.text]
    assert guard.bridge_text in texts
    bridge_pos = texts.index(guard.bridge_text)
    # Some prefix speech precedes the bridge; the clean reply follows it.
    assert bridge_pos > 0
    assert "".join(texts[bridge_pos + 1 :]) == _CLEAN_TEXT


def test_chat_retry_note_via_instructions_channel(monkeypatch):
    guard = SpeechLeakGuardConfig(enabled=True, retry_note_channel="instructions")
    fake = _FakeAresponses([_leak_events(_BARE_JSON_LEAK_DELTAS), _clean_response_events()])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    _run(_collect_chat(provider, config))

    first, second = fake.requests
    assert guard.retry_note not in (first.get("instructions") or "")
    assert (second.get("instructions") or "").endswith(guard.retry_note)
    assert not any(item.get("role") == "developer" for item in second.get("input", []))


def test_chat_without_guard_streams_leak_as_before(monkeypatch):
    """No guard configured → single request, leak streams to TTS (pins the
    pre-guard behavior so enabling is a strict opt-in)."""
    leak_text = "".join(_BARE_JSON_LEAK_DELTAS)
    events = _leak_events(_BARE_JSON_LEAK_DELTAS) + [
        _msg_done(1, "commentary", "msg_1", leak_text),
        _completed(),
    ]
    fake = _FakeAresponses([events])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider = _HttpResponsesProvider(ParsedModelId("openai", "gpt-5.4-mini"), api_key="test")
    config = _normalize_config(LlmConfig(system_prompt="x", zdr_enabled=True))

    chunks = _run(_collect_chat(provider, config))

    assert "".join(c.text for c in chunks if c.text) == leak_text
    assert len(fake.requests) == 1


def test_chat_callable_texts_resolved_at_speak_time(monkeypatch):
    """bridge_text / fallback_text may be zero-arg callables (multilingual
    agents pick the line by current language at speak time, not config time)."""
    lines_spoken: List[str] = []

    def bridge() -> str:
        lines_spoken.append("bridge")
        return "Un instant, je reprends."

    def fallback() -> str:
        lines_spoken.append("fallback")
        return "Désolé, un souci technique. Voulez-vous que je réessaie ?"

    guard = SpeechLeakGuardConfig(
        enabled=True, lookahead_chars=16, bridge_text=bridge, fallback_text=fallback
    )
    prefix = "Merci de patienter, j'ai tout ce qu'il me faut. "
    fake = _FakeAresponses(
        [
            _leak_events([prefix, '{"summary": "x"}']),  # heard prefix → bridge
            _leak_events(_BARE_JSON_LEAK_DELTAS),  # leaks again → fallback
        ]
    )
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    chunks = _run(_collect_chat(provider, config))

    texts = [c.text for c in chunks if c.text]
    assert "Un instant, je reprends." in texts
    assert texts[-1] == "Désolé, un souci technique. Voulez-vous que je réessaie ?"
    assert lines_spoken == ["bridge", "fallback"]  # each invoked exactly once, on demand


def test_chat_callable_text_returning_empty_speaks_nothing(monkeypatch):
    guard = SpeechLeakGuardConfig(enabled=True, fallback_text=lambda: "")
    fake = _FakeAresponses([_leak_events(_BARE_JSON_LEAK_DELTAS), _leak_events(_BARE_JSON_LEAK_DELTAS)])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    chunks = _run(_collect_chat(provider, config))

    assert [c.text for c in chunks if c.text] == []
    assert chunks[-1].is_final is True


def test_chat_crashing_text_callable_falls_back_to_default(monkeypatch):
    """A raising callable must not take down leak recovery: the English
    default line is spoken and the turn still ends cleanly."""

    def boom() -> str:
        raise RuntimeError("translation service down")

    guard = SpeechLeakGuardConfig(enabled=True, max_retries=0, fallback_text=boom)
    fake = _FakeAresponses([_leak_events(_BARE_JSON_LEAK_DELTAS)])
    monkeypatch.setattr("line.llm_agent.http_responses_provider.aresponses", fake)
    provider, config = _make_provider_and_config(guard)

    chunks = _run(_collect_chat(provider, config))

    texts = [c.text for c in chunks if c.text]
    assert texts == [DEFAULT_SPEECH_LEAK_FALLBACK_TEXT]
    assert chunks[-1].is_final is True


# ---------------------------------------------------------------------------
# Degraded harmony-header leak shapes (no paren, no leading brace) — observed
# in production calls and replay sampling: the tool-call header corrupts into
# visible text as tool name + glitch-token run + `json {…`, sometimes with
# `to=functions.…` / `functions.…` markers.
# ---------------------------------------------------------------------------

_GLITCH_HEADER_LEAK_DELTAS = [
    "record",
    "_call",
    "_summary",
    "  彩",
    "神争",
    "霸大发",
    "json",
    ' {"',
    "summary",
    '":"',
    "Caller",
    " asked",
    " for",
    " a",
    " repeat",
    '."}',
]

_TO_FUNCTIONS_LEAK_DELTAS = [
    "record",
    "_call",
    "_summary",
    " to",
    "=functions",
    ".record",
    "_call",
    "_summary",
    "  天天中",
    "彩票出票",
    "json",
    ' {"summary":"x"}',
]


def test_glitch_header_leak_detected_with_nothing_released():
    """Tool name + glitch tokens + `json {"` — no paren, no leading brace.
    Detection lands on the JSON-object alternative at offset ~30; the default
    128-char holdback must keep every character inaudible (with the previous
    64-char default and a longer glitch run, 3 chars escaped in production)."""
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(_GLITCH_HEADER_LEAK_DELTAS), guard)

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(drive())
    leak = exc_info.value
    assert leak.released_any is False
    assert leak.released_chars == 0


def test_to_functions_marker_detected_before_json_arrives():
    """The `to=functions.` recipient marker is pure protocol syntax and fires
    on its own — detection must not have to wait for the `{"` payload."""
    guard = SpeechLeakGuardConfig(enabled=True)
    # Only the deltas up to and including the marker (`to=functions.`);
    # no JSON ever arrives.
    deltas = _TO_FUNCTIONS_LEAK_DELTAS[:6]

    async def drive():
        return await _drive(_leak_events(deltas), guard)

    with pytest.raises(_SpeechLeakDetected) as exc_info:
        _run(drive())
    assert exc_info.value.released_chars == 0


def test_functions_namespace_prefix_detected():
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(["functions", ".record_call", "_summary  xyz"]), guard)

    with pytest.raises(_SpeechLeakDetected):
        _run(drive())


def test_harmony_special_token_delimiter_detected():
    guard = SpeechLeakGuardConfig(enabled=True)

    async def drive():
        return await _drive(_leak_events(["<|", "constrain|>json"]), guard)

    with pytest.raises(_SpeechLeakDetected):
        _run(drive())


def test_prose_containing_functions_word_streams_clean():
    """`functions` as an ordinary English word (followed by space or sentence
    punctuation) must not trip the namespace-prefix alternative, and text
    longer than the 128-char holdback still streams and flushes losslessly."""
    guard = SpeechLeakGuardConfig(enabled=True)
    text = (
        "Everything functions normally on your account. This tool functions. "
        "Also, I want to make sure I have this right — is your property "
        "address 35 North Green Bay Road, Lake Forest, IL 60045?"
    )
    chunks = _run(_drive(_message_events([text[:80], text[80:]]), guard))
    assert "".join(c.text for c in chunks if c.text) == text
