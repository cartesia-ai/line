"""Tests for the HTTPS Responses-API provider.

The most important behavior under test is the `phase: "commentary"`
suppression that fixes duplicated TTS output on gpt-5.4+ reasoning
models.  The Responses API can emit two `message` items in a single
response — one preamble (`commentary`) and one final answer
(`final_answer`) — and when their text is similar/identical the
chat-completions bridge concatenates both deltas onto `choices[0]`,
producing duplicated TTS audio.  The HTTPS Responses provider here
filters commentary items at the event-stream layer.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject

from line.llm_agent.http_responses_provider import _HttpResponseEventStream
from line.llm_agent.provider import StreamChunk


def _run(coro):
    return asyncio.run(coro)


def _pydantify(obj: Any) -> Any:
    """Recursively wrap dicts as BaseLiteLLMOpenAIResponseObject so the
    test fixtures behave like real LiteLLM aresponses events (pydantic
    objects with extra="allow"), not plain Python dicts."""
    if isinstance(obj, dict):
        return BaseLiteLLMOpenAIResponseObject(**{k: _pydantify(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_pydantify(x) for x in obj]
    return obj


async def _aiter(events: List[Dict[str, Any]]) -> AsyncIterator[Any]:
    for event in events:
        yield _pydantify(event)


async def _drive(events: List[Dict[str, Any]]) -> List[StreamChunk]:
    captured_response: Dict[str, Any] = {}

    def on_done(response: Dict[str, Any]) -> None:
        captured_response.update(response)

    stream = _HttpResponseEventStream(_aiter(events), on_done)
    chunks: List[StreamChunk] = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


def test_commentary_message_is_suppressed_final_answer_passes_through():
    """The duplication-fix regression test.

    Two message items in one response: commentary (output_index=0) and
    final_answer (output_index=1) with the same text.  Only the
    final_answer's deltas should reach the caller.
    """
    events = [
        # Commentary message item starts at output_index 0.
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "phase": "commentary", "id": "msg_0"},
        },
        # Streamed deltas for the commentary item — should be suppressed.
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "msg_0",
            "content_index": 0,
            "delta": "Hello there!",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "phase": "commentary",
                "id": "msg_0",
                "content": [{"type": "output_text", "text": "Hello there!"}],
            },
        },
        # Final-answer message item at output_index 1.
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "message", "phase": "final_answer", "id": "msg_1"},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 1,
            "item_id": "msg_1",
            "content_index": 0,
            "delta": "Hello there!",
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "message",
                "phase": "final_answer",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Hello there!"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_abc",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "Hello there!"}],
                    },
                    {
                        "type": "message",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "Hello there!"}],
                    },
                ],
            },
        },
    ]

    chunks = _run(_drive(events))
    text_chunks = [c.text for c in chunks if c.text is not None]
    assert text_chunks == ["Hello there!"], (
        f"expected exactly one 'Hello there!' (commentary suppressed); got {text_chunks!r}"
    )
    assert chunks[-1].is_final is True


def test_text_passes_through_when_phase_is_missing():
    """Older Responses-API responses without `phase` should not be filtered."""
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_0"},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "msg_0",
            "content_index": 0,
            "delta": "Hi.",
        },
        {
            "type": "response.completed",
            "response": {"id": "resp_x", "status": "completed", "output": []},
        },
    ]
    chunks = _run(_drive(events))
    assert [c.text for c in chunks if c.text] == ["Hi."]


def test_tool_call_arguments_accumulate_via_item_id_and_resolve_to_call_id():
    """function_call_arguments.delta uses `item_id`; output_item.done
    provides the canonical `call_id` and full `arguments`."""
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_42",
                "name": "get_weather",
                "id": "fn_item_1",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fn_item_1",
            "delta": '{"city":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fn_item_1",
            "delta": '"SF"}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_42",
                "name": "get_weather",
                "id": "fn_item_1",
                "arguments": '{"city":"SF"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_t",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_42",
                        "name": "get_weather",
                        "arguments": '{"city":"SF"}',
                    }
                ],
            },
        },
    ]
    chunks = _run(_drive(events))
    final = chunks[-1]
    assert final.is_final
    assert len(final.tool_calls) == 1
    tc = final.tool_calls[0]
    assert tc.id == "call_42"
    assert tc.name == "get_weather"
    assert tc.arguments == '{"city":"SF"}'
    assert tc.is_complete


def test_failed_response_raises_runtime_error():
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_0"},
        },
        {
            "type": "response.failed",
            "response": {
                "id": "resp_f",
                "status": "failed",
                "error": {"code": "server_error", "message": "boom"},
            },
        },
    ]
    try:
        _run(_drive(events))
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
