"""Keypad collection without telephony or LLM network dependencies."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocket
from loguru import logger
import pytest

from line.agent import AgentEnv, TurnEnv
from line.events import AgentSendText, UserDtmfSent, UserTextSent, UserTurnEnded, UserTurnStarted
from line.llm_agent import ToolEnv, dtmf_tool
from line.llm_agent.schema_converter import function_tool_to_litellm
from line.voice_agent_app import ConversationRunner


@dataclass
class Timer:
    when: float
    callback: Callable
    cancelled: bool = False

    def cancel(self):
        self.cancelled = True


class Clock:
    """Advance only the collector's clock; callback tasks use the real loop."""

    def __init__(self):
        self.now = 0.0
        self.timers = []

    def create_future(self):
        return asyncio.get_running_loop().create_future()

    def call_later(self, delay, callback):
        timer = Timer(self.now + delay, callback)
        self.timers.append(timer)
        return timer

    def advance(self, seconds):
        self.now += seconds
        for timer in sorted(self.timers, key=lambda t: t.when):
            if not timer.cancelled and timer.when <= self.now:
                timer.cancel()
                timer.callback()


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(
        "line._dtmf.asyncio",
        SimpleNamespace(get_running_loop=lambda: clock, Event=asyncio.Event),
    )
    return clock


def feed(env, buttons):
    for button in buttons:
        assert env._dtmf.handle_event(UserDtmfSent(button=button))


def make_tool(**options):
    backend = AsyncMock(return_value={"status": "verified"})

    @dtmf_tool(prompt="Enter your PIN, then press pound or wait.", **options)
    async def verify_pin(ctx: ToolEnv, digits: str):
        """Verify the caller's PIN."""
        return await backend(digits)

    return verify_pin, backend


async def start(tool, env=None):
    env = env or AgentEnv()
    stream = tool.func(ToolEnv(turn_env=TurnEnv(env)))
    prompt = await anext(stream)
    assert isinstance(prompt, AgentSendText)
    return env, stream


async def finish(stream):
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


def test_schema_has_no_model_supplied_digits():
    tool, _ = make_tool()
    schema = function_tool_to_litellm(tool)["function"]
    assert schema["name"] == "verify_pin"
    assert schema["parameters"]["properties"] == {}
    assert schema["parameters"].get("required", []) == []
    assert "Verify the caller's PIN" in schema["description"]


def test_schema_describes_collection_failures():
    tool, _ = make_tool()
    description = function_tool_to_litellm(tool)["function"]["description"]
    for status in ("no_input", "too_many_digits", "callback_timeout"):
        assert status in description


async def test_pound_captures_input_before_prompt_is_consumed(clock):
    tool, backend = make_tool()
    env, stream = await start(tool)
    feed(env, "0123#")
    clock.advance(30)
    feed(env, "99#")  # Already submitted; cannot alter the frozen entry.
    assert await anext(stream) == {"status": "verified"}
    backend.assert_awaited_once_with("0123")
    assert env._dtmf.active  # Result must reach the caller before releasing the guard.
    await finish(stream)
    assert not env._dtmf.active
    assert all(timer.cancelled for timer in clock.timers)


async def test_two_second_timeout_resets_after_each_digit(clock):
    tool, backend = make_tool()
    env, stream = await start(tool)
    result = asyncio.create_task(anext(stream))
    clock.advance(10)  # The inter-digit timeout does not apply before the first digit.
    assert not result.done()
    feed(env, "0")
    clock.advance(1.5)
    feed(env, "1")
    clock.advance(1.5)
    assert not result.done()
    clock.advance(0.5)
    assert await result == {"status": "verified"}
    backend.assert_awaited_once_with("01")
    await finish(stream)


@pytest.mark.parametrize("pound_first", [True, False])
async def test_timer_and_pound_submit_once(clock, pound_first):
    tool, backend = make_tool()
    env, stream = await start(tool)
    feed(env, "1234")
    if pound_first:
        feed(env, "#")
        clock.advance(2)
    else:
        clock.advance(2)
        feed(env, "#")
    assert await anext(stream) == {"status": "verified"}
    await finish(stream)
    backend.assert_awaited_once_with("1234")


@pytest.mark.parametrize("submit_empty", [True, False])
async def test_empty_input_skips_callback(clock, submit_empty):
    tool, backend = make_tool()
    env, stream = await start(tool)
    if submit_empty:
        feed(env, "#")
    else:
        clock.advance(15)
    assert await anext(stream) == {"status": "no_input"}
    await finish(stream)
    backend.assert_not_awaited()


async def test_overflow_does_not_submit_a_truncated_pin(clock):
    tool, backend = make_tool(max_digits=4)
    env, stream = await start(tool)
    feed(env, "12345#")
    assert await anext(stream) == {"status": "too_many_digits"}
    await finish(stream)
    backend.assert_not_awaited()


async def test_custom_terminator_and_invalid_buttons(clock):
    tool, backend = make_tool(finish_on_key="*")
    env, stream = await start(tool)
    feed(env, ["", "12", "x", "1", "#", "2", "*"])
    assert await anext(stream) == {"status": "verified"}
    await finish(stream)
    backend.assert_awaited_once_with("1#2")


async def test_calls_and_repeat_attempts_have_separate_buffers(clock):
    tool, backend = make_tool()
    env_a, stream_a = await start(tool)
    env_b, stream_b = await start(tool)
    feed(env_a, "12")
    feed(env_b, "34#")
    await anext(stream_b)
    await finish(stream_b)
    feed(env_a, "#")
    await anext(stream_a)
    await finish(stream_a)
    _, retry = await start(tool, env_a)
    feed(env_a, "56#")
    await anext(retry)
    await finish(retry)
    assert [call.args[0] for call in backend.await_args_list] == ["34", "12", "56"]


async def test_rejects_overlapping_collection_without_disturbing_first(clock):
    tool, backend = make_tool()
    env, stream = await start(tool)
    with pytest.raises(RuntimeError, match="already active"):
        await start(tool, env)
    feed(env, "1234#")
    await anext(stream)
    await finish(stream)
    backend.assert_awaited_once_with("1234")


@pytest.mark.parametrize("during_callback", [True, False])
async def test_cancellation_cleans_up_collection_and_callback(clock, during_callback):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @dtmf_tool(prompt="Enter your PIN.")
    async def verify_pin(ctx, digits):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    env, stream = await start(verify_pin)
    task = asyncio.create_task(anext(stream))
    if during_callback:
        feed(env, "1234#")
        await started.wait()
    else:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not env._dtmf.active
    assert all(timer.cancelled for timer in clock.timers)
    assert cancelled.is_set() == during_callback


async def test_callback_timeout_releases_guard(clock):
    cancelled = asyncio.Event()

    @dtmf_tool(prompt="Enter your PIN.", callback_timeout=0.01)
    async def verify_pin(ctx, digits):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    env, stream = await start(verify_pin)
    feed(env, "1234#")
    assert await anext(stream) == {"status": "callback_timeout"}
    await finish(stream)
    assert cancelled.is_set()
    assert not env._dtmf.active


async def test_overlapping_speech_finishes_before_resuming(clock):
    tool, _ = make_tool()
    env, stream = await start(tool)
    assert env._dtmf.handle_event(UserTurnStarted())
    assert env._dtmf.handle_event(UserTextSent(content="done"))
    feed(env, "1234#")
    await anext(stream)
    completing = asyncio.create_task(finish(stream))
    await asyncio.sleep(0)
    assert not completing.done()
    assert env._dtmf.handle_event(UserTurnEnded())
    # Another utterance can start before the waiting tool gets CPU time.
    assert env._dtmf.handle_event(UserTurnStarted())
    await asyncio.sleep(0)
    assert not completing.done()
    assert env._dtmf.handle_event(UserTurnEnded())
    await completing
    assert not env._dtmf.handle_event(UserTurnStarted())
    assert not env._dtmf.handle_event(UserTurnEnded())


@pytest.mark.parametrize(
    "options",
    [
        {"prompt": " "},
        {"finish_on_key": ""},
        {"finish_on_key": "##"},
        {"finish_on_key": "x"},
        {"inter_digit_timeout": 0},
        {"first_digit_timeout": -1},
        {"callback_timeout": float("inf")},
        {"inter_digit_timeout": float("nan")},
        {"max_digits": 0},
        {"max_digits": 2.5},
    ],
)
def test_invalid_config_fails_at_registration(options):
    with pytest.raises(ValueError):
        dtmf_tool(**{"prompt": "Enter your PIN", **options})


def test_invalid_callback_fails_at_registration():
    with pytest.raises(TypeError, match="async callback"):
        dtmf_tool(prompt="PIN")(lambda ctx, digits: digits)

    async def extra_parameter(ctx, digits, extra):
        return digits

    with pytest.raises(TypeError, match="async callback"):
        dtmf_tool(prompt="PIN")(extra_parameter)


async def test_callback_error_waits_for_speech_and_releases_guard(clock):
    failed = asyncio.Event()

    @dtmf_tool(prompt="Enter your PIN.")
    async def verify_pin(ctx, digits):
        failed.set()
        raise ValueError("Verification unavailable")

    env, stream = await start(verify_pin)
    assert env._dtmf.handle_event(UserTurnStarted())
    feed(env, "1234#")
    result = asyncio.create_task(anext(stream))
    await failed.wait()
    assert not result.done()
    assert env._dtmf.active
    assert env._dtmf.handle_event(UserTurnEnded())
    with pytest.raises(ValueError, match="Verification unavailable"):
        await result
    assert not env._dtmf.active


@pytest.mark.parametrize("phase", ["prompt", "collection", "callback"])
async def test_runner_cancellation_allows_retry_before_old_generator_closes(clock, phase):
    env = AgentEnv()
    prompt_sent = asyncio.Event()
    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    @dtmf_tool(prompt="Enter your PIN.")
    async def verify_pin(ctx: ToolEnv, digits: str):
        if digits == "11":
            callback_started.set()
            try:
                await asyncio.Future()
            finally:
                callback_cancelled.set()
        return {"status": "verified", "length": len(digits)}

    # Retain the old tool generator to exercise delayed finalization explicitly.
    old_stream = verify_pin.func(ToolEnv(TurnEnv(env)))
    retry = None

    async def agent(turn_env, event):
        async for output in old_stream:
            yield output

    async def send(_):
        prompt_sent.set()
        if phase == "prompt":
            await asyncio.Future()

    ws = MagicMock(spec=WebSocket)
    ws.send_json = send
    runner = ConversationRunner(ws, agent, env)
    try:
        await runner._handle_event(TurnEnv(env), UserTurnEnded())
        await asyncio.wait_for(prompt_sent.wait(), timeout=1)
        feed(env, "11#" if phase == "callback" else "1")
        if phase == "callback":
            await asyncio.wait_for(callback_started.wait(), timeout=1)
        assert env._dtmf.handle_event(UserTurnStarted())
        await runner._cancel_agent_task()
        assert not env._dtmf.active
        assert all(timer.cancelled for timer in clock.timers)
        assert callback_cancelled.is_set() == (phase == "callback")
        # Cancellation still consumes the old "done" turn, not the next turn.
        assert env._dtmf.handle_event(UserTextSent(content="done"))
        assert env._dtmf.handle_event(UserTurnEnded())
        assert not env._dtmf.handle_event(UserTurnStarted())
        assert not env._dtmf.handle_event(UserTurnEnded())

        _, retry = await start(verify_pin, env)
        await old_stream.aclose()
        assert env._dtmf.active
        feed(env, "22#")
        assert await anext(retry) == {"status": "verified", "length": 2}
        await finish(retry)
        assert not env._dtmf.active
    finally:
        await runner._cancel_agent_task()
        await old_stream.aclose()
        if retry is not None:
            await retry.aclose()


@pytest.mark.parametrize("collecting", [False, True])
async def test_dtmf_logs_omit_buttons_but_preserve_events(clock, collecting):
    tool, _ = make_tool()
    env = AgentEnv()
    stream = None
    if collecting:
        _, stream = await start(tool, env)
    runner = ConversationRunner(MagicMock(spec=WebSocket), AsyncMock(), env)
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        for button in "0123456789*#":
            event, runner.history = runner._process_input_event(runner.history, UserDtmfSent(button=button))
            assert event.button == button
            assert event.history[-1].button == button
        assert len(messages) == 12
        assert all(str(message).strip() == "-> 🧑🔔 User DTMF received" for message in messages)
    finally:
        logger.remove(sink)
        if stream is not None:
            await stream.aclose()
