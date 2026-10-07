"""Eager inference never runs agent code until a correlated, confirmed final."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from line._harness_types import (
    TranscriptionInput,
    UserStateInput,
    UserTurnEagerInput,
    UserTurnResumedInput,
)
from line.agent import AgentEnv
from line.events import CallStarted, UserTurnStarted
from line.llm_agent import LlmAgent, LlmConfig
from line.llm_agent.provider import StreamChunk, ToolCall
from line.llm_agent.tools.utils import FunctionTool
from line.voice_agent_app import ConversationRunner, VoiceAgentApp


class Provider:
    supports_eager_endpointing = True

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.started = asyncio.Event()
        self.hold: asyncio.Event | None = None
        self.closed = asyncio.Event()

    def chat(self, messages, tools=None, config=None, **kwargs):
        async def stream():
            index = len(self.calls)
            self.calls.append(deepcopy(messages))
            self.started.set()
            try:
                response = self.responses[index]
                if isinstance(response, Exception):
                    raise response
                for chunk in response:
                    yield chunk
                if self.hold is not None:
                    await self.hold.wait()
            finally:
                self.closed.set()

        return stream()

    async def aclose(self):
        pass


def make_runner(*responses, tools=None, enabled=True):
    agent = LlmAgent("openai/gpt-4o", "test", tools=tools, config=LlmConfig())
    provider = Provider(*responses)
    agent._llm = provider
    socket = AsyncMock()
    runner = ConversationRunner(socket, agent, AgentEnv(), eager_endpointing=enabled)
    runner.history = [CallStarted(event_id="start"), UserTurnStarted(event_id="speaking")]
    return runner, agent, provider, socket


async def eager(runner, provider, text="hello", event_id="eager-1"):
    await runner._handle_input_message(UserTurnEagerInput(content=text, event_id=event_id))
    if runner._prepared is not None:
        await provider.started.wait()
        await asyncio.sleep(0)


async def final(runner, text="hello", eager_id="eager-1", reason="confirmed"):
    await runner._handle_input_message(TranscriptionInput(content=text, event_id="transcript"))
    await runner._handle_input_message(
        UserStateInput(
            value="idle",
            event_id="final",
            eager_event_id=eager_id,
            turn_end_reason=reason,
        )
    )
    if runner.agent_task:
        await runner.agent_task
    await runner._close_eager_tasks()


def spoken(socket):
    return [call.args[0] for call in socket.send_json.call_args_list if call.args[0]["type"] == "message"]


@pytest.mark.asyncio
async def test_reuses_completion_without_mutating_history_before_final():
    runner, agent, provider, socket = make_runner([StreamChunk(text="Hello!")])
    await eager(runner, provider)
    assert not socket.send_json.called
    assert list(agent.history) == []
    assert agent._user_turns_seen == 0
    assert runner.emitted_agent_text == []
    assert all(event.event_id != "eager-1" for event in runner.history)
    await final(runner)
    assert len(provider.calls) == 1
    assert spoken(socket)[0]["content"] == "Hello!"
    assert spoken(socket)[0]["responding_to"] == "final"
    assert socket.send_json.call_args.args[0] == {"type": "response_end", "responding_to": "final"}
    assert agent._user_turns_seen == 1
    assert [event.event_id for event in runner.history][-2:] == ["transcript", "final"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,eager_id,reason",
    [
        ("changed input", "eager-1", "confirmed"),
        ("hello", "another-attempt", "confirmed"),
        ("hello", None, "confirmed"),
        ("hello", "eager-1", None),
    ],
)
async def test_only_correlated_unchanged_confirmed_input_is_reused(text, eager_id, reason):
    runner, _, provider, socket = make_runner([StreamChunk(text="discarded")], [StreamChunk(text="fresh")])
    await eager(runner, provider)
    await final(runner, text, eager_id, reason)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]
    assert len(provider.calls) == 2
    assert all(message.content != "discarded" for message in provider.calls[1])


@pytest.mark.asyncio
async def test_resume_discards_completed_candidate_and_preserves_committed_history():
    runner, agent, provider, socket = make_runner(
        [StreamChunk(text="discarded")], [StreamChunk(text="fresh")]
    )
    await eager(runner, provider)
    await runner._handle_input_message(UserTurnResumedInput(eager_event_id="eager-1"))
    assert runner._prepared is None
    assert list(agent.history) == []
    assert not socket.send_json.called
    await final(runner)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]


@pytest.mark.asyncio
async def test_stale_resume_cannot_cancel_new_attempt():
    runner, _, provider, socket = make_runner([StreamChunk(text="first")], [StreamChunk(text="second")])
    await eager(runner, provider)
    await eager(runner, provider, event_id="eager-2")
    await runner._handle_input_message(UserTurnResumedInput(eager_event_id="eager-1"))
    await final(runner, eager_id="eager-2")
    assert [message["content"] for message in spoken(socket)] == ["second"]


@pytest.mark.asyncio
async def test_teardown_discards_inference_and_ignores_following_forced_idle():
    runner, agent, provider, socket = make_runner([StreamChunk(text="must not speak")])
    provider.hold = asyncio.Event()
    await eager(runner, provider)
    await runner._handle_input_message(UserStateInput(value="idle", turn_end_reason="teardown"))
    await final(runner)
    assert not socket.send_json.called
    assert agent._user_turns_seen == 0
    assert provider.closed.is_set()


@pytest.mark.asyncio
async def test_failed_eager_completion_falls_back_without_closing_call():
    runner, _, provider, socket = make_runner(
        RuntimeError("speculative failure"), [StreamChunk(text="fresh")]
    )
    await eager(runner, provider)
    await final(runner)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]
    assert not socket.close.called


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_customer_tools_run_only_after_final(background):
    executed = []

    async def tool(ctx):
        executed.append(True)
        return "done"

    function = FunctionTool("write", "Writes state", tool, {}, is_background=background)
    runner, _, provider, socket = make_runner(
        [StreamChunk(tool_calls=[ToolCall(id="tool-1", name="write", arguments="{}", is_complete=True)])],
        [StreamChunk(text="done")],
        tools=[function],
    )
    await eager(runner, provider)
    assert executed == []
    assert not socket.send_json.called
    await final(runner)
    assert executed == [True]
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_config_change_invalidates_candidate():
    runner, agent, provider, socket = make_runner([StreamChunk(text="old")], [StreamChunk(text="fresh")])
    await eager(runner, provider)
    agent.set_config(LlmConfig(system_prompt="Different instructions"))
    await final(runner)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]


@pytest.mark.asyncio
async def test_tool_window_matches_the_confirmed_turn():
    function = FunctionTool("limited", "Limited tool", lambda ctx: None, {}, active_turns=1)
    runner, agent, provider, _ = make_runner([StreamChunk(text="reply")], tools=[function])
    agent._user_turns_seen = 1
    await eager(runner, provider)
    await final(runner)
    assert len(provider.calls) == 1
    assert agent._user_turns_seen == 2


@pytest.mark.asyncio
async def test_default_off_ignores_new_events_even_with_broad_filter():
    calls = []

    async def custom(env, event):
        calls.append(event)
        if False:
            yield

    runner = ConversationRunner(AsyncMock(), (custom, lambda event: True, lambda event: False), AgentEnv())
    await runner._handle_input_message(UserTurnEagerInput(content="hello", event_id="eager"))
    assert not calls
    assert runner.history == []


@pytest.mark.asyncio
async def test_wrapper_falls_back_even_when_app_opts_in():
    calls = []

    async def wrapper(env, event):
        calls.append(event.type)
        if False:
            yield

    runner = ConversationRunner(AsyncMock(), wrapper, AgentEnv(), eager_endpointing=True)
    runner.history = [UserTurnStarted()]
    await runner._handle_input_message(UserTurnEagerInput(content="hello", event_id="eager"))
    await final(runner)
    assert calls == ["user_turn_ended"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_chat_capability_requires_explicit_opt_in(enabled):
    app = VoiceAgentApp(AsyncMock(), eager_endpointing=enabled)
    request = AsyncMock()
    request.json.return_value = {}
    result = await app.create_chat_session(request)
    assert ("eager_endpointing_v1" in result.get("capabilities", [])) == enabled


@pytest.mark.asyncio
async def test_unsupported_provider_and_active_handoff_skip_prefetch():
    runner, agent, provider, _ = make_runner([StreamChunk(text="reply")])
    provider.supports_eager_endpointing = False
    await eager(runner, provider)
    assert runner._prepared is None
    provider.supports_eager_endpointing = True
    agent._handoff_target = AsyncMock()
    await eager(runner, provider)
    assert runner._prepared is None
    assert provider.calls == []


@pytest.mark.asyncio
async def test_live_candidate_streams_buffered_then_new_chunks_in_order():
    runner, _, provider, socket = make_runner([StreamChunk(text="buffered")])
    provider.hold = asyncio.Event()
    await eager(runner, provider)
    await runner._handle_input_message(TranscriptionInput(content="hello"))
    await runner._handle_input_message(
        UserStateInput(
            value="idle",
            event_id="final",
            eager_event_id="eager-1",
            turn_end_reason="confirmed",
        )
    )
    for _ in range(20):
        if spoken(socket):
            break
        await asyncio.sleep(0)
    assert [message["content"] for message in spoken(socket)] == ["buffered"]
    assert not runner.agent_task.done()
    provider.hold.set()
    assert runner.agent_task is not None
    await runner.agent_task
    await runner._close_eager_tasks()
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_cancelled_reader_can_finish_cleanup_without_delaying_new_reply():
    runner, _, provider, socket = make_runner()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    started = asyncio.Event()
    calls = []

    def chat(*args, **kwargs):
        index = len(calls)
        calls.append(index)

        async def stream():
            if index == 0:
                started.set()
                try:
                    yield StreamChunk(text="old")
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await release_cleanup.wait()
                    raise RuntimeError("late failure in abandoned cleanup")
            else:
                yield StreamChunk(text="fresh")

        return stream()

    provider.chat = chat
    await runner._handle_input_message(UserTurnEagerInput(content="hello", event_id="eager-1"))
    await started.wait()
    await runner._handle_input_message(UserTurnResumedInput(eager_event_id="eager-1"))
    await cleanup_started.wait()
    await runner._handle_input_message(TranscriptionInput(content="hello"))
    await runner._handle_input_message(
        UserStateInput(value="idle", event_id="final", turn_end_reason="confirmed")
    )
    await asyncio.wait_for(runner.agent_task, timeout=1)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]
    assert not release_cleanup.is_set()
    release_cleanup.set()
    await asyncio.gather(*runner._eager_tasks)
    assert not socket.close.called


@pytest.mark.asyncio
async def test_background_result_arriving_after_prediction_invalidates_request():
    from line.events import AgentToolCalled, AgentToolReturned

    runner, agent, provider, socket = make_runner([StreamChunk(text="old")], [StreamChunk(text="fresh")])
    await eager(runner, provider)

    async def result():
        yield (
            AgentToolCalled(tool_call_id="prior", tool_name="lookup"),
            AgentToolReturned(tool_call_id="prior", tool_name="lookup", result="new information"),
        )

    agent._get_background_event_queue().subscribe(result())
    await agent._get_background_event_queue().wait()
    await final(runner)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]
    assert len(provider.calls) == 2
    assert any(message.content == "new information" for message in provider.calls[1])


@pytest.mark.asyncio
async def test_prepare_failure_falls_back_without_closing_call():
    runner, agent, provider, socket = make_runner([StreamChunk(text="fresh")])
    agent._prepare_response = AsyncMock(side_effect=ValueError("invalid snapshot"))
    await eager(runner, provider)
    assert runner._prepared is None
    await final(runner)
    assert [message["content"] for message in spoken(socket)] == ["fresh"]
    assert not socket.close.called


@pytest.mark.asyncio
async def test_candidate_backpressure_is_bounded_and_cancellation_closes_reader():
    runner, _, provider, socket = make_runner([StreamChunk(text=str(i)) for i in range(100)])
    await eager(runner, provider)
    prepared = runner._prepared
    assert prepared is not None
    assert prepared._queue.qsize() == 64
    assert not prepared.task.done()
    await runner._handle_input_message(UserTurnResumedInput(eager_event_id="eager-1"))
    await asyncio.wait_for(prepared.task, timeout=1)
    assert provider.closed.is_set()
    assert not socket.send_json.called


@pytest.mark.asyncio
async def test_buffer_copies_mutable_tool_chunks_and_keeps_inference_timing():
    runner, _, provider, _ = make_runner()

    def chat(*args, **kwargs):
        async def stream():
            tool = ToolCall(id="tool", name="lookup", arguments="{")
            yield StreamChunk(tool_calls=[tool])
            tool.arguments = "{}"
            tool.is_complete = True
            yield StreamChunk(tool_calls=[tool])
            yield StreamChunk(text="done")

        return stream()

    provider.chat = chat
    await runner._handle_input_message(UserTurnEagerInput(content="hello", event_id="eager-1"))
    prepared = runner._prepared
    assert prepared is not None
    await prepared.task
    assert prepared._queue.get_nowait().tool_calls[0].arguments == "{"
    assert prepared._queue.get_nowait().tool_calls[0].arguments == "{}"
    assert prepared.first_chunk_ms is not None
    assert prepared.first_text_ms is not None
    assert 0 <= prepared.first_chunk_ms <= prepared.first_text_ms
    runner._discard_prepared()
