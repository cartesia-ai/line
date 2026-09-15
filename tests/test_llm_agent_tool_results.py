"""Tool results can update context without requesting another LLM response."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocket, WebSocketDisconnect
import pytest
from test_llm_agent_llm_agent import collect_outputs, create_agent_with_mock

from line.agent import AgentEnv, TurnEnv
from line.events import AgentToolCalled, AgentToolReturned, UserTextSent
from line.llm_agent import LlmConfig, ToolResult, loopback_tool
from line.llm_agent.provider import StreamChunk, ToolCall, _normalize_messages
from line.voice_agent_app import ConversationRunner

pytestmark = pytest.mark.asyncio


def call_tool(name, call_id="call"):
    return StreamChunk(tool_calls=[ToolCall(id=call_id, name=name, arguments="{}", is_complete=True)])


def user_message(text):
    return UserTextSent(content=text, history=[UserTextSent(content=text)])


@pytest.mark.parametrize("is_background", [False, True])
@pytest.mark.parametrize("result_kind", ["plain", "default", "silent"])
async def test_returned_result_controls_loopback_and_preserves_payload(is_background, result_kind):
    payload = {"status": "saved", "run_llm": "ordinary payload data"}

    @loopback_tool(is_background=is_background)
    async def save(ctx):
        if result_kind == "plain":
            return payload
        if result_kind == "default":
            return ToolResult(payload)
        return ToolResult(payload, run_llm=False)

    agent, llm = create_agent_with_mock(
        [[call_tool("save")], [StreamChunk(text="Saved.", is_final=True)]], tools=[save]
    )
    outputs = await collect_outputs(agent, TurnEnv(agent_env=AgentEnv()), user_message("Save this"))

    assert len(llm._recorded_messages) == (1 if result_kind == "silent" else 2)
    assert [o.result for o in outputs if isinstance(o, AgentToolReturned)] == [payload]
    messages = await agent._build_messages()
    assert any(m.role == "tool" and '"status": "saved"' in m.content for m in messages)
    await agent.cleanup()


@pytest.mark.parametrize("silent_first", [False, True])
async def test_silent_result_does_not_override_another_results_loopback(silent_first):
    @loopback_tool
    async def update(ctx):
        results = ["complete", ToolResult("progress", run_llm=False)]
        for result in reversed(results) if silent_first else results:
            yield result

    agent, llm = create_agent_with_mock(
        [[call_tool("update")], [StreamChunk(text="Done.", is_final=True)]], tools=[update]
    )
    await collect_outputs(agent, TurnEnv(agent_env=AgentEnv()), user_message("Update"))

    assert len(llm._recorded_messages) == 2
    assert {m.content for m in llm._recorded_messages[1] if m.role == "tool"} == {"progress", "complete"}
    await agent.cleanup()


async def test_silent_background_updates_do_not_spend_llm_iterations():
    @loopback_tool(is_background=True)
    async def update(ctx):
        for n in range(10):
            yield ToolResult(n, run_llm=False)
            # Let the consumer drain each update while the tool is still running.
            await asyncio.sleep(0)
        yield "complete"

    agent, llm = create_agent_with_mock(
        [[call_tool("update")], [StreamChunk(text="Done.", is_final=True)]],
        tools=[update],
        max_tool_iterations=2,
    )
    outputs = await collect_outputs(agent, TurnEnv(agent_env=AgentEnv()), user_message("Update"))

    assert len(llm._recorded_messages) == 2
    assert [o.result for o in outputs if isinstance(o, AgentToolReturned)] == [*range(10), "complete"]
    assert llm._recorded_messages[1][-1].content == "complete"
    await agent.cleanup()


@pytest.mark.parametrize("is_background", [False, True])
async def test_error_after_silent_progress_still_triggers_a_response(is_background):
    @loopback_tool(is_background=is_background)
    async def lookup(ctx):
        yield ToolResult("pending", run_llm=False)
        raise ValueError("lookup failed")

    agent, llm = create_agent_with_mock(
        [[call_tool("lookup")], [StreamChunk(text="Lookup failed.", is_final=True)]], tools=[lookup]
    )
    outputs = await collect_outputs(agent, TurnEnv(agent_env=AgentEnv()), user_message("Look it up"))

    assert len(llm._recorded_messages) == 2
    assert [o.result for o in outputs if isinstance(o, AgentToolReturned)] == [
        "pending",
        "error: lookup failed",
    ]
    await agent.cleanup()


async def test_cancellation_between_tool_events_does_not_lose_silent_result():
    release = asyncio.Event()
    called = asyncio.Event()

    @loopback_tool(is_background=True)
    async def lookup(ctx):
        yield ToolResult("pending", run_llm=False)
        await release.wait()
        yield "complete"

    agent, llm = create_agent_with_mock(
        [[call_tool("lookup")], [StreamChunk(text="Done.", is_final=True)]], tools=[lookup]
    )
    env = TurnEnv(agent_env=AgentEnv())
    first_event = user_message("Look it up")

    async def first_turn():
        async for output in agent.process(env, first_event):
            if isinstance(output, AgentToolCalled):
                called.set()
                await asyncio.Event().wait()

    task = asyncio.create_task(first_turn())
    try:
        await asyncio.wait_for(called.wait(), 2)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        assert len(llm._recorded_messages) == 1
        messages = await agent._build_messages()
        assert [m.content for m in messages if m.role == "tool"] == ["pending"]

        release.set()
        await agent._get_background_event_queue().wait()
        second_event = user_message("Done")
        second_event.history = first_event.history + second_event.history
        await collect_outputs(agent, env, second_event)
        assert [m.content for m in llm._recorded_messages[1] if m.role == "tool"] == ["pending", "complete"]
    finally:
        task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await agent.cleanup()


@pytest.mark.parametrize("pending_after_interruption", [False, True])
@pytest.mark.parametrize("complete_during_response", [False, True])
async def test_silent_background_tool_through_websocket_runner(
    pending_after_interruption, complete_during_response
):
    """Real SDK runner and tool loop, with a scripted LLM and gated backend work."""
    started, pending_allowed, pending_queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    backend_release, backend_completed = asyncio.Event(), asyncio.Event()
    response_finished = asyncio.Event()
    invocations = []

    @loopback_tool(is_background=True)
    async def verify(ctx):
        invocations.append("verify")
        started.set()
        await pending_allowed.wait()
        yield ToolResult({"status": "in_progress"}, run_llm=False)
        pending_queued.set()
        await backend_release.wait()
        yield {"status": "verified"}
        backend_completed.set()

    agent, llm = create_agent_with_mock(
        [
            [call_tool("verify")],
            [StreamChunk(text="I have your submission.", is_final=True)],
            [StreamChunk(text="Verified.", is_final=True)],
        ],
        tools=[verify],
        config=LlmConfig(introduction=""),
    )
    original_chat = llm.chat

    def chat(messages, tools=None, **kwargs):
        index = len(llm._recorded_messages)
        stream = original_chat(messages, tools, **kwargs)

        async def response():
            if index == 1 and complete_during_response:
                backend_release.set()
                await backend_completed.wait()
            async for chunk in stream:
                yield chunk
            if index == 1:
                response_finished.set()

        return response()

    llm.chat = chat
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()

    async def receive():
        message = await incoming.get()
        if isinstance(message, Exception):
            raise message
        return message

    async def next_output(kind, **fields):
        while True:
            output = await asyncio.wait_for(outgoing.get(), 2)
            if output["type"] == kind and all(output.get(k) == v for k, v in fields.items()):
                return output

    ws = MagicMock(spec=WebSocket)
    ws.receive_json = receive
    ws.send_json = outgoing.put
    ws.close = AsyncMock()
    runner = ConversationRunner(ws, agent, AgentEnv())
    runner_task = asyncio.create_task(runner.run())
    try:
        await next_output("log_metric", name="agent_turn_ms")
        incoming.put_nowait({"type": "user_state", "value": "speaking"})
        incoming.put_nowait({"type": "message", "content": "Verify my submission"})
        incoming.put_nowait({"type": "user_state", "value": "idle"})
        await asyncio.wait_for(started.wait(), 2)
        first_task = runner.agent_task
        assert first_task is not None
        if not pending_after_interruption:
            pending_allowed.set()
            await next_output("tool_call", result='{"status": "in_progress"}')

        incoming.put_nowait({"type": "user_state", "value": "speaking"})
        await asyncio.wait_for(first_task, 2)
        # Speech cancelled only the generation task, not the background tool.
        assert invocations == ["verify"]
        assert not backend_completed.is_set()
        assert len(llm._recorded_messages) == 1

        pending_allowed.set()
        await asyncio.wait_for(pending_queued.wait(), 2)
        incoming.put_nowait({"type": "message", "content": "Done"})
        incoming.put_nowait({"type": "user_state", "value": "idle"})
        await next_output("message", content="I have your submission.")
        await asyncio.wait_for(response_finished.wait(), 2)
        messages = llm._recorded_messages[1]
        assert [m.content for m in messages if m.role == "tool"] == ['{"status": "in_progress"}']

        backend_release.set()
        result = await next_output("tool_call", result='{"status": "verified"}')
        assert result["arguments"] == {}
        await next_output("message", content="Verified.")
        assert invocations == ["verify"]
        assert len(llm._recorded_messages) == 3
        messages = llm._recorded_messages[2]
        assert messages[-1].role == "tool"
        assert messages[-1].content == '{"status": "verified"}'
        assert _normalize_messages(messages) is not None
        assert not runner_task.done()
    finally:
        pending_allowed.set()
        backend_release.set()
        incoming.put_nowait(WebSocketDisconnect())
        await asyncio.wait_for(runner_task, 2)
        await agent.cleanup()
