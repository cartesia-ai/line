"""
Tests for line/v02/voice_agent_app.py

Focuses on:
1. ConversationRunner webhook loop behavior
2. _cancel_agent_task behavior
3. History management
4. _get_processed_history whitespace restoration
"""

import asyncio
from typing import AsyncIterator, List
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi import WebSocket, WebSocketDisconnect

from line.v02.events import (
    AgentSendText,
    CallEnded,
    CallStarted,
    InputEvent,
    OutputEvent,
    SpecificAgentTextSent,
    SpecificCallEnded,
    SpecificCallStarted,
    SpecificInputEvent,
    SpecificUserTextSent,
    SpecificUserTurnEnded,
    SpecificUserTurnStarted,
    UserTurnEnded,
    UserTurnStarted,
)
from line.v02.voice_agent_app import (
    AgentEnv,
    ConversationRunner,
    _get_processed_history,
    _parse_committed,
)


# ============================================================
# Fixtures and Helpers
# ============================================================

env = AgentEnv()


def create_mock_websocket() -> MagicMock:
    """Create a mock WebSocket with async methods."""
    ws = MagicMock(spec=WebSocket)
    ws.receive_json = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


async def noop_agent(env: AgentEnv, event: InputEvent) -> AsyncIterator[OutputEvent]:
    """Agent that yields nothing."""
    return
    yield  # Make this a generator

class TestConversationRunner:
    # ============================================================
    # WS disconnect
    # ============================================================
    @pytest.mark.asyncio
    async def test_disconnect_creates_call_ended_event(self):
        """Verify CallEnded is added to history on disconnect."""
        ws = create_mock_websocket()
        ws.receive_json.side_effect = [WebSocketDisconnect()]

        runner = ConversationRunner(ws, noop_agent, env)
        await runner.run()

        # History should have CallStarted and CallEnded
        assert len(runner.history) == 2
        assert isinstance(runner.history[0], SpecificCallStarted)
        assert isinstance(runner.history[1], SpecificCallEnded)

    @pytest.mark.asyncio
    async def test_disconnect_sets_shutdown_event(self):
        """Verify shutdown_event is set on disconnect."""
        ws = create_mock_websocket()
        ws.receive_json.side_effect = WebSocketDisconnect()

        runner = ConversationRunner(ws, noop_agent, env)
        assert not runner.shutdown_event.is_set()

        await runner.run()

        assert runner.shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_disconnect_stops_loop(self):
        """Verify the run loop exits after disconnect."""
        ws = create_mock_websocket()
        call_count = 0

        async def receive_with_count():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise WebSocketDisconnect()
            return {"type": "message", "content": "hello"}

        ws.receive_json = receive_with_count

        runner = ConversationRunner(ws, noop_agent, env)
        await runner.run()

        # Loop should have exited - verify we didn't spin forever
        assert call_count == 2

    # ============================================================
    # cancelling agents
    # ============================================================

    @pytest.mark.asyncio
    async def test_cancel_triggered_by_user_turn_started(self):
        """Verify agent task is cancelled when UserTurnStarted arrives."""
        ws = create_mock_websocket()
        agent_started = asyncio.Event()
        agent_cancelled = asyncio.Event()
        proceed_to_yield = asyncio.Future()

        async def blocking_agent(env, event):
            agent_started.set()
            try:
                await proceed_to_yield  # Block until cancelled or released
                yield AgentSendText(text="should not reach")
            except asyncio.CancelledError:
                agent_cancelled.set()
                raise

        msg_idx = 0

        async def receive_messages():
            nonlocal msg_idx
            if msg_idx == 0:
                await agent_started.wait()
                msg_idx += 1
                return {"type": "user_state", "value": "speaking"}  # Triggers cancel
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, blocking_agent, env)
        await runner.run()

        assert agent_cancelled.is_set(), "Agent task should have been cancelled"

    @pytest.mark.asyncio
    async def test_cancel_when_no_task_running(self):
        """Verify no error when cancel filter triggers but no task exists."""
        ws = create_mock_websocket()

        def never_run(e):
            return False

        def default_cancel(e):
            return isinstance(e, UserTurnStarted)

        ws.receive_json.side_effect = [
            {"type": "user_state", "value": "speaking"},  # Would trigger cancel
            WebSocketDisconnect(),
        ]

        runner = ConversationRunner(ws, (noop_agent, never_run, default_cancel), env)

        # Should not raise
        await runner.run()
        assert runner.agent_task is None

    @pytest.mark.asyncio
    async def test_cancel_already_completed_task(self):
        """Verify no error when cancel triggers after task naturally completed."""
        ws = create_mock_websocket()
        agent_completed = asyncio.Event()

        async def quick_agent(env, event):
            yield AgentSendText(text="quick response")
            agent_completed.set()

        call_count = 0

        async def receive_messages():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await agent_completed.wait()
                return {"type": "user_state", "value": "speaking"}  # Triggers cancel
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, quick_agent, env)

        # Should not raise
        await runner.run()

    @pytest.mark.asyncio
    async def test_disconnect_awaits_running_agent_task(self):
        """Disconnect should await/cancel any running agent task.

        EXPECTED: When WebSocketDisconnect occurs, any running agent task
        should be awaited before run() returns.

        BUG: Currently, the task may be left dangling.
        """
        ws = create_mock_websocket()
        agent_started = asyncio.Event()
        agent_finished = asyncio.Event()
        agent_blocking = asyncio.Future()

        async def blocking_agent(env, event):
            agent_started.set()
            try:
                await agent_blocking  # Block until cancelled
                yield AgentSendText(text="response")
            except asyncio.CancelledError:
                pass
            finally:
                agent_finished.set()

        async def receive_messages():
            await agent_started.wait()
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, blocking_agent, env)
        await runner.run()

        # BUG: agent_finished is not set because task wasn't awaited
        # EXPECTED: Task should be cancelled/awaited so agent_finished is set
        assert agent_finished.is_set(), (
            "Agent task should be awaited/cancelled on disconnect so it can clean up"
        )


    """Tests for history accumulation and processing."""
    # ============================================================
    # History management
    # ============================================================

    @pytest.mark.asyncio
    async def test_history_accumulates_events(self):
        """Verify events are appended to history in order."""
        ws = create_mock_websocket()
        call_count = 0

        async def receive_messages():
            nonlocal call_count
            call_count += 1
            messages = [
                {"type": "user_state", "value": "speaking"},
                {"type": "message", "content": "hello"},
                {"type": "user_state", "value": "idle"},
            ]
            if call_count <= len(messages):
                return messages[call_count - 1]
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, noop_agent, env)
        await runner.run()

        # Should have: CallStarted, UserTurnStarted, UserTextSent, UserTurnEnded, CallEnded
        assert len(runner.history) == 5
        assert isinstance(runner.history[0], SpecificCallStarted)
        assert isinstance(runner.history[1], SpecificUserTurnStarted)
        assert isinstance(runner.history[2], SpecificUserTextSent)
        assert runner.history[2].content == "hello"
        assert isinstance(runner.history[3], SpecificUserTurnEnded)
        assert isinstance(runner.history[4], SpecificCallEnded)

    def test_turn_content_collects_events_since_turn_started(self):
        """Verify _turn_content collects the right events."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        # Pass history as argument to the pure function
        history = [
            SpecificCallStarted(),
            SpecificUserTurnStarted(),
            SpecificUserTextSent(content="first"),
            SpecificUserTextSent(content="second"),
        ]

        content = runner._turn_content(
            history,
            SpecificUserTurnStarted,
            (SpecificUserTextSent,),
        )

        assert len(content) == 2
        assert content[0].content == "first"
        assert content[1].content == "second"

    def test_turn_content_empty_when_no_start_event(self):
        """Verify _turn_content returns empty list when no start event found."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        history = [
            SpecificCallStarted(),
            SpecificUserTextSent(content="orphan"),
        ]

        content = runner._turn_content(
            history,
            SpecificUserTurnStarted,
            (SpecificUserTextSent,),
        )

        assert content == []

    def test_wrap_with_history_updates_history(self):
        """Verify _wrap_with_history returns updated history."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        initial_history: List[SpecificInputEvent] = []
        event = SpecificCallStarted()

        result_event, new_history = runner._wrap_with_history(initial_history, event)

        assert len(new_history) == 1
        assert new_history[0] is event
        assert isinstance(result_event, CallStarted)

    def test_wrap_with_history_preserves_existing_events(self):
        """Verify existing history is preserved when adding new events."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        existing = [SpecificCallStarted(), SpecificUserTurnStarted()]
        new_event = SpecificUserTextSent(content="test")

        _, new_history = runner._wrap_with_history(existing, new_event)

        assert len(new_history) == 3
        assert new_history[0] is existing[0]
        assert new_history[1] is existing[1]
        assert new_history[2] is new_event


# ============================================================
# 2) _get_processed_history tests
# ============================================================


class TestGetProcessedHistory:
    """Tests for whitespace restoration in history."""

    def test_empty_history(self):
        result = _get_processed_history("x", [])
        assert result == []

    def test_no_agent_text_events(self):
        history = [SpecificCallStarted(), SpecificUserTextSent(content="x")]
        result = _get_processed_history("", history)
        assert result == history

    def test_restores_whitespace(self):
        result = _get_processed_history("a b", [SpecificAgentTextSent(content="ab")])
        assert len(result) == 1
        assert result[0].content == "a b"

    def test_partial_commit(self):
        result = _get_processed_history("a b c", [SpecificAgentTextSent(content="ab")])
        assert len(result) == 1
        assert result[0].content == "a b"

    def test_multiple_agent_text_events(self):
        history = [
            SpecificAgentTextSent(content="ab"),
            SpecificAgentTextSent(content="cd"),
        ]
        result = _get_processed_history("a b c d", history)
        assert len(result) == 2
        assert result[0].content == "a b"
        assert result[1].content == "c d"

    def test_no_spaces_passthrough(self):
        result = _get_processed_history("ab", [SpecificAgentTextSent(content="ab")])
        assert len(result) == 1
        assert result[0].content == "ab"

    def test_mixed_events_preserved(self):
        history = [
            SpecificCallStarted(),
            SpecificAgentTextSent(content="ab"),
            SpecificUserTurnStarted(),
        ]
        result = _get_processed_history("a b", history)
        assert len(result) == 3
        assert isinstance(result[0], SpecificCallStarted)
        assert result[1].content == "a b"
        assert isinstance(result[2], SpecificUserTurnStarted)


# ============================================================
# _parse_committed tests (helper function)
# ============================================================


class TestParseCommitted:
    """Tests for the _parse_committed helper function."""

    def test_exact_match(self):
        """When speech exactly matches pending (minus whitespace)."""
        committed, remaining = _parse_committed("a b c", "abc")
        assert committed == "a b c"
        assert remaining == ""

    def test_partial_match(self):
        """When speech matches only the beginning of pending."""
        committed, remaining = _parse_committed("a b c d", "abc")
        assert committed == "a b c"
        assert remaining == " d"

    def test_empty_pending(self):
        """Empty pending text returns speech text for non-latin handling."""
        committed, remaining = _parse_committed("abc", "abc")
        assert committed == "abc"
        assert remaining == ""

    def test_preserves_punctuation(self):
        """Punctuation is preserved during matching."""
        committed, remaining = _parse_committed("a!", "a!")
        assert committed == "a!"
        assert remaining == ""

    def test_non_space_commit_preserves_remaining(self):
        """Partial commit of non-latin text should preserve remaining.
        """
        # Pending has two "sentences", only first is committed
        pending = "ab"  
        speech = "a"  

        committed, remaining = _parse_committed(pending, speech)

        assert committed == "a" 
        assert remaining == "b", (
            f"Expected remaining to be 'b', got '{remaining}'. "
            "Non-latin partial commit should preserve remaining text."
        )

    def test_non_space_commit_preserves_skips_as_necessary(self):
        """Partial commit of non-latin text should preserve remaining.
        """
        # Pending has two "sentences", only first is committed
        pending = "abc"  
        speech = "b"  

        committed, remaining = _parse_committed(pending, speech)

        assert committed == "b" 
        assert remaining == "c", (
            f"Expected remaining to be 'c', got '{remaining}'. "
            "Non-latin partial commit should preserve remaining text."
        )
