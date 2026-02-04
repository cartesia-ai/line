"""
Tests for line/voice_agent_app.py

Focuses on:
1. ConversationRunner webhook loop behavior
2. _cancel_agent_task behavior
3. History management
4. _get_processed_history whitespace restoration
"""

import asyncio
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocket, WebSocketDisconnect
import pytest

from line.agent import TurnEnv
from line.events import (
    AgentSendText,
    AgentTextSent,
    CallEnded,
    CallStarted,
    InputEvent,
    OutputEvent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)
from line.voice_agent_app import (
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


async def noop_agent(env: TurnEnv, event: InputEvent) -> AsyncIterator[OutputEvent]:
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
        assert isinstance(runner.history[0], CallStarted)
        assert isinstance(runner.history[1], CallEnded)

    @pytest.mark.asyncio
    async def test_disconnect_sets_shutdown_event(self):
        """Verify shutdown_event is set on disconnect."""
        ws = create_mock_websocket()
        ws.receive_json.side_effect = WebSocketDisconnect()

        runner = ConversationRunner(ws, noop_agent, env)
        assert not runner.shutdown_event.is_set()

        await runner.run()

        assert runner.shutdown_event.is_set()

    # ============================================================
    # Fatal error handling
    # ============================================================
    @pytest.mark.asyncio
    async def test_fatal_error_closes_websocket(self):
        """Verify websocket is closed when a fatal exception occurs during message processing."""
        ws = create_mock_websocket()
        ws.close = AsyncMock()

        # First call raises a generic exception, simulating a fatal error
        ws.receive_json.side_effect = RuntimeError("Simulated fatal error")

        runner = ConversationRunner(ws, noop_agent, env)
        await runner.run()

        # Verify shutdown_event is set
        assert runner.shutdown_event.is_set()

        # Verify error was sent with full traceback
        ws.send_json.assert_called()
        sent_data = ws.send_json.call_args[0][0]
        assert "error" in sent_data.get("type", "") or "content" in sent_data
        # Error message should contain both the exception and traceback info
        assert "Simulated fatal error" in str(sent_data)
        assert "RuntimeError" in str(sent_data)

        # Verify websocket was closed
        ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fatal_error_stops_loop(self):
        """Verify the run loop exits after a fatal error."""
        ws = create_mock_websocket()
        ws.close = AsyncMock()
        call_count = 0

        async def receive_with_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"type": "message", "content": "hello"}
            # Second call raises a fatal error
            raise ValueError("Something went wrong")

        ws.receive_json = receive_with_error

        runner = ConversationRunner(ws, noop_agent, env)
        await runner.run()

        # Loop should have exited after the error
        assert call_count == 2
        assert runner.shutdown_event.is_set()
        ws.close.assert_called_once()

        # Verify error message contains full traceback
        ws.send_json.assert_called()
        sent_data = ws.send_json.call_args[0][0]
        assert "Something went wrong" in str(sent_data)
        assert "ValueError" in str(sent_data)

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
            await agent_blocking  # Block until cancelled
            yield AgentSendText(text="response")
            agent_finished.set()

        async def receive_messages():
            await agent_started.wait()
            print("Disconnecting")
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, blocking_agent, env)
        runner_task = runner.run()
        agent_blocking.set_result(None)
        await runner_task

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
        assert isinstance(runner.history[0], CallStarted)
        assert isinstance(runner.history[1], UserTurnStarted)
        assert isinstance(runner.history[2], UserTextSent)
        assert runner.history[2].content == "hello"
        assert isinstance(runner.history[3], UserTurnEnded)
        assert isinstance(runner.history[4], CallEnded)

    def test_turn_content_collects_events_since_turn_started(self):
        """Verify _turn_content collects the right events."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        # Pass history as argument to the pure function
        history = [
            CallStarted(),
            UserTurnStarted(),
            UserTextSent(content="first"),
            UserTextSent(content="second"),
        ]

        content = runner._turn_content(
            history,
            UserTurnStarted,
            (UserTextSent,),
        )

        assert len(content) == 2
        assert content[0].content == "first"
        assert content[1].content == "second"

    def test_turn_content_empty_when_no_start_event(self):
        """Verify _turn_content returns empty list when no start event found."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        history = [
            CallStarted(),
            UserTextSent(content="orphan"),
        ]

        content = runner._turn_content(
            history,
            UserTurnStarted,
            (UserTextSent,),
        )

        assert content == []

    def test_process_input_event_updates_history(self):
        """Verify _process_input_event returns updated history."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        initial_history: List[InputEvent] = []
        event = CallStarted()

        result_event, new_history = runner._process_input_event(initial_history, event)

        assert len(new_history) == 1
        assert new_history[0] is event
        assert isinstance(result_event, CallStarted)

    def test_process_input_event_preserves_existing_events(self):
        """Verify existing history is preserved when adding new events."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        existing = [CallStarted(), UserTurnStarted()]
        new_event = UserTextSent(content="test")

        _, new_history = runner._process_input_event(existing, new_event)

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
        history = [CallStarted(), UserTextSent(content="x")]
        result = _get_processed_history("", history)
        assert result == history

    def test_restores_whitespace(self):
        result = _get_processed_history("a b", [AgentTextSent(content="ab")])
        assert len(result) == 1
        assert result[0].content == "a b"

    def test_partial_commit(self):
        result = _get_processed_history("a b c", [AgentTextSent(content="ab")])
        assert len(result) == 1
        assert result[0].content == "a b"

    def test_multiple_agent_text_events(self):
        history = [
            AgentTextSent(content="ab"),
            AgentTextSent(content="cd"),
        ]
        result = _get_processed_history("a b c d", history)
        assert len(result) == 1
        assert result[0].content == "a b c d"

    def test_no_spaces_passthrough(self):
        result = _get_processed_history("ab", [AgentTextSent(content="ab")])
        assert len(result) == 1
        assert result[0].content == "ab"

    def test_mixed_events_preserved(self):
        history = [
            CallStarted(),
            AgentTextSent(content="ab"),
            UserTurnStarted(),
        ]
        result = _get_processed_history("a b", history)
        assert len(result) == 3
        assert isinstance(result[0], CallStarted)
        assert result[1].content == "a b"
        assert isinstance(result[2], UserTurnStarted)


# ============================================================
# _parse_committed tests (helper function)
# ============================================================


class TestParseCommitted:
    """Tests for the _parse_committed helper function."""

    def test_exact_match(self):
        """When speech exactly matches pending (minus whitespace)."""
        committed, _, remaining = _parse_committed("abc", "a b c")
        assert committed == "a b c"
        assert remaining == ""

    def test_partial_match(self):
        """When speech matches only the beginning of pending."""
        committed, _, remaining = _parse_committed("abc", "a b c d")
        assert committed == "a b c"
        assert remaining == " d"

    def test_empty_pending(self):
        """Empty pending text returns speech text for non-latin handling."""
        committed, _, remaining = _parse_committed("abc", "abc")
        assert committed == "abc"
        assert remaining == ""

    def test_preserves_punctuation(self):
        """Punctuation is preserved during matching."""
        committed, _, remaining = _parse_committed("a!", "a!")
        assert committed == "a!"
        assert remaining == ""

    def test_non_space_commit_preserves_remaining(self):
        """Partial commit of non-latin text should preserve remaining."""
        # Pending has two "sentences", only first is committed
        speech = "a"
        pending = "ab"

        committed, _, remaining = _parse_committed(speech, pending)

        assert committed == "a"
        assert remaining == "b", (
            f"Expected remaining to be 'b', got '{remaining}'. "
            "Non-latin partial commit should preserve remaining text."
        )

    def test_non_space_commit_preserves_skips_as_necessary(self):
        """Partial commit of non-latin text should preserve remaining."""
        # Pending has two "sentences", only first is committed
        speech = "b"
        pending = "abc"

        committed, _, remaining = _parse_committed(speech, pending)

        assert committed == "b"
        assert remaining == "c", (
            f"Expected remaining to be 'c', got '{remaining}'. "
            "Non-latin partial commit should preserve remaining text."
        )

    def test_emojis_spliced_back_like_whitespace(self):
        """Emojis stripped by API harness are restored like whitespace."""
        # API harness strips emojis from AgentTextSent, just like whitespace
        committed, _, remaining = _parse_committed("Hello world", "Hello 👋 world")
        assert committed == "Hello 👋 world"
        assert remaining == ""
