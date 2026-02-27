"""
Tests for line/voice_agent_app.py

Focuses on:
1. ConversationRunner webhook loop behavior
2. _cancel_agent_task behavior
3. History management
4. _get_processed_history whitespace restoration
"""

import asyncio
from datetime import datetime
from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocket, WebSocketDisconnect
import pytest

from line.agent import TurnEnv
from line.events import (
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentUpdateCall,
    CallEnded,
    CallStarted,
    InputEvent,
    LogMessage,
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

    def test_tts_inserted_full_stop_is_skipped(self):
        """TTS-inserted full stop absent from pending is ignored."""
        # TTS adds a period after "Hello" that wasn't in the original text
        committed, remaining_committed, remaining = _parse_committed("Hello.World", "Hello World")
        assert committed == "Hello World"
        assert remaining_committed == ""
        assert remaining == ""

    def test_tts_inserted_devanagari_danda_is_skipped(self):
        """TTS-inserted Devanagari danda (।) absent from pending is ignored."""
        committed, remaining_committed, remaining = _parse_committed("नमस्ते।दुनिया", "नमस्ते दुनिया")
        assert committed == "नमस्ते दुनिया"
        assert remaining_committed == ""
        assert remaining == ""

    def test_tts_inserted_cjk_full_stop_is_skipped(self):
        """TTS-inserted ideographic full stop (。) absent from pending is ignored."""
        committed, remaining_committed, remaining = _parse_committed("你好。世界", "你好世界")
        assert committed == "你好世界"
        assert remaining_committed == ""
        assert remaining == ""

    def test_full_stop_present_in_both_is_preserved(self):
        """A full stop that exists in both committed and pending is kept."""
        committed, remaining_committed, remaining = _parse_committed("Hello.World", "Hello. World")
        assert committed == "Hello. World"
        assert remaining_committed == ""
        assert remaining == ""

    def test_trailing_tts_full_stop_is_skipped(self):
        """A TTS-inserted full stop at the end of committed is cleaned up."""
        committed, remaining_committed, remaining = _parse_committed("Hello.", "Hello World")
        assert committed == "Hello"
        assert remaining_committed == ""
        assert remaining == " World"

    def test_empty_pending_with_committed_returns_committed(self):
        """When pending_text is empty but committed has content, return committed as-is."""
        committed, remaining_committed, remaining = _parse_committed("Hello world", "")
        assert committed == "Hello world"
        assert remaining_committed == ""
        assert remaining == ""


# ============================================================
# AgentUpdateCall -> ConfigOutput mapping tests
# ============================================================


class TestUpdateCallMapping:
    """Tests for _map_output_event handling of AgentUpdateCall language mapping."""

    def _map(self, event):
        """Helper to call _map_output_event on a ConversationRunner."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)
        return runner._map_output_event(event)

    def test_language_sets_both(self):
        """language='fr' sets TTS language and STT config (backward compat)."""
        result = self._map(AgentUpdateCall(language="fr"))
        assert result.tts.language == "fr"
        assert result.stt is not None
        assert result.stt.language == "fr"

    def test_multilingual_sets_both_to_none(self):
        """language='multilingual' maps to None for both STT and TTS."""
        result = self._map(AgentUpdateCall(language="multilingual"))
        assert result.tts.language is None
        assert result.stt is not None
        assert result.stt.language is None
        assert result.language is None

    def test_all_none_defaults(self):
        """No fields set -> TTS language None, STT config None (no change)."""
        result = self._map(AgentUpdateCall())
        assert result.tts.language is None
        assert result.stt is None


# ============================================================
# WebSocket output truncation tests
# ============================================================


class TestEventMessageTruncation:
    """Tests for _map_output_event truncation of large output event payloads."""

    def _map(self, event):
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)
        return runner._map_output_event(event)

    def test_small_result_not_truncated(self):
        """Tool results under the limit are passed through unchanged."""
        event = AgentToolReturned(tool_call_id="1", tool_name="menu_info", tool_args={}, result="small")
        output = self._map(event)
        assert output.result == "small"

    def test_large_result_truncated(self):
        """Tool results over 30KB are truncated to avoid large WebSocket payloads."""
        large_result = "x" * 40_000
        event = AgentToolReturned(tool_call_id="1", tool_name="menu_info", tool_args={}, result=large_result)
        output = self._map(event)
        assert len(output.result) < 32_000
        assert output.result.endswith("... [truncated]")
        assert output.result.startswith("x" * 100)

    def test_result_at_boundary_not_truncated(self):
        """Tool results exactly at 30000 chars are not truncated."""
        boundary_result = "y" * 30_000
        event = AgentToolReturned(
            tool_call_id="1", tool_name="menu_info", tool_args={}, result=boundary_result
        )
        output = self._map(event)
        assert output.result == boundary_result

    def test_none_result_unchanged(self):
        """None results are passed through as None."""
        event = AgentToolReturned(tool_call_id="1", tool_name="menu_info", tool_args={}, result=None)
        output = self._map(event)
        assert output.result is None

    def test_large_tool_args_truncated(self):
        """AgentToolCalled with large tool_args gets a sentinel dict."""
        large_args = {"data": "x" * 40_000}
        event = AgentToolCalled(tool_call_id="1", tool_name="big_tool", tool_args=large_args)
        output = self._map(event)
        assert output.arguments["_truncated"] is True
        assert "_preview" in output.arguments
        assert len(output.arguments["_preview"]) == 200

    def test_small_tool_args_not_truncated(self):
        """AgentToolCalled with small tool_args passes through unchanged."""
        args = {"query": "hello"}
        event = AgentToolCalled(tool_call_id="1", tool_name="search", tool_args=args)
        output = self._map(event)
        assert output.arguments == args

    def test_large_tool_returned_args_truncated(self):
        """AgentToolReturned also truncates large tool_args."""
        large_args = {"data": "x" * 40_000}
        event = AgentToolReturned(tool_call_id="1", tool_name="big_tool", tool_args=large_args, result="ok")
        output = self._map(event)
        assert output.arguments["_truncated"] is True

    def test_large_log_metadata_truncated(self):
        """LogMessage with large metadata truncates only the inner metadata key."""
        large_metadata = {"blob": "z" * 40_000}
        event = LogMessage(name="test", level="info", message="hi", metadata=large_metadata)
        output = self._map(event)
        assert output.metadata["level"] == "info"
        assert output.metadata["message"] == "hi"
        inner = output.metadata["metadata"]
        assert isinstance(inner, dict)
        assert inner["_truncated"] is True
        assert "_preview" in inner

    def test_small_log_metadata_not_truncated(self):
        """LogMessage with small metadata passes through with original values."""
        event = LogMessage(name="test", level="info", message="hi", metadata={"key": "val"})
        output = self._map(event)
        assert output.metadata["level"] == "info"
        assert output.metadata["message"] == "hi"
        assert output.metadata["metadata"] == {"key": "val"}

    def test_non_serializable_result_does_not_crash(self):
        """Non-JSON-serializable result (e.g. datetime) is converted via str, not crash."""
        event = AgentToolReturned(tool_call_id="1", tool_name="t", tool_args={}, result=datetime.now())
        output = self._map(event)
        assert isinstance(output.result, str)

    def test_non_serializable_tool_args_does_not_crash(self):
        """Non-JSON-serializable values in tool_args don't crash json.dumps."""
        event = AgentToolCalled(tool_call_id="1", tool_name="t", tool_args={"ts": datetime.now()})
        output = self._map(event)
        assert isinstance(output.arguments, dict)

    def test_non_serializable_log_metadata_does_not_crash(self):
        """Non-JSON-serializable values in LogMessage metadata don't crash json.dumps."""
        event = LogMessage(name="test", level="info", message="hi", metadata={"ts": datetime.now()})
        output = self._map(event)
        assert isinstance(output.metadata, dict)
