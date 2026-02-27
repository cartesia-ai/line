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

from line._harness_types import MessageOutput
from line.agent import TurnEnv
from line.events import (
    AgentEnableMultilingualSTT,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentUpdateCall,
    AgentUpdateTTS,
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
    _consume_expected_ack_back_prefix,
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
# Uninterruptible message tests
# ============================================================


class TestUninterruptibleMessages:
    """Tests for AgentSendText(interruptible=False) handling."""

    @pytest.mark.asyncio
    async def test_uninterruptible_maps_to_message_output_with_interruptible_false(self):
        """AgentSendText(interruptible=False) maps to MessageOutput(interruptible=False)."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        result = runner._map_output_event(AgentSendText(text="This call is recorded.", interruptible=False))
        assert isinstance(result, MessageOutput)
        assert result.content == "This call is recorded."
        assert result.interruptible is False

    @pytest.mark.asyncio
    async def test_regular_send_text_maps_to_interruptible_true(self):
        """AgentSendText maps to MessageOutput(interruptible=True) by default."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        result = runner._map_output_event(AgentSendText(text="Hello!"))
        assert isinstance(result, MessageOutput)
        assert result.content == "Hello!"
        assert result.interruptible is True

    @pytest.mark.asyncio
    async def test_cancel_filter_still_applies_during_uninterruptible(self):
        """Cancel filter applies normally even during uninterruptible playback."""
        ws = create_mock_websocket()
        agent_started = asyncio.Event()
        agent_cancelled = asyncio.Event()
        proceed_to_yield = asyncio.Future()

        async def blocking_agent(env, event):
            if isinstance(event, CallStarted):
                agent_started.set()
                try:
                    yield AgentSendText(text="Legal disclaimer.", interruptible=False)
                    await proceed_to_yield
                except asyncio.CancelledError:
                    agent_cancelled.set()
                    raise

        msg_idx = 0

        async def receive_messages():
            nonlocal msg_idx
            if msg_idx == 0:
                await agent_started.wait()
                msg_idx += 1
                return {"type": "user_state", "value": "speaking"}
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, blocking_agent, env)
        await runner.run()
        assert agent_cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancel_does_not_clear_suppression_buffer(self):
        """Canceling an LM task must not clear expected ack-back."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)
        runner._suppression_buffer = "Legal disclaimer."

        task_started = asyncio.Event()

        async def blocking_task():
            task_started.set()
            await asyncio.Future()

        runner.agent_task = asyncio.create_task(blocking_task())
        await task_started.wait()

        await runner._cancel_agent_task()

        assert runner._suppression_buffer == "Legal disclaimer."
        assert runner.agent_task is None

    @pytest.mark.asyncio
    async def test_late_ack_back_after_cancel_is_still_suppressed(self):
        """Late ack-back chunks should still be suppressed after task cancellation."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)
        runner._suppression_buffer = "Legal disclaimer."

        task_started = asyncio.Event()

        async def blocking_task():
            task_started.set()
            await asyncio.Future()

        runner.agent_task = asyncio.create_task(blocking_task())
        await task_started.wait()
        await runner._cancel_agent_task()

        result = runner._suppress_ack_back(AgentTextSent(content="Legaldisclaimer."))
        assert result is None
        assert runner._suppression_buffer == ""

    @pytest.mark.asyncio
    async def test_late_ack_back_after_cancel_does_not_leak_into_processed_history(self):
        """Late disclaimer ack-back should not leak or duplicate user events after cancel+restart."""
        ws = create_mock_websocket()
        first_turn_started = asyncio.Event()
        first_turn_block = asyncio.Future()

        async def scripted_agent(env, event):
            if isinstance(event, CallStarted):
                yield AgentSendText(text="This call may be recorded.", interruptible=False)
                first_turn_started.set()
                await first_turn_block
            elif isinstance(event, UserTurnEnded):
                yield AgentSendText(text="Thank you", interruptible=True)

        runner = ConversationRunner(ws, scripted_agent, env)

        await runner._start_agent_task(TurnEnv(), CallStarted())
        await first_turn_started.wait()

        # Partial disclaimer ack-back arrives before the old task is canceled.
        partial = runner._suppress_ack_back(AgentTextSent(content="Thiscallmay"))
        assert partial is None
        assert runner._suppression_buffer

        # UserTurnEnded starts a new LM run and cancels the old task.
        await runner._start_agent_task(TurnEnv(), UserTurnEnded())

        # Late disclaimer ack-back chunk after cancel should still be suppressed.
        late = runner._suppress_ack_back(AgentTextSent(content="berecorded."))
        assert late is None
        assert runner._suppression_buffer == ""

        if runner.agent_task:
            await runner.agent_task

        history: List[InputEvent] = []
        _, history = runner._process_input_event(history, CallStarted())
        _, history = runner._process_input_event(history, UserTextSent(content="Hello?"))
        _, history = runner._process_input_event(history, UserTurnEnded(content=[]))
        processed_event, history = runner._process_input_event(history, AgentTextSent(content="Thankyou"))

        assert isinstance(processed_event, AgentTextSent)
        assert processed_event.content == "Thank you"

        processed_history = processed_event.history or []
        user_texts = [
            evt for evt in processed_history if isinstance(evt, UserTextSent) and evt.content == "Hello?"
        ]
        assert len(user_texts) == 1

        leaked_disclaimer = [
            evt
            for evt in processed_history
            if isinstance(evt, AgentTextSent) and "recorded" in evt.content.lower()
        ]
        assert leaked_disclaimer == []

    def test_exact_ack_back_uninterruptible_text_is_suppressed(self):
        """Exact ack-back matching emitted uninterruptible text is dropped."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "This call is recorded."
        result = runner._suppress_ack_back(AgentTextSent(content="This call is recorded."))
        assert result is None
        assert runner._suppression_buffer == ""

    def test_ack_back_uninterruptible_text_is_suppressed(self):
        """Ack-back matching emitted uninterruptible text is dropped."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "This call is recorded."
        result = runner._suppress_ack_back(AgentTextSent(content="Thiscallisrecorded."))
        assert result is None
        assert runner._suppression_buffer == ""

    def test_ack_back_uninterruptible_prefix_is_suppressed_and_suffix_kept(self):
        """Only the uninterruptible prefix is dropped when a chunk contains additional text."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Legal notice."
        result = runner._suppress_ack_back(AgentTextSent(content="Legalnotice.Hello"))

        assert result is not None
        assert result.content == "Hello"
        assert runner._suppression_buffer == ""

    def test_non_prefix_text_is_not_suppressed(self):
        """Suppression requires prefix alignment and does not skip pending text."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Legal Hello"
        result = runner._suppress_ack_back(AgentTextSent(content="Hello"))

        assert result is not None
        assert result.content == "Hello"
        assert runner._suppression_buffer == "Legal Hello"

    def test_multichunk_ack_back_then_interruptible_text(self):
        """After uninterruptible ack-back is consumed, following chunk passes through untouched."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Legal notice."
        first = runner._suppress_ack_back(AgentTextSent(content="Legalnotice."))
        second = runner._suppress_ack_back(AgentTextSent(content="Hello there"))

        assert first is None
        assert second is not None
        assert second.content == "Hello there"
        assert runner._suppression_buffer == ""

    def test_trailing_stripped_ack_back_chars_are_drained(self):
        """Trailing pending chars stripped by harness do not get stuck in ack-back buffer."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Legal notice. "
        result = runner._suppress_ack_back(AgentTextSent(content="Legalnotice."))

        assert result is None
        assert runner._suppression_buffer == ""

    def test_trailing_tts_full_stop_in_ack_back_is_consumed(self):
        """TTS-injected trailing full stop should not leak through suppression."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        # Agent emitted "Hello" without a period, but TTS appended one
        runner._suppression_buffer = "Hello"
        result = runner._suppress_ack_back(AgentTextSent(content="Hello."))

        assert result is None
        assert runner._suppression_buffer == ""

    @pytest.mark.asyncio
    async def test_agent_exception_clears_suppression_buffer(self):
        """Unhandled agent exceptions clear suppression buffer state."""
        ws = create_mock_websocket()
        ws.close = AsyncMock()

        async def failing_agent(env, event):
            if isinstance(event, CallStarted):
                yield AgentSendText(text="Legal disclaimer.", interruptible=False)
                raise RuntimeError("boom")

        ws.receive_json.side_effect = [WebSocketDisconnect()]

        runner = ConversationRunner(ws, failing_agent, env)
        await runner.run()

        assert runner._suppression_buffer == ""

    @pytest.mark.asyncio
    async def test_deferred_commit_interruptible_then_uninterruptible_before_user_event(self):
        """Interruptible ack-back passes through; uninterruptible text is buffered and committed before user event."""
        ws = create_mock_websocket()
        agent_yielded = asyncio.Event()

        async def scripted_agent(env, event):
            if isinstance(event, CallStarted):
                yield AgentSendText(text="Hello there.", interruptible=True)
                yield AgentSendText(text="This call is recorded.", interruptible=False)
                agent_yielded.set()
                await asyncio.Future()

        msg_idx = 0

        async def receive_messages():
            nonlocal msg_idx
            if msg_idx == 0:
                await agent_yielded.wait()
                msg_idx += 1
                # Interruptible ack-back arrives before user event
                return {"type": "agent_speech", "content": "Hellothere."}
            if msg_idx == 1:
                msg_idx += 1
                return {"type": "user_state", "value": "speaking"}
            if msg_idx == 2:
                msg_idx += 1
                return {"type": "transcription", "content": "Hi"}
            raise WebSocketDisconnect()

        ws.receive_json = receive_messages

        runner = ConversationRunner(ws, scripted_agent, env)
        await runner.run()

        # Interruptible ack-back should appear in history (passed through)
        # Uninterruptible text should be flushed before user event
        agent_text_indices = []
        user_turn_started_idx = None
        for i, evt in enumerate(runner.history):
            if isinstance(evt, AgentTextSent):
                agent_text_indices.append(i)
            if isinstance(evt, UserTurnStarted) and user_turn_started_idx is None:
                user_turn_started_idx = i

        # Both ack-back and deferred commit should be in history
        assert len(agent_text_indices) >= 2
        assert user_turn_started_idx is not None
        # Deferred commit (uninterruptible) should appear before user event
        assert agent_text_indices[-1] < user_turn_started_idx

    @pytest.mark.asyncio
    async def test_deferred_commit_tool_calls_between_texts(self):
        """Tool calls between interruptible and uninterruptible text are preserved as separate commits."""
        ws = create_mock_websocket()

        async def scripted_agent(env, event):
            if isinstance(event, CallStarted):
                yield AgentSendText(text="Let me check.", interruptible=True)
                yield AgentToolCalled(tool_call_id="tc1", tool_name="lookup", tool_args={})
                yield AgentToolReturned(tool_call_id="tc1", tool_name="lookup", tool_args={}, result="found")
                yield AgentSendText(text="Legal notice.", interruptible=False)
                yield AgentSendText(text="How can I help?", interruptible=True)

        runner = ConversationRunner(ws, scripted_agent, env)
        await runner._start_agent_task(TurnEnv(), CallStarted())
        await runner.agent_task

        # Flush pending commits (simulating a user event arriving)
        runner._flush_pending_commits()

        # Pending commits should contain uninterruptible + subsequent interruptible
        # (already flushed), so check history for individual entries
        agent_texts = [e for e in runner.history if isinstance(e, AgentTextSent)]
        assert len(agent_texts) == 2
        assert agent_texts[0].content == "Legal notice."
        assert agent_texts[1].content == "How can I help?"

        # emitted_agent_text should contain all text for whitespace restoration
        assert runner.emitted_agent_text == "Let me check.Legal notice.How can I help?"


# ============================================================
# Adversarial _consume_expected_ack_back_prefix tests
# ============================================================


class TestConsumeExpectedAckBackAdversarial:
    """Adversarial edge cases for _consume_expected_ack_back_prefix."""

    # -- Pure full-stop committed text should NOT match --

    def test_lone_full_stop_does_not_match_pending(self):
        """A lone TTS-injected '.' must not falsely suppress against real pending text."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix(".", "Hello")
        assert consumed == 0
        assert remaining_c == "."
        assert remaining_p == "Hello"

    def test_multiple_full_stops_do_not_match_pending(self):
        """Pure full-stop committed '...' must not match against any pending text."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("...", "Hello world")
        assert consumed == 0
        assert remaining_c == "..."
        assert remaining_p == "Hello world"

    def test_devanagari_full_stop_does_not_match_pending(self):
        """Devanagari danda '।' alone must not falsely suppress."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("।", "नमस्ते")
        assert consumed == 0
        assert remaining_c == "।"
        assert remaining_p == "नमस्ते"

    def test_cjk_full_stop_does_not_match_pending(self):
        """CJK full stop '。' alone must not falsely suppress."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("。", "你好")
        assert consumed == 0
        assert remaining_c == "。"
        assert remaining_p == "你好"

    # -- Full stops combined with real content SHOULD match --

    def test_leading_full_stop_before_real_content_is_consumed(self):
        """A TTS-injected leading '.' before real content is valid noise."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix(".Hello", "Hello")
        assert consumed == 6  # ".Hello" fully consumed
        assert remaining_c == ""
        assert remaining_p == ""

    def test_multiple_trailing_full_stops_after_real_match(self):
        """Multiple TTS-injected trailing full stops are swept after a real match."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hello...", "Hello")
        assert consumed == 8
        assert remaining_c == ""
        assert remaining_p == ""

    def test_mixed_full_stop_types_around_real_content(self):
        """Mixed full stop characters (.।。) around real content are all swept."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix(".Hello।。", "Hello")
        assert consumed == 8
        assert remaining_c == ""
        assert remaining_p == ""

    # -- Whitespace and emoji in pending --

    def test_leading_whitespace_in_pending_is_skipped(self):
        """Leading whitespace in pending is harness-stripped and should be skipped."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hello", "  Hello")
        assert consumed == 5
        assert remaining_c == ""
        assert remaining_p == ""

    def test_emoji_in_pending_is_skipped(self):
        """Emoji characters in pending are harness-stripped and should be skipped."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hellothere", "Hello 👋 there")
        assert consumed == 10
        assert remaining_c == ""
        assert remaining_p == ""

    def test_emoji_only_pending_does_not_match(self):
        """Pure emoji pending with real committed text should not match."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hello", "👋🎉")
        assert consumed == 0
        assert remaining_c == "Hello"
        assert remaining_p == "👋🎉"

    # -- Partial consumption and multi-chunk scenarios --

    def test_partial_committed_against_longer_pending(self):
        """Committed is a prefix of the full pending text; remainder stays pending."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hello", "Hello world goodbye")
        assert consumed == 5
        assert remaining_c == ""
        # Leading whitespace after match is drained (it's harness-stripped)
        assert remaining_p == "world goodbye"

    def test_multi_chunk_drain_then_corrupt_chunk(self):
        """After partial consumption, a non-matching chunk passes through untouched."""
        # Chunk 1: partial match
        consumed1, rem_c1, rem_p = _consume_expected_ack_back_prefix("Hello", "Hello world")
        assert consumed1 == 5
        assert rem_c1 == ""
        assert rem_p == "world"

        # Chunk 2: corrupted, doesn't match remaining pending
        consumed2, rem_c2, rem_p2 = _consume_expected_ack_back_prefix("Xorld", rem_p)
        assert consumed2 == 0
        assert rem_c2 == "Xorld"
        assert rem_p2 == "world"

    def test_committed_longer_than_pending(self):
        """Committed extends beyond pending — partial match, remainder in committed."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix(
            "HelloworldExtra", "Hello world"
        )
        assert consumed > 0
        assert remaining_c == "Extra"
        assert remaining_p == ""

    def test_single_char_match(self):
        """A single-character match is sufficient to count as a real match."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("H", "Hello")
        assert consumed == 1
        assert remaining_c == ""
        assert remaining_p == "ello"

    # -- Unicode and accented characters --

    def test_accented_characters(self):
        """Accented characters should match exactly."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Héllocafé", "Héllo café")
        assert consumed == 9
        assert remaining_c == ""
        assert remaining_p == ""

    # -- Empty / degenerate inputs --

    def test_empty_committed_returns_zero(self):
        """Empty committed text means nothing to consume."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("", "Hello")
        assert consumed == 0
        assert remaining_c == ""
        assert remaining_p == "Hello"

    def test_empty_pending_returns_zero(self):
        """Empty pending text means nothing to match against."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hello", "")
        assert consumed == 0
        assert remaining_c == "Hello"
        assert remaining_p == ""

    def test_both_empty(self):
        """Both empty — no-op."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("", "")
        assert consumed == 0
        assert remaining_c == ""
        assert remaining_p == ""

    # -- Boundary between uninterruptible and interruptible in same chunk --

    def test_interruptible_suffix_preserved_after_uninterruptible_prefix(self):
        """When a chunk straddles the boundary, only the uninterruptible prefix is consumed."""
        # Agent emitted "Hold." (uninterruptible), then "So yes" (interruptible)
        # Harness bundles: "Hold.Soyes"
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hold.Soyes", "Hold.")
        assert consumed == 5  # "Hold." consumed
        assert remaining_c == "Soyes"
        assert remaining_p == ""

    def test_tts_injects_full_stop_at_boundary(self):
        """TTS injects a period between uninterruptible and interruptible text."""
        # Agent emitted "Hold" (uninterruptible), TTS adds ".", then interruptible text follows
        # Harness sends: "Hold.Soyes"
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix("Hold.Soyes", "Hold")
        # "Hold" matches real chars, trailing "." is swept
        assert consumed == 5  # "Hold" + "."
        assert remaining_c == "Soyes"
        assert remaining_p == ""

    # -- Full stop at chunk boundary after partial drain --

    def test_full_stop_only_chunk_after_pending_drained(self):
        """Once pending is empty, a full-stop-only chunk should pass through."""
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix(".", "")
        assert consumed == 0
        assert remaining_c == "."
        assert remaining_p == ""

    def test_full_stop_only_chunk_with_remaining_pending(self):
        """A full-stop-only ack-back must not consume from remaining pending text."""
        # This is the key adversarial case: pending has real text left,
        # but the ack-back is just TTS noise — must not falsely suppress.
        consumed, remaining_c, remaining_p = _consume_expected_ack_back_prefix(".", " world")
        assert consumed == 0
        assert remaining_c == "."
        assert remaining_p == " world"


# ============================================================
# Adversarial _suppress_ack_back integration tests
# ============================================================


class TestSuppressAckBackAdversarial:
    """Integration tests: adversarial scenarios through the runner's suppress method."""

    def test_pure_full_stop_ack_back_not_suppressed(self):
        """A lone '.' ack-back must not be suppressed even when pending text exists."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Hello world"
        result = runner._suppress_ack_back(AgentTextSent(content="."))

        assert result is not None
        assert result.content == "."
        # Pending must be untouched
        assert runner._suppression_buffer == "Hello world"

    def test_multi_chunk_with_interleaved_full_stop_noise(self):
        """Real chunks suppress correctly even if a full-stop-only chunk appears between them."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Hello world"

        # Chunk 1: real content, partially drains pending
        r1 = runner._suppress_ack_back(AgentTextSent(content="Hello"))
        assert r1 is None
        assert runner._suppression_buffer == "world"

        # Chunk 2: TTS noise — must pass through, not corrupt pending
        r2 = runner._suppress_ack_back(AgentTextSent(content="."))
        assert r2 is not None
        assert r2.content == "."
        assert runner._suppression_buffer == "world"

        # Chunk 3: real content, finishes draining
        r3 = runner._suppress_ack_back(AgentTextSent(content="world"))
        assert r3 is None
        assert runner._suppression_buffer == ""

    def test_unicode_ack_back_suppressed(self):
        """Accented/unicode uninterruptible text is suppressed correctly."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Héllo café"
        result = runner._suppress_ack_back(AgentTextSent(content="Héllocafé"))

        assert result is None
        assert runner._suppression_buffer == ""

    def test_emoji_in_pending_suppressed(self):
        """Emoji in pending (harness-stripped) doesn't prevent suppression."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Hello 👋 there"
        result = runner._suppress_ack_back(AgentTextSent(content="Hellothere"))

        assert result is None
        assert runner._suppression_buffer == ""

    def test_completely_unrelated_ack_back_passes_through(self):
        """An ack-back with no character overlap passes through untouched."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Hello"
        result = runner._suppress_ack_back(AgentTextSent(content="XYZ"))

        assert result is not None
        assert result.content == "XYZ"
        assert runner._suppression_buffer == "Hello"

    def test_event_id_preserved_on_trimmed_ack_back(self):
        """When only the prefix is suppressed, the returned event preserves the original event_id."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Legal."
        original = AgentTextSent(content="Legal.Hello")
        result = runner._suppress_ack_back(original)

        assert result is not None
        assert result.content == "Hello"
        assert result.event_id == original.event_id

    def test_repeated_suppression_drains_long_pending(self):
        """Multiple ack-back chunks progressively drain a long pending buffer."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "This call may be recorded for quality purposes"

        r1 = runner._suppress_ack_back(AgentTextSent(content="Thiscallmaybe"))
        assert r1 is None

        r2 = runner._suppress_ack_back(AgentTextSent(content="recordedforquality"))
        assert r2 is None

        r3 = runner._suppress_ack_back(AgentTextSent(content="purposes"))
        assert r3 is None
        assert runner._suppression_buffer == ""

    def test_trailing_tts_full_stop_with_interruptible_suffix(self):
        """TTS adds '.' at uninterruptible boundary, followed by interruptible content."""
        ws = create_mock_websocket()
        runner = ConversationRunner(ws, noop_agent, env)

        runner._suppression_buffer = "Hold"
        result = runner._suppress_ack_back(AgentTextSent(content="Hold.Anyway"))

        assert result is not None
        assert result.content == "Anyway"
        assert runner._suppression_buffer == ""


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

    def test_enable_multilingual_stt(self):
        """AgentEnableMultilingualSTT sets STT language to None, leaves TTS alone."""
        result = self._map(AgentEnableMultilingualSTT())
        assert result.stt is not None
        assert result.stt.language is None
        assert result.tts is None

    def test_all_none_defaults(self):
        """No fields set -> TTS language None, STT config None (no change)."""
        result = self._map(AgentUpdateCall())
        assert result.tts.language is None
        assert result.stt is None

    def test_update_tts_with_values(self):
        """AgentUpdateTTS sets TTS voice and language, leaves STT alone."""
        result = self._map(AgentUpdateTTS(voice_id="v1", language="fr"))
        assert result.tts.voice_id == "v1"
        assert result.tts.language == "fr"
        assert result.stt is None

    def test_update_tts_defaults(self):
        """AgentUpdateTTS with no fields -> TTS voice/language None, STT None."""
        result = self._map(AgentUpdateTTS())
        assert result.tts.voice_id is None
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
