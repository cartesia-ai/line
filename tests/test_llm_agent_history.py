"""
Tests for the History class and _build_full_history merge algorithm.

uv run pytest tests/test_llm_agent_history.py -v
"""

import pytest

from line.events import (
    AgentEndCall,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    CallEnded,
    CustomHistoryEntry,
    UserTextSent,
)
from line.llm_agent.history import History, _build_full_history

# Use anyio for async test support with asyncio backend only (trio not installed)
pytestmark = [pytest.mark.anyio, pytest.mark.parametrize("anyio_backend", ["asyncio"])]


# =============================================================================
# Tests: _build_full_history
# =============================================================================


class TestBuildFullHistory:
    """Tests for the _build_full_history function.

    The function merges input_history (canonical) with local_history using these rules:
    1. Input history is the source of truth for all events
    2. Local events are grouped by the event_id of the input that triggered them
    3. For input events with responsive local events, interpolation/matching is applied
    4. For input events without responsive local events, they pass through as-is
    5. Current local events (not yet observed) are appended at the end

    Observable output events: AgentSendText, AgentSendDtmf, AgentEndCall
    Observable input events: AgentTextSent, AgentDtmfSent, CallEnded

    local_history format: List[tuple[str, OutputEvent]] where str is the triggering event_id
    """

    @staticmethod
    def _annotate(events: list, event_id: str) -> list[tuple]:
        """Annotate events with a triggering event_id."""
        return [(event_id, e) for e in events]

    async def test_both_histories_empty(self):
        """When both histories are empty, return empty list."""
        result = _build_full_history([], [], current_event_id="current")
        assert result == []

    async def test_only_input_history_with_user_message(self):
        """When only input_history exists with non-observable event, include it."""
        input_history = [UserTextSent(content="Hello")]
        result = _build_full_history(input_history, [], current_event_id="current")

        assert len(result) == 1
        assert isinstance(result[0], UserTextSent)
        assert result[0].content == "Hello"

    async def test_only_input_history_with_observable_event(self):
        """When only input_history exists with observable event, include it (pass through)."""
        input_history = [AgentTextSent(content="Hi there")]
        result = _build_full_history(input_history, [], current_event_id="current")

        # Observable input with no responsive local passes through
        assert len(result) == 1
        assert isinstance(result[0], AgentTextSent)
        assert result[0].content == "Hi there"

    async def test_only_local_history_with_unobservable_event_current(self):
        """When only local_history exists with unobservable current event, include it."""
        # Current events (event_id == current_event_id) are appended
        local_history = self._annotate(
            [AgentToolCalled(tool_call_id="1", tool_name="test", tool_args={})],
            event_id="current",
        )
        result = _build_full_history([], local_history, current_event_id="current")

        assert len(result) == 1
        assert isinstance(result[0], AgentToolCalled)
        assert result[0].tool_name == "test"

    async def test_only_local_history_with_observable_event_prior_excluded(self):
        """Prior local observable events without matching input are excluded."""
        # Prior events (event_id != current_event_id) with no input to match
        local_history = self._annotate(
            [AgentSendText(text="Unmatched response")],
            event_id="prior",
        )
        result = _build_full_history([], local_history, current_event_id="current")

        # Observable prior local events without matching input events are excluded
        assert len(result) == 0

    async def test_matching_observable_events_uses_input_canonical(self):
        """When observable events match, use the input_history version (canonical)."""
        # Create input event and use its event_id for local
        input_evt = AgentTextSent(content="Hello world")
        input_history = [input_evt]
        local_history = self._annotate(
            [AgentSendText(text="Hello world")],
            event_id=input_evt.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 1
        # Should be the input_history version (AgentTextSent), not local (AgentSendText)
        assert isinstance(result[0], AgentTextSent)
        assert result[0].content == "Hello world"

    async def test_unobservable_local_event_interpolated_before_observable(self):
        """Unobservable local events appear before their following observable event."""
        # Input: [User0, Agent1] - User0 triggers local events
        user0 = UserTextSent(content="Question")
        input_history = [
            user0,
            AgentTextSent(content="Response"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="lookup", tool_args={"q": "test"}),
                AgentToolReturned(
                    tool_call_id="1", tool_name="lookup", tool_args={"q": "test"}, result="data"
                ),
                AgentSendText(text="Response"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Expected order: User, ToolCalled, ToolReturned, AgentTextSent
        assert len(result) == 4
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentTextSent)

    async def test_non_observable_input_event_included(self):
        """Non-observable input events (like UserTextSent) are always included."""
        user0 = UserTextSent(content="User question")
        input_history = [
            user0,
            AgentTextSent(content="Agent answer"),
        ]
        local_history = self._annotate(
            [AgentSendText(text="Agent answer")],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 2
        assert isinstance(result[0], UserTextSent)
        assert result[0].content == "User question"
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "Agent answer"

    async def test_unmatched_observable_local_event_excluded(self):
        """Observable local events without matching input events are excluded."""
        user0 = UserTextSent(content="Question")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentSendText(text="First response"),  # Observable, no match in input
                AgentSendText(text="Second response"),  # Observable, no match in input
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Only the user message should be included
        assert len(result) == 1
        assert isinstance(result[0], UserTextSent)

    async def test_complex_conversation_with_tools(self):
        """Test a realistic conversation with user messages, tool calls, and agent responses."""
        user0 = UserTextSent(content="What's the weather?")
        input_history = [
            user0,
            AgentTextSent(content="The weather is sunny."),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="get_weather", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="get_weather", tool_args={}, result="sunny"),
                AgentSendText(text="The weather is sunny."),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Expected: UserText, ToolCalled, ToolReturned, AgentTextSent (canonical)
        assert len(result) == 4
        assert isinstance(result[0], UserTextSent)
        assert result[0].content == "What's the weather?"
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentTextSent)
        assert result[3].content == "The weather is sunny."

    async def test_multiple_tool_calls_interleaved(self):
        """Test multiple tool calls with responses interleaved."""
        user0 = UserTextSent(content="Get weather and time")
        input_history = [
            user0,
            AgentTextSent(content="Weather is sunny, time is 3pm."),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="get_weather", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="get_weather", tool_args={}, result="sunny"),
                AgentToolCalled(tool_call_id="2", tool_name="get_time", tool_args={}),
                AgentToolReturned(tool_call_id="2", tool_name="get_time", tool_args={}, result="3pm"),
                AgentSendText(text="Weather is sunny, time is 3pm."),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # All unobservable events interpolated, then the matching observable
        assert len(result) == 6
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert result[1].tool_name == "get_weather"
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentToolCalled)
        assert result[3].tool_name == "get_time"
        assert isinstance(result[4], AgentToolReturned)
        assert isinstance(result[5], AgentTextSent)

    async def test_multiple_matching_observable_events(self):
        """Test conversation with multiple turns, each with matching observables."""
        user0 = UserTextSent(content="Hi")
        user2 = UserTextSent(content="How are you?")
        input_history = [
            user0,
            AgentTextSent(content="Hello!"),
            user2,
            AgentTextSent(content="I'm good!"),
        ]
        # First turn (user0) triggers Hello, second turn (user2) triggers I'm good
        local_history = self._annotate(
            [AgentSendText(text="Hello!")],
            event_id=user0.event_id,
        ) + self._annotate(
            [AgentSendText(text="I'm good!")],
            event_id=user2.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 4
        assert isinstance(result[0], UserTextSent)
        assert result[0].content == "Hi"
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "Hello!"
        assert isinstance(result[2], UserTextSent)
        assert result[2].content == "How are you?"
        assert isinstance(result[3], AgentTextSent)
        assert result[3].content == "I'm good!"

    async def test_unobservable_events_between_observables(self):
        """Test unobservable events appear in correct position between observables."""
        user0 = UserTextSent(content="Start")
        user2 = UserTextSent(content="User interjects")
        input_history = [
            user0,
            AgentTextSent(content="First"),
            user2,
            AgentTextSent(content="Second"),
        ]
        # user0 triggers First + tools, user2 triggers Second
        local_history = self._annotate(
            [
                AgentSendText(text="First"),
                AgentToolCalled(tool_call_id="1", tool_name="middle_tool", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="middle_tool", tool_args={}, result="done"),
            ],
            event_id=user0.event_id,
        ) + self._annotate(
            [AgentSendText(text="Second")],
            event_id=user2.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User0, First, tools (drained after First), User2, Second
        assert len(result) == 6
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "First"
        assert isinstance(result[2], AgentToolCalled)
        assert isinstance(result[3], AgentToolReturned)
        assert isinstance(result[4], UserTextSent)
        assert isinstance(result[5], AgentTextSent)
        assert result[5].content == "Second"

    async def test_input_observable_without_local_match_passes_through(self):
        """Input observable events pass through when no local responds to them.

        This happens when input_history has events from previous agents or turns
        that the current agent instance didn't generate.
        """
        input_history = [
            AgentTextSent(content="Previous agent said this"),
            UserTextSent(content="User reply"),
        ]
        local_history = []  # This agent hasn't generated anything yet

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Both events pass through (no local to match/interpolate)
        assert len(result) == 2
        assert isinstance(result[0], AgentTextSent)
        assert isinstance(result[1], UserTextSent)

    async def test_partial_match_only_matching_local_observable_kept(self):
        """When local has more observables than input, only matching ones are kept."""
        user0 = UserTextSent(content="Question")
        input_history = [
            user0,
            AgentTextSent(content="Answer"),
        ]
        local_history = self._annotate(
            [
                AgentSendText(text="Partial..."),  # This was streamed but not in input yet
                AgentSendText(text="Answer"),  # This matches
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # "Partial..." should be excluded (no match), "Answer" should match
        assert len(result) == 2
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "Answer"

    async def test_empty_local_with_full_input_history(self):
        """When local is empty, full input history passes through."""
        input_history = [
            UserTextSent(content="First"),
            AgentTextSent(content="Response 1"),
            UserTextSent(content="Second"),
            AgentTextSent(content="Response 2"),
        ]
        local_history = []

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # All input events pass through (no local to match/interpolate)
        assert len(result) == 4
        assert all(isinstance(r, (UserTextSent, AgentTextSent)) for r in result)

    async def test_mismatched_text_content_not_matched(self):
        """Observable events with different content don't match."""
        agent0 = AgentTextSent(content="Hello")
        input_history = [agent0]
        local_history = self._annotate(
            [AgentSendText(text="Goodbye")],  # Different content
            event_id=agent0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # local event is excluded (no match), input event is matched with slice
        # but no match found, so slice produces nothing for this observable
        # Since local responds to agent0, we process slice with local
        # but they don't match, so local is skipped and input passes through
        assert len(result) == 1
        assert isinstance(result[0], AgentTextSent)
        assert result[0].content == "Hello"

    async def test_all_unobservable_local_history(self):
        """When local history contains only unobservable events, all are included."""
        user0 = UserTextSent(content="Do something")
        input_history = [user0]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
                AgentToolCalled(tool_call_id="2", tool_name="tool2", tool_args={}),
                AgentToolReturned(tool_call_id="2", tool_name="tool2", tool_args={}, result="r2"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentToolCalled)
        assert isinstance(result[4], AgentToolReturned)

    async def test_agent_end_call_observable_matching(self):
        """Test that AgentEndCall matches with CallEnded."""
        user0 = UserTextSent(content="Goodbye")
        input_history = [
            user0,
            CallEnded(),
        ]
        local_history = self._annotate(
            [
                AgentSendText(text="Bye!"),  # Unmatched, will be excluded
                AgentEndCall(),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User message + CallEnded (canonical), AgentSendText excluded (no match)
        assert len(result) == 2
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], CallEnded)

    # =========================================================================
    # Tests: Prefix Matching
    # =========================================================================

    async def test_prefix_match_input_is_prefix_of_local(self):
        """When input text is a prefix of local text, match and carry forward suffix."""
        user0 = UserTextSent(content="Question")
        input_history = [
            user0,
            AgentTextSent(content="Hello"),  # Prefix of "Hello world!"
        ]
        local_history = self._annotate(
            [AgentSendText(text="Hello world!")],  # Has suffix " world!"
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Should include user message and the canonical input (prefix match)
        # The suffix " world!" should be excluded since there's no matching input for it
        assert len(result) == 2
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "Hello"

    async def test_prefix_match_suffix_matches_next_input(self):
        """Suffix from prefix match can match the next input event."""
        # user0 triggers local, slice includes [User0, Agent1, User2, Agent3]
        user0 = UserTextSent(content="Start")
        input_history = [
            user0,
            AgentTextSent(content="Hello"),  # Prefix of "Hello world!"
            UserTextSent(content="..."),
            AgentTextSent(content=" world!"),  # Matches carried suffix
        ]
        local_history = self._annotate(
            [AgentSendText(text="Hello world!")],  # Will be split via prefix match
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User0, Hello (prefix match), user message, then " world!" matches suffix
        assert len(result) == 4
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "Hello"
        assert isinstance(result[2], UserTextSent)
        assert isinstance(result[3], AgentTextSent)
        assert result[3].content == " world!"

    async def test_prefix_match_with_tool_calls_between(self):
        """Prefix matching works when tool calls precede the text."""
        user0 = UserTextSent(content="Get weather")
        input_history = [
            user0,
            AgentTextSent(content="The weather "),
            AgentTextSent(content="is sunny today!"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="weather", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="weather", tool_args={}, result="sunny"),
                AgentSendText(text="The weather is sunny today!"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User, ToolCalled, ToolReturned, then two text events (prefix match then suffix)
        assert len(result) == 5
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentTextSent)
        assert result[3].content == "The weather "
        assert isinstance(result[4], AgentTextSent)
        assert result[4].content == "is sunny today!"

    async def test_prefix_match_single_input_split(self):
        """Single input text that is prefix of local text - suffix is dropped."""
        agent0 = AgentTextSent(content="Hello")  # Prefix of "Hello world!"
        input_history = [agent0]
        local_history = self._annotate(
            [AgentSendText(text="Hello world!")],
            event_id=agent0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Only the input (prefix) is included, suffix " world!" is dropped (no matching input)
        assert len(result) == 1
        assert isinstance(result[0], AgentTextSent)
        assert result[0].content == "Hello"

    async def test_prefix_match_multiple_splits_with_separators(self):
        """Local text split across multiple non-contiguous input events."""
        user0 = UserTextSent(content="Start")
        input_history = [
            user0,
            AgentTextSent(content="A"),
            UserTextSent(content="u1"),
            AgentTextSent(content="B"),
            UserTextSent(content="u2"),
            AgentTextSent(content="C"),
        ]
        local_history = self._annotate(
            [AgentSendText(text="ABC")],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # User0, A (prefix), u1, B (prefix of "BC"), u2, C (exact match)
        assert len(result) == 6
        assert isinstance(result[0], UserTextSent)
        assert result[1].content == "A"
        assert isinstance(result[2], UserTextSent)
        assert result[3].content == "B"
        assert isinstance(result[4], UserTextSent)
        assert result[5].content == "C"

    async def test_preprocessing_local_longer_than_input(self):
        """Local text is longer than input after preprocessing - prefix match."""
        agent0 = AgentTextSent(content="Hello")  # Single event
        input_history = [agent0]
        local_history = self._annotate(
            [
                # These get concatenated to "Hello world!"
                AgentSendText(text="Hello"),
                AgentSendText(text=" world!"),
            ],
            event_id=agent0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Input "Hello" is prefix of local "Hello world!"
        # Suffix " world!" has no matching input, so dropped
        assert len(result) == 1
        assert isinstance(result[0], AgentTextSent)
        assert result[0].content == "Hello"

    async def test_pre_tool_call_result_grouped_with_agent_text(self):
        """Tool call and result are grouped together in the final history."""
        user0 = UserTextSent(content="Question1")
        input_history = [
            user0,
            AgentTextSent(content="Response1"),
            UserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
                AgentSendText(text="Response1"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentTextSent)
        assert isinstance(result[4], UserTextSent)

    # ==== TOOL CALL ORDERING ====

    async def test_circum_tool_call_result_grouped_with_agent_text(self):
        """Tool call appears before text, tool result after due to local ordering."""
        user0 = UserTextSent(content="Question1")
        input_history = [
            user0,
            AgentTextSent(content="Response1"),
            UserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentSendText(text="Response1 and more"),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentTextSent)
        assert isinstance(result[3], AgentToolReturned)
        assert isinstance(result[4], UserTextSent)

    async def test_pre_tool_call_result_grouped_with_trimmed_agent_text(self):
        """Tool call and result are grouped together in the final history."""
        user0 = UserTextSent(content="Question1")
        input_history = [
            user0,
            AgentTextSent(content="Response1"),
            UserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
                AgentSendText(text="Response1 and more"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        assert len(result) == 5
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentToolReturned)
        assert isinstance(result[3], AgentTextSent)
        assert isinstance(result[4], UserTextSent)

    async def test_circum_tool_call_result_grouped_with_trimmed_agent_text(self):
        """Tool call appears before text, tool result after due to local ordering."""
        user0 = UserTextSent(content="Question1")
        input_history = [
            user0,
            AgentTextSent(content="Response1"),
            UserTextSent(content="Question2"),
        ]
        local_history = self._annotate(
            [
                AgentToolCalled(tool_call_id="1", tool_name="tool1", tool_args={}),
                AgentSendText(text="Response1"),
                AgentToolReturned(tool_call_id="1", tool_name="tool1", tool_args={}, result="r1"),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Tool call interpolated around matched text
        assert len(result) == 5
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], AgentToolCalled)
        assert isinstance(result[2], AgentTextSent)
        assert isinstance(result[3], AgentToolReturned)
        assert isinstance(result[4], UserTextSent)

    async def test_tool_call_after_text_not_orphaned_by_subsequent_user_messages(self):
        """Tool calls after text in the same generation stay grouped with the text.

        Regression test: when agent generates text + tool call in the same turn,
        and user sends messages afterward, the tool call events should appear right
        after the agent's text, not pushed to the end after the user's messages.

        Reproduces the bug where end_call tool was orphaned from the "goodbye!" message:
          assistant: "goodbye!"
          user: "bye"
          user: "bye"
          assistant: end_call    <-- wrong, should be right after "goodbye!"
          tool: success
        """
        user0 = UserTextSent(content="Nope")
        input_history = [
            user0,
            AgentTextSent(content="Thanks for choosing Flighty, goodbye!"),
            UserTextSent(content="All right, bye."),
            UserTextSent(content="Bye."),
        ]
        local_history = self._annotate(
            [
                AgentSendText(text="Thanks for choosing Flighty, goodbye!"),
                AgentToolCalled(tool_call_id="call_end", tool_name="end_call", tool_args={}),
                AgentToolReturned(
                    tool_call_id="call_end", tool_name="end_call", tool_args={}, result="success"
                ),
            ],
            event_id=user0.event_id,
        )

        result = _build_full_history(input_history, local_history, current_event_id="current")

        # Expected order: User, Goodbye text, ToolCalled, ToolReturned, User1, User2
        # NOT: User, Goodbye text, User1, User2, ToolCalled, ToolReturned
        assert len(result) == 6
        assert isinstance(result[0], UserTextSent)
        assert result[0].content == "Nope"
        assert isinstance(result[1], AgentTextSent)
        assert result[1].content == "Thanks for choosing Flighty, goodbye!"
        assert isinstance(result[2], AgentToolCalled)
        assert result[2].tool_name == "end_call"
        assert isinstance(result[3], AgentToolReturned)
        assert result[3].tool_name == "end_call"
        assert isinstance(result[4], UserTextSent)
        assert result[4].content == "All right, bye."
        assert isinstance(result[5], UserTextSent)
        assert result[5].content == "Bye."


# =============================================================================
# Tests: History class public API
# =============================================================================


class TestHistory:
    """Tests for the History class public API: add_entry positioning, update segments, iteration."""

    def _make_history(self, input_events, local_events=None, current_event_id="current"):
        """Create a History and populate it with input/local events."""
        h = History()
        local = []
        if local_events:
            for eid, evt in local_events:
                local.append((eid, evt))
        h._input_history = input_events
        h._local_history = local
        h._current_event_id = current_event_id
        return h

    # ------------------------------------------------------------------
    # __iter__ / __len__
    # ------------------------------------------------------------------

    async def test_iter_empty(self):
        h = History()
        assert list(h) == []
        assert len(h) == 0

    async def test_iter_with_input(self):
        h = self._make_history([UserTextSent(content="hi"), AgentTextSent(content="hello")])
        events = list(h)
        assert len(events) == 2
        assert isinstance(events[0], UserTextSent)
        assert isinstance(events[1], AgentTextSent)

    async def test_len_matches_iter(self):
        h = self._make_history([UserTextSent(content="a"), UserTextSent(content="b")])
        assert len(h) == len(list(h))

    # ------------------------------------------------------------------
    # add_entry (no positioning)
    # ------------------------------------------------------------------

    async def test_add_entry_stores_mutation(self):
        h = History()
        h.add_entry("some context")
        assert len(h._mutations) == 1
        result = list(h)
        assert len(result) == 1
        assert isinstance(result[0], CustomHistoryEntry)
        assert result[0].content == "some context"
        assert result[0].role == "user"

    async def test_add_entry_with_user_role(self):
        h = History()
        h.add_entry("user injected", role="user")
        result = list(h)
        assert len(result) == 1
        assert result[0].role == "user"

    async def test_add_entry_invalidates_cache(self):
        h = self._make_history([UserTextSent(content="hi")])
        _ = list(h)  # populate cache
        assert h._cache is not None
        h.add_entry("new")
        assert h._cache is None

    # ------------------------------------------------------------------
    # add_entry(before=...)
    # ------------------------------------------------------------------

    async def test_add_entry_before(self):
        user_evt = UserTextSent(content="hello")
        agent_evt = AgentTextSent(content="world")
        h = self._make_history([user_evt, agent_evt])

        # The merge produces these exact objects, so grab them
        merged = list(h)
        assert len(merged) == 2

        h.add_entry("injected", before=merged[1])
        result = list(h)

        assert len(result) == 3
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], CustomHistoryEntry)
        assert result[1].content == "injected"
        assert isinstance(result[2], AgentTextSent)

    async def test_add_entry_before_first(self):
        user_evt = UserTextSent(content="hello")
        h = self._make_history([user_evt])
        merged = list(h)

        h.add_entry("prefix", before=merged[0])
        result = list(h)

        assert len(result) == 2
        assert isinstance(result[0], CustomHistoryEntry)
        assert result[0].content == "prefix"
        assert isinstance(result[1], UserTextSent)

    # ------------------------------------------------------------------
    # add_entry(after=...)
    # ------------------------------------------------------------------

    async def test_add_entry_after(self):
        user_evt = UserTextSent(content="hello")
        agent_evt = AgentTextSent(content="world")
        h = self._make_history([user_evt, agent_evt])
        merged = list(h)

        h.add_entry("suffix", after=merged[0])
        result = list(h)

        assert len(result) == 3
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], CustomHistoryEntry)
        assert result[1].content == "suffix"
        assert isinstance(result[2], AgentTextSent)

    async def test_add_entry_after_last(self):
        user_evt = UserTextSent(content="hello")
        h = self._make_history([user_evt])
        merged = list(h)

        h.add_entry("epilogue", after=merged[-1])
        result = list(h)

        assert len(result) == 2
        assert isinstance(result[0], UserTextSent)
        assert isinstance(result[1], CustomHistoryEntry)
        assert result[1].content == "epilogue"

    # ------------------------------------------------------------------
    # add_entry error cases
    # ------------------------------------------------------------------

    async def test_add_entry_both_before_and_after_raises(self):
        user_evt = UserTextSent(content="hello")
        h = self._make_history([user_evt])
        merged = list(h)

        with pytest.raises(ValueError, match="Cannot specify both"):
            h.add_entry("bad", before=merged[0], after=merged[0])

    async def test_add_entry_before_missing_anchor_raises(self):
        h = self._make_history([UserTextSent(content="hello")])
        _ = list(h)  # populate

        phantom = UserTextSent(content="not in history")
        with pytest.raises(ValueError, match="Anchor event not found"):
            h.add_entry("bad", before=phantom)

    async def test_add_entry_after_missing_anchor_raises(self):
        h = self._make_history([UserTextSent(content="hello")])
        _ = list(h)

        phantom = UserTextSent(content="not in history")
        with pytest.raises(ValueError, match="Anchor event not found"):
            h.add_entry("bad", after=phantom)

    # ------------------------------------------------------------------
    # update (no start/end — prefix semantics)
    # ------------------------------------------------------------------

    async def test_update_prefixes_history(self):
        h = self._make_history(
            [
                UserTextSent(content="a"),
                AgentTextSent(content="b"),
            ]
        )
        _ = list(h)  # populate

        h.update([UserTextSent(content="x")])
        result = list(h)

        assert len(result) == 3
        assert result[0].content == "x"
        assert result[1].content == "a"
        assert result[2].content == "b"

    async def test_update_on_empty_history_sets_initial(self):
        h = History()
        h.update([UserTextSent(content="x")])
        result = list(h)

        assert len(result) == 1
        assert result[0].content == "x"

    # ------------------------------------------------------------------
    # update(start=..., end=...)
    # ------------------------------------------------------------------

    async def test_update_segment(self):
        """Replace a middle segment, preserving events before and after."""
        events = [
            UserTextSent(content="a"),
            AgentTextSent(content="b"),
            UserTextSent(content="c"),
        ]
        h = self._make_history(events)
        merged = list(h)

        # Replace just the middle event
        h.update([CustomHistoryEntry(content="replaced")], start=merged[1], end=merged[1])
        result = list(h)

        assert len(result) == 3
        assert result[0].content == "a"
        assert isinstance(result[1], CustomHistoryEntry)
        assert result[1].content == "replaced"
        assert result[2].content == "c"

    async def test_update_segment_multi(self):
        """Replace a range of events."""
        events = [
            UserTextSent(content="a"),
            AgentTextSent(content="b"),
            UserTextSent(content="c"),
            AgentTextSent(content="d"),
        ]
        h = self._make_history(events)
        merged = list(h)

        # Replace b and c with a single event
        h.update([CustomHistoryEntry(content="bc")], start=merged[1], end=merged[2])
        result = list(h)

        assert len(result) == 3
        assert result[0].content == "a"
        assert isinstance(result[1], CustomHistoryEntry)
        assert result[1].content == "bc"
        assert result[2].content == "d"

    async def test_update_with_only_start(self):
        """When only start is given, end defaults to last event."""
        events = [
            UserTextSent(content="a"),
            AgentTextSent(content="b"),
            UserTextSent(content="c"),
        ]
        h = self._make_history(events)
        merged = list(h)

        h.update([CustomHistoryEntry(content="tail")], start=merged[1])
        result = list(h)

        assert len(result) == 2
        assert result[0].content == "a"
        assert result[1].content == "tail"

    async def test_update_with_only_end(self):
        """When only end is given, start defaults to first event."""
        events = [
            UserTextSent(content="a"),
            AgentTextSent(content="b"),
            UserTextSent(content="c"),
        ]
        h = self._make_history(events)
        merged = list(h)

        h.update([CustomHistoryEntry(content="head")], end=merged[1])
        result = list(h)

        assert len(result) == 2
        assert result[0].content == "head"
        assert result[1].content == "c"

    # ------------------------------------------------------------------
    # update error cases
    # ------------------------------------------------------------------

    async def test_update_start_not_found_raises(self):
        h = self._make_history([UserTextSent(content="a")])
        _ = list(h)

        phantom = UserTextSent(content="nope")
        with pytest.raises(ValueError, match="Start event not found"):
            h.update([], start=phantom)

    async def test_update_end_not_found_raises(self):
        h = self._make_history([UserTextSent(content="a")])
        _ = list(h)

        phantom = UserTextSent(content="nope")
        with pytest.raises(ValueError, match="End event not found"):
            h.update([], end=phantom)

    async def test_update_end_before_start_raises(self):
        events = [
            UserTextSent(content="a"),
            AgentTextSent(content="b"),
        ]
        h = self._make_history(events)
        merged = list(h)

        with pytest.raises(ValueError, match="appears before start"):
            h.update([], start=merged[1], end=merged[0])

    # ------------------------------------------------------------------
    # Multiple mutations compose
    # ------------------------------------------------------------------

    async def test_multiple_add_entry_mutations(self):
        h = self._make_history([UserTextSent(content="a"), AgentTextSent(content="b")])
        merged = list(h)

        h.add_entry("before-b", before=merged[1])
        h.add_entry("after-a", after=merged[0])
        result = list(h)

        contents = [e.content for e in result]
        assert contents == ["a", "after-a", "before-b", "b"]

    async def test_add_entry_then_update(self):
        h = self._make_history(
            [
                UserTextSent(content="a"),
                AgentTextSent(content="b"),
                UserTextSent(content="c"),
            ]
        )
        merged = list(h)

        # First add an entry after "a"
        h.add_entry("injected", after=merged[0])

        # Then replace "b" with something else
        h.update([CustomHistoryEntry(content="replaced")], start=merged[1], end=merged[1])

        result = list(h)
        contents = [e.content for e in result]
        assert contents == ["a", "injected", "replaced", "c"]

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    async def test_set_input_invalidates_cache(self):
        h = self._make_history([UserTextSent(content="old")])
        _ = list(h)
        assert h._cache is not None

        h._set_input([UserTextSent(content="new")], "new-eid")
        assert h._cache is None

        result = list(h)
        assert len(result) == 1
        assert result[0].content == "new"

    async def test_append_local_invalidates_cache(self):
        h = self._make_history([UserTextSent(content="hi")])
        _ = list(h)
        assert h._cache is not None

        h._append_local(CustomHistoryEntry(content="injected"))
        assert h._cache is None

    async def test_mutations_survive_cache_rebuild(self):
        """Mutations are replayed on every rebuild when anchor events still exist."""
        user_a = UserTextSent(content="a")
        agent_b = AgentTextSent(content="b")
        h = self._make_history([user_a, agent_b])
        merged = list(h)

        # Add a mutation anchored to agent_b
        h.add_entry("injected", before=merged[1])
        result1 = list(h)
        assert len(result1) == 3

        # Append a local event — invalidates cache but keeps same input objects
        h._append_local(CustomHistoryEntry(content="local"))
        result2 = list(h)
        # Should have: a, injected, b, local
        assert len(result2) == 4
        assert any(isinstance(e, CustomHistoryEntry) and e.content == "injected" for e in result2)
