"""
History - Manages conversation history for LlmAgent.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Union

from line.events import (
    AgentDtmfSent,
    AgentEndCall,
    AgentHandedOff,
    AgentSendDtmf,
    AgentSendText,
    AgentTextSent,
    AgentToolCalled,
    AgentToolReturned,
    AgentTurnEnded,
    AgentTurnStarted,
    AgentUpdateCall,
    CallEnded,
    CallStarted,
    CustomHistoryEntry,
    HistoryEvent,
    InputEvent,
    LogMessage,
    LogMetric,
    AgentTransferCall,
    OutputEvent,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    UserTurnStarted,
)

# Type for events stored in local history (OutputEvent or CustomHistoryEntry)
_LocalEvent = Union[OutputEvent, CustomHistoryEntry]

# Sentinel event_id used before the first process() call.
# _build_full_history prepends entries tagged with this ID at the start of the history.
_INIT_EVENT_ID = "__init__"


# ---------------------------------------------------------------------------
# Sentinels for sequence boundaries
# ---------------------------------------------------------------------------

class _SequenceBoundary:
    """Sentinel for start/end of the event sequence."""
    def __init__(self, name: str) -> None:
        self._name = name
    def __repr__(self) -> str:
        return self._name

_SEQUENCE_START = _SequenceBoundary("_SEQUENCE_START")

# Union type for anchors: either a real event or a boundary sentinel
_Anchor = Union[HistoryEvent, _SequenceBoundary]


# ---------------------------------------------------------------------------
# Mutation types (internal, stored in History._mutations)
# ---------------------------------------------------------------------------

@dataclass
class _InsertEntry:
    """Insert a CustomHistoryEntry before or after an anchor event."""
    entry: CustomHistoryEntry
    anchor: _Anchor
    position: Literal["before", "after"]


@dataclass
class _ReplaceSegment:
    """Replace a segment [start..end] inclusive with new events."""
    events: List[HistoryEvent]
    start: _Anchor
    end: _Anchor


_Mutation = Union[_InsertEntry, _ReplaceSegment]


# ---------------------------------------------------------------------------
# History class
# ---------------------------------------------------------------------------

class History:
    """Manages conversation history with lazy merge and mutation support.

    Public API:
        add_entry(content, role, *, before, after) — insert a custom entry
        update(events, *, start, end) — replace a segment of history
        __iter__ / __len__ — iterate/count merged history events

    Private API (called by LlmAgent):
        _set_input(input_history, current_event_id) — set merge sources
        _append_local(event) — append event tagged with current event id
        _append_local_with_event_id(event, event_id) — append with explicit id
        _current_event_id — property for the current triggering event id
    """

    def __init__(self) -> None:
        self._input_history: List[InputEvent] = []
        self._local_history: List[tuple[str, _LocalEvent]] = []
        self._current_event_id: str = _INIT_EVENT_ID
        self._cache: Optional[List[HistoryEvent]] = None
        self._mutations: List[_Mutation] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(
        self,
        content: str,
        role: Literal["system", "user"] = "system",
        *,
        before: Optional[HistoryEvent] = None,
        after: Optional[HistoryEvent] = None,
    ) -> None:
        """Insert a CustomHistoryEntry into history.

        Without before/after: inserts at the end (after _SEQUENCE_START).
        With before=X: inserts before X.
        With after=X: inserts after X.

        Raises ValueError if both before and after are specified, or if the
        anchor event is not found in the current history.
        """
        if before is not None and after is not None:
            raise ValueError("Cannot specify both 'before' and 'after'")

        entry = CustomHistoryEntry(content=content, role=role)
        merged = self._ensure_merged()

        if before is not None:
            anchor: _Anchor = before
            position: Literal["before", "after"] = "before"
        elif after is not None:
            anchor = after
            position = "after"
        else:
            anchor = merged[-1] if merged else _SEQUENCE_START
            position = "after"

        # Validate real anchors exist in the current history
        if not isinstance(anchor, _SequenceBoundary):
            if anchor not in merged:
                raise ValueError(f"Anchor event not found in history: {anchor}")

        self._mutations.append(_InsertEntry(entry=entry, anchor=anchor, position=position))
        self._cache = None

    def update(
        self,
        events: List[HistoryEvent],
        *,
        start: Optional[HistoryEvent] = None,
        end: Optional[HistoryEvent] = None,
    ) -> None:
        """Replace a segment of history with new events.

        Without start/end: prefixes the history (events appear before existing).
        With only start: replaces from start to end of sequence (eagerly resolved).
        With only end: replaces from _SEQUENCE_START through end inclusive.
        With both: replaces the segment [start..end] inclusive.

        Raises ValueError if start/end not found or end appears before start.
        """
        resolved_start: _Anchor = start if start is not None else _SEQUENCE_START
        resolved_end: _Anchor

        if end is not None:
            resolved_end = end
        elif start is not None:
            # start given, end omitted → eagerly resolve to last event
            merged = self._ensure_merged()
            resolved_end = merged[-1] if merged else _SEQUENCE_START
        else:
            # Neither given → prefix semantics
            resolved_end = _SEQUENCE_START

        # Validate real anchors exist in the current history
        merged = self._ensure_merged()
        if not isinstance(resolved_start, _SequenceBoundary):
            if resolved_start not in merged:
                raise ValueError(f"Start event not found in history: {resolved_start}")
        if not isinstance(resolved_end, _SequenceBoundary):
            if resolved_end not in merged:
                raise ValueError(f"End event not found in history: {resolved_end}")

        # Check ordering when both are real events
        if not isinstance(resolved_start, _SequenceBoundary) and not isinstance(resolved_end, _SequenceBoundary):
            start_idx = merged.index(resolved_start)
            end_idx = merged.index(resolved_end)
            if end_idx < start_idx:
                raise ValueError(
                    f"End event (index {end_idx}) appears before start event (index {start_idx})"
                )

        self._mutations.append(_ReplaceSegment(events=events, start=resolved_start, end=resolved_end))
        self._cache = None

    def __iter__(self) -> Iterator[HistoryEvent]:
        return iter(self._ensure_merged())

    def __len__(self) -> int:
        return len(self._ensure_merged())

    # ------------------------------------------------------------------
    # Private API (called by LlmAgent)
    # ------------------------------------------------------------------

    def _set_input(self, input_history: List[InputEvent], current_event_id: str) -> None:
        """Update merge sources from a new process() call."""
        self._input_history = input_history
        self._current_event_id = current_event_id
        self._cache = None  # invalidate cache

    def _append_local(self, event: _LocalEvent) -> None:
        """Append an event to local history tagged with the current event id."""
        self._local_history.append((self._current_event_id, event))
        self._cache = None  # invalidate cache

    def _append_local_with_event_id(self, event: _LocalEvent, event_id: str) -> None:
        """Append an event to local history with an explicit event id."""
        self._local_history.append((event_id, event))
        self._cache = None  # invalidate cache

    # ------------------------------------------------------------------
    # Internal: lazy rebuild
    # ------------------------------------------------------------------

    def _ensure_merged(self) -> List[HistoryEvent]:
        """Build or return cached merged history with mutations applied."""
        if self._cache is not None:
            return self._cache

        result = _build_full_history(
            self._input_history, self._local_history, self._current_event_id
        )

        for mutation in self._mutations:
            if isinstance(mutation, _InsertEntry):
                if mutation.anchor is _SEQUENCE_START:
                    # "after" _SEQUENCE_START → insert at beginning
                    result.insert(0, mutation.entry)
                else:
                    idx = result.index(mutation.anchor)
                    if mutation.position == "before":
                        result.insert(idx, mutation.entry)
                    else:
                        result.insert(idx + 1, mutation.entry)
            elif isinstance(mutation, _ReplaceSegment):
                # _SEQUENCE_START → virtual position before index 0
                start_idx = 0 if mutation.start is _SEQUENCE_START else result.index(mutation.start)
                end_excl = 0 if mutation.end is _SEQUENCE_START else result.index(mutation.end) + 1
                result[start_idx:end_excl] = mutation.events

        self._cache = result
        return result


# ---------------------------------------------------------------------------
# Merge algorithm (moved from llm_agent.py)
# ---------------------------------------------------------------------------

def _build_full_history(
    input_history: List[InputEvent],
    local_history: List[tuple[str, _LocalEvent]],
    current_event_id: str,
) -> List[HistoryEvent]:
    """
    We have a split brain situation, where "input_history" (maintained by the external harness) is the source
    of truth for what actually happened in the conversation, but "local_history" (maintained by the agent)
    contains additional events that we want to include in the history (tool calls, custom entries). we need
    to merge them together in a coherent way before building messages for the LLM.

    To that end, we have certain events that are "matchable". They show up in both input and local history,
    but the input version is considered "canonical"
    (it's what actually occured in the conversation)

    Invariants:
    1. Input history ordering is preserved exactly
    2. Every local non-matchable event appears exactly once in output
    3. Matchable locals appear after their triggering input event
    4. Non-matchable locals maintain their original relative order
    5. Matchable events use canonical (input) version
    6. Unmatched local matchables are dropped

    Algorithm:
    1) For each input event, if it triggered any output events, load a queue
    of those local output events.
    2) Unmatchable input events are output directly.
    3) Matchable input events are matched against the queue, with non-matchable locals (tool calls, custom
    entries) drained alongside their matched matchable.

    Examples:

    1) Simple turn with a tool call:

        input_history:  [UserTextSent("hi", id=A), AgentTextSent("hello", id=B)]
        local_history:  [(A, AgentToolCalled(...)), (A, AgentToolReturned(...)), (A, AgentSendText("hello"))]

        result: [UserTextSent("hi"), AgentToolCalled(...), AgentToolReturned(...), AgentTextSent("hello")]

        UserTextSent is non-matchable so it's emitted directly. AgentTextSent is matchable and matches
        the local AgentSendText("hello"), so the canonical input version is used. The non-matchable tool
        events are drained alongside it.

    2) Multi-turn with interleaved tool calls:

        input_history:  [UserTextSent("weather?", id=A), AgentTextSent("It's sunny", id=B),
                         UserTextSent("thanks", id=C), AgentTextSent("You're welcome", id=D)]
        local_history:  [(A, AgentToolCalled(weather)), (A, AgentToolReturned(weather)),
                         (A, AgentSendText("It's sunny")),
                         (C, AgentSendText("You're welcome"))]

        result: [UserTextSent("weather?"), AgentToolCalled(weather), AgentToolReturned(weather),
                 AgentTextSent("It's sunny"), UserTextSent("thanks"), AgentTextSent("You're welcome")]

        When we reach UserTextSent("thanks", id=C), its id is in local_by_event_id so we flush the
        remaining queue from the previous trigger (nothing left) and load the new queue. The tool events
        from the first turn are properly interleaved before the matched AgentTextSent.

    """
    # Split local history into prior and current based on event_id
    prior_local = [(eid, e) for eid, e in local_history if eid != current_event_id]
    current_local = [e for eid, e in local_history if eid == current_event_id]

    # Build map from event_id to list of responsive local events
    local_by_event_id: dict[str, List[_LocalEvent]] = defaultdict(list)
    for eid, event in prior_local:
        local_by_event_id[eid].append(event)

    # Prepend init entries (added before the first process() call)
    init_events = local_by_event_id.pop(_INIT_EVENT_ID, [])

    result: List[Union[InputEvent, _LocalEvent]] = list(init_events)
    queue: List[_LocalEvent] = []

    for input_evt in input_history:
        # If this input event generated any output events
        # flush old output events (they are semantically "prior" to this input event)
        # and load new queue of output events triggered by this input event
        if input_evt.event_id in local_by_event_id:
            result.extend(_flush_queue(queue))
            local_slice = local_by_event_id[input_evt.event_id]
            queue = _concat_contiguous_agent_send_text(local_slice)

        if not _is_input_matchable(input_evt):
            # Non-matchable input (UserTextSent, etc.) — emit directly
            result.append(input_evt)
        else:
            # Matchable input — match against queue
            emitted, queue = _match_matchable(input_evt, queue)
            result.extend(emitted)

    # Flush any remaining locals from the last trigger
    result.extend(_flush_queue(queue))

    # Append current-turn events (not yet observed, use local version)
    result.extend(current_local)

    # Convert matchable OutputEvents to InputEvent counterparts
    return [h for e in result if (h := _to_history_event(e)) is not None]


def _flush_queue(queue: List[_LocalEvent]) -> List[_LocalEvent]:
    """Return non-matchable events from queue, discarding unmatched matchables."""
    return [e for e in queue if not _is_local_matchable(e)]


def _split_leading_non_matchables(
    queue: List[_LocalEvent],
) -> tuple[List[_LocalEvent], List[_LocalEvent]]:
    """Split queue into leading non-matchables and the rest.

    Returns:
        (non_matchables, remaining) where remaining starts at the first matchable
        or is empty if none exist.
    """
    for i, event in enumerate(queue):
        if _is_local_matchable(event):
            return queue[:i], queue[i:]
    return queue, []


def _match_matchable(
    input_evt: InputEvent,
    queue: List[_LocalEvent],
) -> tuple[List[Union[InputEvent, _LocalEvent]], List[_LocalEvent]]:
    """Match a matchable input event against the local queue.

    Splits off leading non-matchables, then tries to match the head matchable.
    On match, emits canonical input version and splits off trailing non-matchables.
    On prefix match, also prepends the suffix to the remaining queue.
    On no match, discards the local matchable and retries.
    If queue exhausted with no match, emits input as-is.

    Returns:
        (emitted_events, remaining_queue)
    """
    remaining = queue
    emitted: List[Union[InputEvent, _LocalEvent]] = []

    while remaining:
        non_obs, rest = _split_leading_non_matchables(remaining)
        emitted.extend(non_obs)

        if not rest:
            break

        head_local = rest[0]
        match_result = _try_match_events(head_local, input_evt)

        if match_result is not None:
            matched_input, suffix_event = match_result
            emitted.append(matched_input)
            # Split off non-matchables that follow the matched matchable
            trailing_non_obs, after = _split_leading_non_matchables(rest[1:])
            emitted.extend(trailing_non_obs)
            # On prefix match, prepend suffix to remaining queue
            remaining = ([suffix_event] + after) if suffix_event is not None else after
            return emitted, remaining

        # No match — discard this local matchable and try next
        remaining = rest[1:]

    # Queue exhausted with no match — emit input as-is
    emitted.append(input_evt)
    return emitted, []


_HISTORY_EVENT_TYPES = (
    # InputEvent types
    CallStarted,
    CallEnded,
    AgentHandedOff,
    UserTurnStarted,
    UserDtmfSent,
    UserTextSent,
    UserTurnEnded,
    AgentTurnStarted,
    AgentTextSent,
    AgentDtmfSent,
    AgentTurnEnded,
    # Tool events
    AgentToolCalled,
    AgentToolReturned,
    # Custom entries
    CustomHistoryEntry,
)


def _to_history_event(event: object) -> Optional[HistoryEvent]:
    """Convert an event to a HistoryEvent.

    Matchable OutputEvents are converted to their InputEvent counterparts.
    Non-history OutputEvents (LogMetric, etc.) are filtered out (returns None).
    All other events (InputEvent, AgentToolCalled, AgentToolReturned, CustomHistoryEntry)
    pass through unchanged.
    """
    # Matchable OutputEvents → convert to InputEvent counterparts
    if isinstance(event, AgentSendText):
        return AgentTextSent(content=event.text)
    elif isinstance(event, AgentSendDtmf):
        return AgentDtmfSent(button=event.button)
    elif isinstance(event, AgentEndCall):
        return CallEnded()
    # HistoryEvent pass-through (tool events, custom entries)
    elif isinstance(event, (AgentToolCalled, AgentToolReturned, CustomHistoryEntry)):
        return event
    # InputEvent types pass through
    elif isinstance(
        event,
        (
            CallStarted,
            CallEnded,
            AgentHandedOff,
            UserTurnStarted,
            UserDtmfSent,
            UserTextSent,
            UserTurnEnded,
            AgentTurnStarted,
            AgentTextSent,
            AgentDtmfSent,
            AgentTurnEnded,
        ),
    ):
        return event
    # Non-history OutputEvents are filtered out
    elif isinstance(event, (AgentTransferCall, LogMetric, LogMessage, AgentUpdateCall)):
        return None
    else:
        raise ValueError(f"Unknown event type in history: {type(event).__name__}")


def _concat_contiguous_agent_send_text(local_history: List[_LocalEvent]) -> List[_LocalEvent]:
    """
    Since the LLM streams output, we likely will have many AgentSendText events that are part of the same
    logical "message" from the LLM. This concats them into a single AgentSendText event for easier matching
    to the input history, which only has one AgentTextSent per LLM message.
    """
    if not local_history:
        return []
    result: List[_LocalEvent] = []
    current = local_history[0]
    for event in local_history[1:]:
        if isinstance(current, AgentSendText) and isinstance(event, AgentSendText):
            current = AgentSendText(text=current.text + event.text)
        else:
            result.append(current)
            current = event
    result.append(current)
    return result


# Matchable OutputEvent types - these can be matched between local and input history
# Corresponds to events that the external system tracks/observes
MATCHABLE_OUTPUT_EVENT_TYPES = (
    AgentSendDtmf,  # => AgentDtmfSent
    AgentSendText,  # => AgentTextSent
    AgentEndCall,  # => CallEnded
)


def _is_local_matchable(event: _LocalEvent) -> bool:
    """Check if a local event is matchable (can be matched to input history)."""
    return isinstance(event, MATCHABLE_OUTPUT_EVENT_TYPES)


MATCHABLE_INPUT_EVENT_TYPES = (
    AgentDtmfSent,
    AgentTextSent,
    CallEnded,
)


def _is_input_matchable(event: InputEvent) -> bool:
    """Check if an InputEvent is matchable (can be matched to local history)."""
    return isinstance(event, MATCHABLE_INPUT_EVENT_TYPES)


def _try_match_events(
    local: _LocalEvent, input_evt: InputEvent
) -> Optional[tuple[InputEvent, Optional[_LocalEvent]]]:
    """Try to match a local matchable event to an input matchable event.

    Returns:
        None: No match
        (input_evt, None): Exact match - use input_evt as canonical
        (input_evt, suffix_event): Prefix match - use input_evt and carry forward suffix_event

    For text events, supports prefix matching (input is prefix of local).
    For DTMF and EndCall events, only exact matching is supported.
    """
    if isinstance(local, AgentSendText) and isinstance(input_evt, AgentTextSent):
        if local.text == input_evt.content:
            return (input_evt, None)
        if local.text.startswith(input_evt.content):
            suffix = local.text[len(input_evt.content):]
            return (input_evt, AgentSendText(text=suffix))
    elif isinstance(local, AgentSendDtmf) and isinstance(input_evt, AgentDtmfSent):
        if local.button == input_evt.button:
            return (input_evt, None)
    elif isinstance(local, AgentEndCall) and isinstance(input_evt, CallEnded):
        return (input_evt, None)
    return None
