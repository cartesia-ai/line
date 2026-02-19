"""Unified, mutable conversation history for LlmAgent."""

import inspect
import traceback
from typing import (
    Awaitable,
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

from loguru import logger

from line.events import CustomHistoryEntry, HistoryEvent, InputEvent, OutputEvent

# Type for events stored in local history (OutputEvent or CustomHistoryEntry)
_LocalEvent = Union[OutputEvent, CustomHistoryEntry]

# Sentinel event_id used before the first process() call.
_INIT_EVENT_ID = "__init__"


class History:
    """Unified, mutable conversation history.

    Provides read, write, and transform APIs for the conversation history
    managed by LlmAgent.

    Read:
        events      - merged canonical + local history
        iter/len    - list-like interface

    Write:
        append()    - persistent insertion
        replace()   - persistent replacement (auto-appends new turns)

    Transform:
        set_processor() - ongoing transformation (sync or async)

    Build (used by LlmAgent._build_messages):
        build()     - returns processed history for LLM
    """

    def __init__(self) -> None:
        # Canonical history from the external harness (set via _sync)
        self._canonical: List[InputEvent] = []
        # Local events annotated with (triggering_event_id, event)
        self._local: List[tuple[str, _LocalEvent]] = []
        # Event ID of the current triggering input event
        self._current_event_id: str = _INIT_EVENT_ID
        # Registered history transform
        self._process_history_fn: Optional[
            Callable[[List[HistoryEvent]], Union[List[HistoryEvent], Awaitable[List[HistoryEvent]]]]
        ] = None
        # Replace override state
        self._override_events: Optional[List[HistoryEvent]] = None
        self._override_watermark: int = 0

    # --- Read ---

    @property
    def events(self) -> List[HistoryEvent]:
        """Merged canonical + local history.

        If replace() was called, returns the override events plus any new
        canonical events that arrived after the replacement.
        """
        from line.llm_agent.llm_agent import _build_full_history

        if self._override_events is not None:
            new_canonical = self._canonical[self._override_watermark :]
            if new_canonical or self._local:
                new_history = _build_full_history(new_canonical, self._local, self._current_event_id)
                return list(self._override_events) + new_history
            return list(self._override_events)

        return _build_full_history(self._canonical, self._local, self._current_event_id)

    def __iter__(self) -> Iterator[HistoryEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    # --- Write ---

    def append(self, content: str, role: Literal["system", "user"] = "system") -> None:
        """Insert a CustomHistoryEntry into local history (persistent insertion).

        The entry appears as a message with the given role in the LLM conversation.
        """
        event = CustomHistoryEntry(content=content, role=role)
        self._append_local(event)

    def replace(self, events: List[HistoryEvent]) -> None:
        """Replace the conversation history (persistent replacement).

        Stores the provided events as the new baseline. Any new canonical events
        that arrive after this call (via _sync) are automatically appended.
        Local history is cleared.
        """
        self._override_events = list(events)
        self._override_watermark = len(self._canonical)
        self._local = []

    # --- Transform ---

    def set_processor(
        self,
        fn: Callable[[List[HistoryEvent]], Union[List[HistoryEvent], Awaitable[List[HistoryEvent]]]],
    ) -> None:
        """Register a transform that processes the history before message building.

        The transform receives the full history and can filter, reorder, or inject events.
        Both sync and async callables are supported.
        """
        self._process_history_fn = fn

    # --- Build ---

    async def build(self, *, context: Union[str, List[HistoryEvent], None] = None) -> List[HistoryEvent]:
        """Build the processed history for the LLM.

        Pipeline: events -> processor -> context append.
        Context is appended AFTER the processor (processor cannot strip it).
        """
        from line.llm_agent.llm_agent import _validate_processed_history

        full_history = self.events

        # Apply registered history transform
        if self._process_history_fn is not None:
            try:
                result = self._process_history_fn(full_history)
                if inspect.isawaitable(result):
                    result = await result
                full_history = _validate_processed_history(result)
            except Exception:
                logger.error(
                    f"History processor failed, using unprocessed history:\n{traceback.format_exc(limit=-10)}"
                )

        # Append context AFTER processor
        if context is not None:
            if isinstance(context, str):
                full_history = list(full_history) + [CustomHistoryEntry(content=context)]
            elif isinstance(context, list):
                full_history = list(full_history) + list(context)

        return full_history

    # --- Internal (used by LlmAgent) ---

    def _sync(self, canonical: List[InputEvent], event_id: str) -> None:
        """Sync canonical events from event.history."""
        self._canonical = canonical
        self._current_event_id = event_id

    def _append_local(self, event: _LocalEvent) -> None:
        """Append an event to local history, annotated with the current event_id."""
        self._local.append((self._current_event_id, event))

    def _append_local_with_id(self, event: _LocalEvent, event_id: str) -> None:
        """Append an event to local history with a specific event_id.

        Used by background tasks that captured the event_id at start time.
        """
        self._local.append((event_id, event))
