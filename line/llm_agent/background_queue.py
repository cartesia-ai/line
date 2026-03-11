"""BackgroundQueue - a cancellation-safe queue for async iterables."""

import asyncio
from collections import deque
from typing import AsyncIterable, Generic, List, Optional, TypeVar

T = TypeVar("T")


class BackgroundQueue(Generic[T]):
    """A queue that consumes multiple async iterables and buffers their items.

    Sources run as independent tasks — cancelling the consumer of get()
    does not cancel the source tasks, since get() only awaits an Event,
    not the tasks themselves.

    Interface:
        subscribe(source)  — register an AsyncIterable as a source
        get_nowait() -> List[T]  — drain all buffered items (non-blocking)
        get() -> T | None  — wait for next item; None means all sources done + empty
        is_active -> bool  — True if items buffered or sources still running
        wait() -> None  — wait for all sources to complete; raises on source error
    """

    def __init__(self) -> None:
        self._buffer: deque[T] = deque()
        self._notify: asyncio.Event = asyncio.Event()
        self._sources: set[asyncio.Task[None]] = set()
        self._error: Optional[BaseException] = None

    def subscribe(self, source: AsyncIterable[T]) -> None:
        """Register an async iterable as a source.

        Items yielded by the source are buffered for later consumption via
        get() or get_nowait(). If the source raises, the error is stored and
        re-raised from the next get() or wait() call.
        """

        async def _consume() -> None:
            async for item in source:
                self._buffer.append(item)
                self._notify.set()

        task = asyncio.create_task(_consume())
        self._sources.add(task)
        task.add_done_callback(self._on_source_done)

    def _on_source_done(self, task: asyncio.Task[None]) -> None:
        self._sources.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._error = task.exception()
        # Wake up get() so it can re-check the done/error condition
        self._notify.set()

    def get_nowait(self) -> List[T]:
        """Drain and return all currently buffered items (non-blocking)."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    async def get(self) -> Optional[T]:
        """Wait for the next item. Returns None when all sources are done and buffer is empty.

        Cancellation-safe: cancellation can only occur at the await point,
        where no item has been dequeued yet.
        """
        while True:
            if self._error is not None:
                raise self._error
            if self._buffer:
                return self._buffer.popleft()
            if not self._sources:
                return None
            self._notify.clear()
            await self._notify.wait()

    @property
    def is_active(self) -> bool:
        """True if there are buffered items or any source is still running."""
        return bool(self._buffer) or bool(self._sources)

    async def wait(self) -> None:
        """Wait for all source tasks to complete. Raises if any source failed."""
        if self._sources:
            await asyncio.gather(*list(self._sources), return_exceptions=True)
        if self._error is not None:
            raise self._error
