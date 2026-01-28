"""
Utility functions for the v02 module.
"""

import asyncio
from typing import AsyncIterable, Awaitable, Optional, Tuple, TypeVar

T = TypeVar("T")


async def shield_async_iterable(
    async_gen: AsyncIterable[T],
) -> tuple[AsyncIterable[T], Awaitable[None]]:
    """Iterate over an async iterable with cancellation shielding.

    The underlying async generator is run in a separate task. If this coroutine
    is cancelled while iterating, the task continues running in the background.

    Args:
        async_gen: The async iterable to shield from cancellation.
    Returns:
        An async iterable yielding items from async_gen.
        An awaitable Task representing the background task running the generator, for cleanup
    """
    queue: asyncio.Queue[Tuple[bool, Optional[T]]] = asyncio.Queue()
    # Queue items are (is_sentinel, value). Sentinel = (True, None).

    async def run_generator() -> None:
        try:
            async for item in async_gen:
                await queue.put((False, item))
        finally:
            await queue.put((True, None))

    task = asyncio.create_task(run_generator())

    try:
        while True:
            is_sentinel, item = await asyncio.shield(queue.get())
            if is_sentinel:
                exc = task.exception()
                if exc is not None:
                    raise exc
                return
            yield item  # type: ignore[misc]
    except asyncio.CancelledError:
        # Task continues in background; don't remove from list
        raise


def shield_awaitable(
    awaitable: Awaitable[T],
) -> Awaitable[T]:
    return asyncio.shield(awaitable)
