"""
Tests for BackgroundQueue.

uv run pytest tests/test_llm_agent_background_queue.py -v
"""

import asyncio

import pytest

from line.llm_agent.background_queue import BackgroundQueue


async def _aiter(*items):
    for item in items:
        yield item


# ------------------------------------------------------------------
# Basic functionality
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_source():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter(1, 2, 3))
    results = []
    while True:
        item = await q.get()
        if item is None:
            break
        results.append(item)
    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_multiple_sources():
    q = BackgroundQueue[str]()
    q.subscribe(_aiter("a", "b"))
    q.subscribe(_aiter("c", "d"))
    results = []
    while True:
        item = await q.get()
        if item is None:
            break
        results.append(item)
    assert sorted(results) == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_no_sources_returns_none_immediately():
    q = BackgroundQueue[int]()
    assert await q.get() is None


@pytest.mark.asyncio
async def test_get_nowait_drains_buffer():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter(1, 2, 3))
    # Let the source task run to completion
    await q.wait()
    items = q.get_nowait()
    assert items == [1, 2, 3]
    # Second call returns empty
    assert q.get_nowait() == []


@pytest.mark.asyncio
async def test_is_active():
    q = BackgroundQueue[int]()
    assert not q.is_active

    q.subscribe(_aiter(1))
    assert q.is_active

    await q.wait()
    # Source done, but items still buffered
    assert q.is_active

    q.get_nowait()
    assert not q.is_active


# ------------------------------------------------------------------
# Error propagation
# ------------------------------------------------------------------


async def _aiter_then_raise(*items):
    for item in items:
        yield item
    raise ValueError("source failed")


@pytest.mark.asyncio
async def test_error_propagated_via_get():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter_then_raise(1, 2))
    assert await q.get() == 1
    assert await q.get() == 2
    with pytest.raises(ValueError, match="source failed"):
        await q.get()


@pytest.mark.asyncio
async def test_error_propagated_via_wait():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter_then_raise())
    with pytest.raises(ValueError, match="source failed"):
        await q.wait()


@pytest.mark.asyncio
async def test_error_does_not_prevent_buffered_items():
    """Items buffered before the error should still be retrievable via get_nowait."""
    q = BackgroundQueue[int]()
    q.subscribe(_aiter_then_raise(10, 20))
    await asyncio.sleep(0)  # let source run
    await asyncio.sleep(0)
    # Source has yielded items and then failed
    # wait() should raise
    with pytest.raises(ValueError):
        await q.wait()
    # But items yielded before the error are in the buffer
    items = q.get_nowait()
    assert 10 in items and 20 in items


# ------------------------------------------------------------------
# Cancellation safety
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_cancellation_does_not_cancel_source():
    """Cancelling a get() call should not cancel the source task."""
    started = asyncio.Event()
    gate = asyncio.Event()

    async def _gated_source():
        started.set()
        await gate.wait()
        yield 42

    q = BackgroundQueue[int]()
    q.subscribe(_gated_source())

    await started.wait()

    # Start a get() and cancel it
    get_task = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    get_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await get_task

    # Source should still be active
    assert q.is_active

    # Ungate the source
    gate.set()
    result = await q.get()
    assert result == 42


# ------------------------------------------------------------------
# wait()
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_completes_when_all_sources_done():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter(1))
    q.subscribe(_aiter(2))
    await q.wait()
    assert not q._sources
    assert sorted(q.get_nowait()) == [1, 2]


@pytest.mark.asyncio
async def test_wait_with_no_sources():
    q = BackgroundQueue[int]()
    await q.wait()  # should not hang
