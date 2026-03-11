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


async def _aiter_then_raise(*items):
    for item in items:
        yield item
    raise ValueError("source failed")


# ------------------------------------------------------------------
# get()
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_single_source():
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
async def test_get_multiple_sources():
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
async def test_get_no_sources_returns_none():
    q = BackgroundQueue[int]()
    assert await q.get() is None


@pytest.mark.asyncio
async def test_get_raises_on_queued_error():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter_then_raise(1, 2))
    assert await q.get() == 1
    assert await q.get() == 2
    with pytest.raises(ValueError, match="source failed"):
        await q.get()


@pytest.mark.asyncio
async def test_get_cancellation_does_not_cancel_source():
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
# get_nowait()
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_nowait_pops_one_item():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter(1, 2, 3))
    await q.wait()
    assert q.get_nowait() == 1
    assert q.get_nowait() == 2
    assert q.get_nowait() == 3
    assert q.get_nowait() is None


@pytest.mark.asyncio
async def test_get_nowait_empty_returns_none():
    q = BackgroundQueue[int]()
    assert q.get_nowait() is None


@pytest.mark.asyncio
async def test_get_nowait_raises_on_queued_error():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter_then_raise(10, 20))
    # Let source run to completion
    for _ in range(10):
        await asyncio.sleep(0)
    assert q.get_nowait() == 10
    assert q.get_nowait() == 20
    with pytest.raises(ValueError, match="source failed"):
        q.get_nowait()


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
    items = []
    while (item := q.get_nowait()) is not None:
        items.append(item)
    assert sorted(items) == [1, 2]


@pytest.mark.asyncio
async def test_wait_raises_on_source_error():
    q = BackgroundQueue[int]()
    q.subscribe(_aiter_then_raise())
    with pytest.raises(ValueError, match="source failed"):
        await q.wait()


@pytest.mark.asyncio
async def test_wait_with_no_sources():
    q = BackgroundQueue[int]()
    await q.wait()  # should not hang


# ------------------------------------------------------------------
# is_active
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_active():
    q = BackgroundQueue[int]()
    assert not q.is_active

    q.subscribe(_aiter(1))
    assert q.is_active

    await q.wait()
    # Source done, but items still buffered
    assert q.is_active

    q.get_nowait()  # pop the single item
    assert not q.is_active
