"""
Tests for ConversationHarness WebSocket message ordering.

This test verifies that concurrent _send() calls preserve message order,
which is critical for correct message delivery to clients.
"""

import asyncio

import pytest

from line.harness import ConversationHarness


class OrderTrackingWebSocket:
    """Mock WebSocket that tracks message order with artificial delay."""

    def __init__(self, delays: list[float]):
        self.received_messages = []
        self.delays = delays  # Different delay per message to force reordering
        self._call_count = 0

    async def send_json(self, data):
        call_index = self._call_count
        self._call_count += 1
        # Apply different delays to each call to force race condition
        await asyncio.sleep(self.delays[call_index % len(self.delays)])
        self.received_messages.append(data)


@pytest.mark.asyncio
async def test_concurrent_sends_preserve_order():
    """
    Test that concurrent _send() calls preserve message order.

    This test FAILS before the fix (messages arrive out of order)
    and PASSES after adding the lock (messages arrive in order).
    """
    # Delays: [0.1, 0.05, 0.01] - first message slowest, third fastest
    # Without a lock, message 3 would arrive before message 1
    mock_ws = OrderTrackingWebSocket(delays=[0.1, 0.05, 0.01])

    harness = ConversationHarness(
        websocket=mock_ws,
        shutdown_event=asyncio.Event(),
    )

    # Spawn concurrent tasks (simulating fire-and-forget from bus.py)
    messages = ["message_1", "message_2", "message_3"]
    tasks = [asyncio.create_task(harness.send_message(msg)) for msg in messages]

    await asyncio.gather(*tasks)

    # Extract content from received messages
    received_order = [msg["content"] for msg in mock_ws.received_messages]

    # This assertion FAILS without the lock, PASSES with the lock
    assert received_order == messages, (
        f"Messages arrived out of order!\n"
        f"Expected: {messages}\n"
        f"Received: {received_order}"
    )
