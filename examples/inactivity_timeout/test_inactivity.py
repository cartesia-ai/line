"""
Test script for the inactivity timeout feature.

Connects to a running agent via WebSocket and verifies that the agent
re-prompts after silence. Sends proper agent_state events to simulate
TTS finishing, which triggers the inactivity timer.

Usage:
    1. Start the agent:   uv run python main.py
    2. Run the test:      uv run python test_inactivity.py
"""

import asyncio
import json
import time

import httpx
import websockets

AGENT_URL = "http://localhost:8000"


async def create_session() -> str:
    """POST /chats and return the websocket path."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AGENT_URL}/chats",
            json={
                "call_id": f"test-inactivity-{int(time.time() * 1000)}",
                "from_": "+15551234567",
                "to": "agent",
                "agent_call_id": "test-inactivity",
                "agent": {"inactivity_timeout_ms": 5000},
            },
        )
        resp.raise_for_status()
        return resp.json()["websocket_url"]


async def send_json(ws, msg: dict):
    """Send a JSON message over WebSocket."""
    await ws.send(json.dumps(msg))


async def recv_all(ws, timeout: float = 15.0) -> list[dict]:
    """Receive all messages until a quiet period."""
    messages = []
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            messages.append(msg)
            # Use short timeout after first message to drain remaining chunks
            timeout = 2.0
        except asyncio.TimeoutError:
            break
    return messages


def extract_agent_text(messages: list[dict]) -> str:
    """Extract concatenated agent text from messages."""
    return "".join(msg.get("content", "") for msg in messages if msg.get("type") == "message")


async def ack_agent_speech(ws, messages: list[dict]):
    """Send agent_speech ack-backs and agent_state events for agent messages."""
    text_messages = [m for m in messages if m.get("type") == "message"]
    if not text_messages:
        return

    # Agent starts speaking
    await send_json(ws, {"type": "agent_state", "value": "speaking"})

    # Ack each text chunk
    for msg in text_messages:
        await send_json(ws, {"type": "agent_speech", "content": msg["content"]})

    # Agent finishes speaking
    await send_json(ws, {"type": "agent_state", "value": "idle"})


async def send_user_message(ws, text: str):
    """Send a user message with the full 3-event protocol."""
    await send_json(ws, {"type": "user_state", "value": "speaking"})
    await send_json(ws, {"type": "message", "content": text})
    await send_json(ws, {"type": "user_state", "value": "idle"})


async def test_inactivity_timeout():
    """Test that the agent re-prompts after user silence."""
    print("=" * 60)
    print("TEST: Inactivity Timeout")
    print("=" * 60)

    ws_path = await create_session()
    ws_url = AGENT_URL.replace("http://", "ws://") + ws_path

    async with websockets.connect(ws_url) as ws:
        # 1. Wait for agent introduction
        print("\n[1] Waiting for agent introduction...")
        intro_msgs = await recv_all(ws, timeout=10.0)
        intro_text = extract_agent_text(intro_msgs)
        print(f"    Agent: {intro_text}")
        assert intro_text, "Agent should send an introduction"

        # Ack the introduction (including agent_state events)
        await ack_agent_speech(ws, intro_msgs)

        # 2. Send a user message
        print("\n[2] Sending user message...")
        await send_user_message(ws, "Hello, how are you?")
        print("    User: Hello, how are you?")

        # 3. Wait for agent response
        print("\n[3] Waiting for agent response...")
        response_msgs = await recv_all(ws, timeout=15.0)
        response_text = extract_agent_text(response_msgs)
        print(f"    Agent: {response_text}")
        assert response_text, "Agent should respond to user"

        # Ack the response (sends agent_state idle -> starts inactivity timer)
        await ack_agent_speech(ws, response_msgs)

        # 4. Stay silent and wait for inactivity timeout re-prompt
        print("\n[4] Staying silent... waiting for inactivity timeout (5s)...")
        start = time.time()
        timeout_msgs = await recv_all(ws, timeout=15.0)
        elapsed = time.time() - start
        timeout_text = extract_agent_text(timeout_msgs)
        print(f"    Agent (after {elapsed:.1f}s): {timeout_text}")

        if timeout_text:
            print(f"\n    PASS: Agent re-prompted after {elapsed:.1f}s of silence")
        else:
            print(f"\n    FAIL: No re-prompt received after {elapsed:.1f}s")

        await ws.close()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_inactivity_timeout())
