#!/usr/bin/env python3
"""Ad-hoc integration tests for the WebSocket provider.

Tests:
    1. previous_response_not_found recovery: inject a fake response_id and
       verify the retry logic recovers.
    2. WebSocket model config: verify temperature, tools, and reasoning_effort
       are accepted by the real API through the WebSocket provider.

Usage:
    OPENAI_API_KEY=... uv run python tests/real_api_test_websocket_provider.py
"""

import asyncio
import os
import sys
from typing import Annotated

from loguru import logger

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.provider import Message
from line.llm_agent.tools.decorators import loopback_tool
from line.llm_agent.websocket_provider import _WebSocketProvider


@loopback_tool
async def get_weather(ctx, city: Annotated[str, "City name"]) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


logger.enable("line")


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    model = "gpt-5-nano"
    config = _normalize_config(LlmConfig(system_prompt="You are a helpful assistant. Be brief."))
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort=None)

    try:
        # -- Turn 1: normal request to establish real history --
        messages = [Message(role="user", content="Say 'hello' and nothing else.")]
        text = ""
        async for chunk in provider.chat(messages, config=config):
            if chunk.text:
                text += chunk.text

        print(f"Turn 1 response: {text.strip()!r}")
        print(f"History length after turn 1: {len(provider._history)}")
        print(f"Response IDs in history: {[rid for _, rid in provider._history if rid]}")

        # -- Corrupt history: replace every response_id with a fake one --
        provider._history = [
            (identity, "resp_FAKE_000" if rid else None) for identity, rid in provider._history
        ]
        print("\nCorrupted history — all response_ids replaced with 'resp_FAKE_000'")

        # -- Turn 2: this will send previous_response_id="resp_FAKE_000" --
        # The API should reject it, the provider should retry from scratch.
        messages.append(Message(role="assistant", content=text))
        messages.append(Message(role="user", content="Now say 'goodbye' and nothing else."))

        text2 = ""
        async for chunk in provider.chat(messages, config=config):
            if chunk.text:
                text2 += chunk.text

        print(f"\nTurn 2 response: {text2.strip()!r}")
        print(f"History length after turn 2: {len(provider._history)}")
        print(f"Response IDs in history: {[rid for _, rid in provider._history if rid]}")

        if text2.strip():
            print("\nPASS: retry after previous_response_not_found succeeded")
            return 0
        else:
            print("\nFAIL: no response text on turn 2")
            return 1

    finally:
        await provider.aclose()


async def test_websocket_model_config():
    """Verify the WebSocket provider accepts temperature, tools, and reasoning_effort.

    Catches issues like the API rejecting temperature as a decimal, or tools/
    reasoning_effort being silently dropped.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1

    model = "gpt-5.2"
    failures = []

    # -- Test 1: temperature as a float (e.g. 0.7) ----------------------------
    # The WebSocket API does not accept temperature as a decimal — it closes
    # the connection. This test verifies we get an error, not a silent success.
    print("\n" + "=" * 60)
    print(f"Test: WebSocket rejects temperature=0.7 ({model})")
    print("=" * 60)

    config = _normalize_config(
        LlmConfig(
            system_prompt="You are a helpful assistant. Be brief.",
            temperature=0.7,
        )
    )
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort=None)
    try:
        messages = [Message(role="user", content="Say 'hello' and nothing else.")]
        text = ""
        async for chunk in provider.chat(messages, config=config):
            if chunk.text:
                text += chunk.text
        if text.strip():
            print(f"FAIL: expected rejection but got response: {text.strip()!r}")
            failures.append("temperature=0.7 (expected rejection)")
        else:
            print("FAIL: empty response instead of error")
            failures.append("temperature=0.7 (expected rejection)")
    except (RuntimeError, Exception) as e:
        print(f"PASS: temperature=0.7 correctly rejected: {type(e).__name__}: {e}")
    finally:
        await provider.aclose()

    # -- Test 2: temperature=0 (edge case: integer-like) -----------------------
    print("\n" + "=" * 60)
    print(f"Test: WebSocket rejects temperature=0 ({model})")
    print("=" * 60)

    config = _normalize_config(
        LlmConfig(
            system_prompt="You are a helpful assistant. Be brief.",
            temperature=0.0,
        )
    )
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort=None)
    try:
        messages = [Message(role="user", content="Say 'hello' and nothing else.")]
        text = ""
        async for chunk in provider.chat(messages, config=config):
            if chunk.text:
                text += chunk.text
        if text.strip():
            print(f"FAIL: expected rejection but got response: {text.strip()!r}")
            failures.append("temperature=0 (expected rejection)")
        else:
            print("FAIL: empty response instead of error")
            failures.append("temperature=0 (expected rejection)")
    except (RuntimeError, Exception) as e:
        print(f"PASS: temperature=0 correctly rejected: {type(e).__name__}: {e}")
    finally:
        await provider.aclose()

    # -- Test 3: reasoning_effort ----------------------------------------------
    print("\n" + "=" * 60)
    print(f"Test: WebSocket completion with reasoning_effort=low ({model})")
    print("=" * 60)

    config = _normalize_config(
        LlmConfig(
            system_prompt="You are a helpful assistant. Be brief.",
            reasoning_effort="low",
        )
    )
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort="low")
    try:
        messages = [Message(role="user", content="Say 'hello' and nothing else.")]
        text = ""
        async for chunk in provider.chat(messages, config=config):
            if chunk.text:
                text += chunk.text
        if text.strip():
            print(f"PASS: got response with reasoning_effort=low: {text.strip()!r}")
        else:
            print("FAIL: empty response with reasoning_effort=low")
            failures.append("reasoning_effort=low")
    except Exception as e:
        print(f"FAIL: reasoning_effort=low raised {type(e).__name__}: {e}")
        failures.append("reasoning_effort=low")
    finally:
        await provider.aclose()

    # -- Test 4: tools ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Test: WebSocket completion with tools ({model})")
    print("=" * 60)

    config = _normalize_config(
        LlmConfig(
            system_prompt="You are a helpful assistant. Use the get_weather tool when asked about weather.",
        )
    )
    provider = _WebSocketProvider(model=model, api_key=api_key, default_reasoning_effort=None)
    try:
        messages = [Message(role="user", content="What's the weather in Tokyo?")]
        got_tool_call = False
        async for chunk in provider.chat(messages, tools=[get_weather], config=config):
            if chunk.tool_calls:
                got_tool_call = True
        if got_tool_call:
            print("PASS: model issued a tool call via WebSocket")
        else:
            print("FAIL: no tool call received")
            failures.append("tools")
    except Exception as e:
        print(f"FAIL: tools raised {type(e).__name__}: {e}")
        failures.append("tools")
    finally:
        await provider.aclose()

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"DONE — {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    else:
        print("All WebSocket model config tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    if exit_code == 0:
        exit_code = asyncio.run(test_websocket_model_config())
    sys.exit(exit_code)
