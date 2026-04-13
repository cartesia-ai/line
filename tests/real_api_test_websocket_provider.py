#!/usr/bin/env python3
"""Ad-hoc integration test: force a ``previous_response_not_found`` by injecting
a fake response_id into the provider's history, then verify the retry logic
recovers and completes the request.

Usage:
    OPENAI_API_KEY=... uv run python tests/test_ws_response_not_found.py
"""

import asyncio
import os
import sys

from loguru import logger

from line.llm_agent.config import LlmConfig, _normalize_config
from line.llm_agent.provider import Message
from line.llm_agent.websocket_provider import _WebSocketProvider

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


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
